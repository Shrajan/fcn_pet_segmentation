import os, json, SimpleITK, torch, gc, nnunetv2
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Union, Tuple
import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch._dynamo import OptimizedModule
from torch.backends import cudnn
from tqdm import tqdm


class CustomPredictor(nnUNetPredictor):
    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None):
        torch.set_num_threads(7)
        with torch.no_grad():
            self.network = self.network.to(self.device)
            self.network.eval()

            if self.verbose:
                print('preprocessing')
            preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)
            data, _ = preprocessor.run_case_npy(input_image, None, image_properties,
                                                self.plans_manager,
                                                self.configuration_manager,
                                                self.dataset_json)

            data = torch.from_numpy(data)
            del input_image
            if self.verbose:
                print('predicting')

            predicted_logits = self.predict_preprocessed_image(data)

            if self.verbose: print('Prediction done')

            segmentation = self.convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits,
                                                                                            image_properties)
        return segmentation

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(join(model_training_output_dir, f'fold_{f}', checkpoint_name))

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @torch.inference_mode(mode=True)
    def predict_preprocessed_image(self, image):
        empty_cache(self.device)
        data_device = torch.device('cpu')
        predicted_logits_device = torch.device('cpu')
        gaussian_device = torch.device('cpu')
        compute_device = torch.device('cuda:0')

        data, slicer_revert_padding = pad_nd_image(image, self.configuration_manager.patch_size,
                                                   'constant', {'value': 0}, True,
                                                   None)
        del image

        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

        empty_cache(self.device)

        data = data.to(data_device)
        predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                       dtype=torch.half,
                                       device=predicted_logits_device)
        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                    value_scaling_factor=10,
                                    device=gaussian_device, dtype=torch.float16)

        if not self.allow_tqdm and self.verbose:
            print(f'running prediction: {len(slicers)} steps')

        for p in self.list_of_parameters:
            # network weights have to be updated outside autocast!
            # we are loading parameters on demand instead of loading them upfront. This reduces memory footprint a lot.
            # each set of parameters is only used once on the test set (one image) so run time wise this is almost the
            # same
            self.network.load_state_dict(torch.load(p, map_location=compute_device)['network_weights'])
            with torch.autocast(self.device.type, enabled=True):
                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    pred = self._internal_maybe_mirror_and_predict(data[sl][None].to(compute_device))[0].to(
                        predicted_logits_device)
                    pred /= (pred.max() / 100)
                    predicted_logits[sl] += (pred * gaussian)
                del pred
        empty_cache(self.device)
        return predicted_logits

    def convert_predicted_logits_to_segmentation_with_correct_shape(self, predicted_logits, props):
        old = torch.get_num_threads()
        torch.set_num_threads(7)

        # resample to original shape
        spacing_transposed = [props['spacing'][i] for i in self.plans_manager.transpose_forward]
        current_spacing = self.configuration_manager.spacing if \
            len(self.configuration_manager.spacing) == \
            len(props['shape_after_cropping_and_before_resampling']) else \
            [spacing_transposed[0], *self.configuration_manager.spacing]
        predicted_logits = self.configuration_manager.resampling_fn_probabilities(predicted_logits,
                                                                                  props[
                                                                                      'shape_after_cropping_and_before_resampling'],
                                                                                  current_spacing,
                                                                                  [props['spacing'][i] for i in
                                                                                   self.plans_manager.transpose_forward])

        segmentation = None
        pp = None
        try:
            with torch.no_grad():
                pp = predicted_logits.to('cuda:0')
                segmentation = pp.argmax(0).cpu()
                del pp
        except RuntimeError:
            del segmentation, pp
            torch.cuda.empty_cache()
            segmentation = predicted_logits.argmax(0)
        del predicted_logits

        # segmentation may be torch.Tensor but we continue with numpy
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()

        # put segmentation in bbox (revert cropping)
        segmentation_reverted_cropping = np.zeros(props['shape_before_cropping'],
                                                  dtype=np.uint8 if len(
                                                      self.label_manager.foreground_labels) < 255 else np.uint16)
        slicer = bounding_box_to_slice(props['bbox_used_for_cropping'])
        segmentation_reverted_cropping[slicer] = segmentation
        del segmentation

        # revert transpose
        segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(self.plans_manager.transpose_backward)
        torch.set_num_threads(old)
        return segmentation_reverted_cropping


class Autopet_baseline:

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        # according to the specified grand-challenge interfaces
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"
        self.nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"
        self.output_path_category = "/output/data-centric-model.json"
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.result_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))
        
    
    def predict_tumourseg(self, im, prop, tumourseg_trained_model, tumourseg_folds):
        # initialize predictors
        pred_tumourseg = CustomPredictor(
            tile_step_size=0.5,
            use_mirroring=True,
            use_gaussian=True,
            perform_everything_on_device=False,
            allow_tqdm=True
        )
        pred_tumourseg.initialize_from_trained_model_folder(
            tumourseg_trained_model,
            use_folds=tumourseg_folds,
            checkpoint_name='checkpoint_final.pth'
        )

        tumourseg_pred = pred_tumourseg.predict_single_npy_array(
            im, prop, None
        )
        torch.cuda.empty_cache()
        gc.collect()
        return tumourseg_pred

    def predict(self):
        print("nnUNet segmentation starting!")
        
        rw = SimpleITKIO()
        input_fnames = [os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
                        os.path.join(self.nii_path, "TCIA_001_0001.nii.gz")
                        ]
        output_fname = os.path.join(self.result_path, self.nii_seg_file)
        
        im, prop = rw.read_images(input_fnames)
        
        tumourseg_trained_model = "/opt/algorithm/nnUNet_results/Dataset222_Autopet3_FCN/nnUNetTrainer__nnUNetResEncUNetLPlans_torchres__3d_fullres/"
        tumourseg_folds = [0,1]

        with torch.no_grad():
            tumourseg_pred = self.predict_tumourseg(im, prop, tumourseg_trained_model, tumourseg_folds)
            torch.cuda.empty_cache()
            gc.collect()

        # now merge all tumours.
        tumourseg_pred = np.where(tumourseg_pred>0, 1, 0)

        # now save
        rw.write_seg(tumourseg_pred, output_fname, prop)
        print("Prediction finished")

    def save_datacentric(self, value: bool):
        print("Saving datacentric json to " + self.output_path_category)
        with open(self.output_path_category, "w") as json_file:
            json.dump(value, json_file, indent=4)

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.save_datacentric(False)
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()
