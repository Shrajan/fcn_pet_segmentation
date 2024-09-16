FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git

# Configure Git, clone the nnUNet repository.
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/Shrajan/nnUNet.git /opt/algorithm/nnunet/ && \
    cd /opt/algorithm/nnunet/ && \
    pip3 install -e /opt/algorithm/nnunet

# Configure Git, clone the dynamic architecture repository.
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/Shrajan/dynamic-network-architectures /opt/algorithm/dynamic-network-architectures/ && \
    cd /opt/algorithm/dynamic-network-architectures/ && \
    pip3 install -e /opt/algorithm/dynamic-network-architectures

# Configure Git, clone the batchgenerators repository.
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/MIC-DKFZ/batchgeneratorsv2.git /opt/algorithm/batchgeneratorsv2/ && \
    cd /opt/algorithm/batchgeneratorsv2/ && \
    pip3 install -e /opt/algorithm/batchgeneratorsv2

RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/automated-petct-lesion-segmentation  && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm nnUNet_results /opt/algorithm/nnUNet_results

RUN python -m pip install --user -U pip && \
    python -m pip install --user -r requirements.txt && \
    mkdir -p /opt/algorithm/nnUNet_raw && \
    mkdir -p /opt/algorithm/nnUNet_preprocessed && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result

ENV nnUNet_raw="/opt/algorithm/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"

ENTRYPOINT ["python", "-m", "process", "$0", "$@"]
