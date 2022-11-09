# Adapted from https://github.com/DIAGNijmegen/diag-nnunet/blob/master/Dockerfile
FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Configuration
RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "OMP_NUM_THREADS=1" >> /etc/environment

ARG env_det_num_threads=6
ARG env_det_verbose=0
# set env_det_verbose=1 for tqdm progress bars during training

# Setup environment variables
ENV det_data=/home/user/data det_models=/output/models det_num_threads=$env_det_num_threads det_verbose=$env_det_verbose OMP_NUM_THREADS=1

# Install a few dependencies that are not automatically installed
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y \
    git \
    cmake \
    make \
    wget \
    gnupg \
    build-essential \
    software-properties-common \
    gdb \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install \
        graphviz \
        carbontracker && \
    rm -rf ~/.cache/pip

# Create folders
RUN mkdir -p ${det_data}

# Install mibaumgartner code
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt  \
  && pip3 install hydra-core --upgrade --pre \
  && pip3 install git+https://github.com/mibaumgartner/pytorch_model_summary.git

# Install nnDetection
RUN git clone https://github.com/MIC-DKFZ/nnDetection /opt/code/nnDetection
COPY ./consolidate.py /opt/code/nnDetection/scripts/consolidate.py
COPY ./predict.py /opt/code/nnDetection/scripts/predict.py
COPY ./utils.py /opt/code/nnDetection/scripts/utils.py
RUN FORCE_CUDA=1 pip3 install -v -e /opt/code/nnDetection

# updating requests and urllib3 fixed compatibility
RUN pip install numpy \
  && pip install --upgrade requests \
  && pip install --upgrade urllib3

# Patch shutil copy operations that don't work on SOL
COPY shutil_sol.py /tmp/shutil_sol.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mkdir -p "$SITE_PKG/shutil_sol" && \
    mv /tmp/shutil_sol.py "$SITE_PKG/shutil_sol/__init__.py" && \
    find "$SITE_PKG/nnunet" -type f -name "*.py" \
        -exec sed -i.bak \
            -e "s|import shutil|import shutil, shutil_sol|g" \
            -e "s|shutil.copy(|shutil_sol.copyfile(|g" \
            -e "s|shutil.copytree(|shutil_sol.copytree(|g" {} +

# Copy utility scripts
COPY ./nndet_generate_detection_maps.py /opt/code/nndet_generate_detection_maps.py

# Configure entrypoint
COPY --chown=user nndetection_wrapper.py /opt/code/nndetection_wrapper.py
RUN chmod +x /opt/code/nndetection_wrapper.py && \
    ln -s /opt/code/nndetection_wrapper.py /usr/local/bin/nndet
