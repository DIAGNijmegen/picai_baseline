# Adapted from https://github.com/DIAGNijmegen/diag-nnunet/blob/master/Dockerfile
FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Configuration
RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "OMP_NUM_THREADS=1" >> /etc/environment

# Install a few dependencies that are not automatically installed, plus nnU-net
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/DIAGNijmegen/nnUNet.git /home/user/nnunet && \
    git -C /home/user/nnunet checkout 1.7.0-3 && \
    chown -R user /home/user/nnunet && \
    pip3 install \
        -e /home/user/nnunet \
        graphviz \
        git+https://github.com/nanohanno/hiddenlayer.git@bugfix/get_trace_graph#egg=hiddenlayer \
        carbontracker && \
    rm -rf ~/.cache/pip

# Install MSK-Tiger IO subset
COPY io.py /tmp/io.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/io.py "$SITE_PKG/nnunet/utilities/io.py"


# Install PI-CAI repositories
RUN pip3 install \
    picai_eval>=1.4.4 \
    picai_prep>=2.1.2 \
    picai_baseline>=0.8.1 \
    mlxtend==0.19.0

# copy custom files to nnUNet folder
COPY nnUNetTrainerV2_focalLoss.py /tmp/nnUNetTrainerV2_focalLoss.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/nnUNetTrainerV2_focalLoss.py "$SITE_PKG/nnunet/training/network_training/nnUNet_variants/loss_function/nnUNetTrainerV2_focalLoss.py"

COPY nnUNetTrainerV2_Loss_CE_checkpoints.py /tmp/nnUNetTrainerV2_Loss_CE_checkpoints.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/nnUNetTrainerV2_Loss_CE_checkpoints.py "$SITE_PKG/nnunet/training/network_training/nnUNetTrainerV2_Loss_CE_checkpoints.py"

COPY nnUNetTrainerV2_Loss_FL_and_CE.py /tmp/nnUNetTrainerV2_Loss_FL_and_CE.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/nnUNetTrainerV2_Loss_FL_and_CE.py "$SITE_PKG/nnunet/training/network_training/nnUNetTrainerV2_Loss_FL_and_CE.py"

COPY run_training.py /tmp/run_training.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/run_training.py "$SITE_PKG/nnunet/run/run_training.py"

# Configure entrypoint
COPY --chown=user nnunet_wrapper.py /home/user/nnunet_wrapper.py
RUN chmod +x /home/user/nnunet_wrapper.py && \
    ln -s /home/user/nnunet_wrapper.py /usr/local/bin/nnunet
