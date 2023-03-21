FROM joeranbosma/picai_nnunet:1.7.0-customized-v1.14

COPY --chown=user train.py /opt/algorithm/train.py
RUN chmod +x /opt/algorithm/train.py && \
    ln -s /opt/algorithm/train.py /usr/local/bin/train
