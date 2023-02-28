FROM joeranbosma/picai_nnunet:latest

COPY --chown=user train.py /opt/algorithm/train.py
RUN chmod +x /opt/algorithm/train.py && \
    ln -s /opt/algorithm/train.py /usr/local/bin/train
