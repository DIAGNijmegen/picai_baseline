FROM joeranbosma/picai_nnunet:latest

COPY --chown=user train.py /home/user/train.py
RUN chmod +x /home/user/train.py && \
    ln -s /home/user/train.py /usr/local/bin/train
