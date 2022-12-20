FROM joeranbosma/picai_nnunet:latest

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install git+https://github.com/DIAGNijmegen/picai_baseline

ENTRYPOINT $0 $@
