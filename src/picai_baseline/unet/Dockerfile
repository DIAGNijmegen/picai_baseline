FROM joeranbosma/picai_nnunet:latest

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/DIAGNijmegen/picai_baseline

ENTRYPOINT $0 $@
