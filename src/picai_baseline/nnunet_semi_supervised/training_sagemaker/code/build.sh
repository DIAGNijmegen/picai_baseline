#!/usr/bin/env bash

docker build . \
  --tag joeranbosma/picai_nnunet_train:v1.1 \
  --tag joeranbosma/picai_nnunet_train:latest
