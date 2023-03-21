#!/usr/bin/env bash

docker build . \
  --tag joeranbosma/picai_nnunet_train:v1.2 \
  --tag joeranbosma/picai_nnunet_train:latest
