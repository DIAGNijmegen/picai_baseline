#!/usr/bin/env bash

docker build . \
  --tag joeranbosma/picai_nnunet:1.7.0-customized-v1.5 \
  --tag joeranbosma/picai_nnunet:latest
