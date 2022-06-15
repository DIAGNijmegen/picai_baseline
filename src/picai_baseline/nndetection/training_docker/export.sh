#!/usr/bin/env bash

./build.sh

docker save picai_nndetection | gzip -c > picai_nndetection.tar.gz
