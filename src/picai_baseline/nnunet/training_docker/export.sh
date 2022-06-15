#!/usr/bin/env bash

./build.sh

docker save picai_nnunet | gzip -c > picai_nnunet.tar.gz
