#!/usr/bin/env bash

./build.sh

docker save picai_swinunetr | gzip -c > picai_swinunetr.tar.gz
