#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

gcc -O2 -shared -fPIC \
    fusion_mechanism.c \
    -lm \
    -o libfusion.so
