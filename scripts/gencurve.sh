#!/bin/bash

if [ $# != 2 ]; then
    echo "Usage: <bits> <file>"
    exit 1
fi

echo -e "load('gencurve.sage')\ngenerate_ecdlp($1,'$2')" | sage
