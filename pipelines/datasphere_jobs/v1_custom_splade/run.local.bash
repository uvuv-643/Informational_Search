#!/bin/bash

python main.py \
    --collection 'collection.tar.gz' \
    --model 'model.pt' \
    --max-queries 15 \
    --batch-size 32 \
    --output 'metrics.json'