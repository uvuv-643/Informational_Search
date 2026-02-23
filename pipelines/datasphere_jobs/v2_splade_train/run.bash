#!/bin/bash
# Launch SPLADEv2 training as a DataSphere Job
#
# Prerequisites:
#   1. Run: python prepare_data.py   (creates triplets.jsonl.gz ~400MB)
#   2. Ensure triplets.jsonl.gz is in this directory

set -euo pipefail

# Check data file exists
if [ ! -f "triplets.jsonl.gz" ]; then
    echo "ERROR: triplets.jsonl.gz not found!"
    echo "Run first: python prepare_data.py"
    exit 1
fi

echo "Data file: $(ls -lh triplets.jsonl.gz)"

source ~/.bash_profile

datasphere -t $YC_OAUTH_TOKEN project job execute -p bt1ab5gdi0fklq9dt7oh -c config.yaml
