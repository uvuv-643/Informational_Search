#!/bin/bash
# Launch SPLADEv2 training as a DataSphere Job (ir_datasets variant)
# Data is downloaded directly via ir_datasets — no prepare_data.py needed.
# Note: first run may take 20-30 min to download MS MARCO data.

set -euo pipefail

source ~/.bash_profile

datasphere -t $YC_OAUTH_TOKEN project job execute -p bt1ab5gdi0fklq9dt7oh -c config.yaml
