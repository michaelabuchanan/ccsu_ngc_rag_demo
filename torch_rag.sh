#!/bin/bash

pip install -q --root-user-action=ignore transformers sentence-transformers

export HF_HOME='/workspace/local_ws/models'
export TRANFSORMERS_CACHE='workspace/local_ws/models'
export HF_TOKEN=$(cat hf_token.txt)

python ./scripts/torch_rag_script.py "$1"