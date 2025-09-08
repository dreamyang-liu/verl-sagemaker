#!/bin/bash
set -ex

BASE_DIR=/opt/ml/input

WORKSPACE_DIR=$BASE_DIR/project
HUGGINGFACE_CACHE_DIR=$BASE_DIR/huggingface-cache

mkdir -p $WORKSPACE_DIR
mkdir -p $HUGGINGFACE_CACHE_DIR

export HF_HOME=$BASE_DIR/huggingface-cache
export OUTPUT_BASE_PATH=/opt/ml/output

# Check if OUTPUT_BASE_PATH exists
if [ ! -d "$OUTPUT_BASE_PATH" ]; then
    echo "Error: Output directory $OUTPUT_BASE_PATH does not exist!"
    exit 1
fi

cd $WORKSPACE_DIR
git clone https://github.com/volcengine/verl verl-sagemaker
cd verl-sagemaker
pip3 install --no-deps -e .

# Execute the command passed to the script
"$@"
