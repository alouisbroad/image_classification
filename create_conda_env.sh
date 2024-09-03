#!/bin/bash

ENV_NAME="classification"
REQUIREMENTS_FILE="requirements.txt"

conda create -n $ENV_NAME python=3.12 -y
source activate $ENV_NAME
conda install --file $REQUIREMENTS_FILE -y

echo "Conda environment '$ENV_NAME' created and requirements installed."
