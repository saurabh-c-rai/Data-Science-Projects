#!/bin/bash

export TRAINING_DATA=/mnt/d/PythonEnvironment/AirlinesHackathon/input/train_folds.csv
export TRAIN_FILE_PATH=/mnt/d/PythonEnvironment/AirlinesHackathon/input/train.csv
export TEST_FILE_PATH=/mnt/d/PythonEnvironment/AirlinesHackathon/input/test.csv
export MODEL=$1
# echo "Choose a FOLD value for validation set"
# read FOLD

# export FOLD


# python3.8 -m create_folds
# FOLD=0 python3.8 -m train
# FOLD=1 python3.8 -m train
# FOLD=2 python3.8 -m train
# FOLD=3 python3.8 -m train
# FOLD=4 python3.8 -m train
python3.8 -m predict