#!/bin/bash

export TRAINING_DATA=/Users/raisaurabh04/Downloads/GitHub/Data-Science-Projects/AirlinesHackathon/input/train_folds.csv
export FOLD=0
export TRAIN_FILE_PATH=/Users/raisaurabh04/Downloads/GitHub/Data-Science-Projects/AirlinesHackathon/input/train.csv
export TEST_FILE_PATH=/Users/raisaurabh04/Downloads/GitHub/Data-Science-Projects/AirlinesHackathon/input/test.csv
export MODEL=$1

# echo "Choose a FOLD value for validation set"
# read FOLD

# export FOLD


python3 -m create_folds
echo create_folds over
FOLD=0 python3 -m train
FOLD=1 python3 -m train
FOLD=2 python3 -m train
FOLD=3 python3 -m train
FOLD=4 python3 -m train
python3 -m predict