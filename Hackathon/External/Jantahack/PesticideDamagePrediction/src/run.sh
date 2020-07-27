#!/bin/bash

export TRAINING_DATA=../tmp/train_folds.csv
export FOLD=0
export TRAIN_FILE_PATH=../input/train.csv
export TEST_FILE_PATH=../input/test.csv
export MODEL=$1
export MLFLOW_TRACKING_URI=sqlite:////Users/saurabhrai/Data-Science-Projects/Hackathon/External/Jantahack/PesticideDamagePrediction/databases/mlflow.db
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
echo training completed
python3 -m predict
echo prediction completed