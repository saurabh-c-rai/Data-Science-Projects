#!/bin/bash

export TRAINING_DATA=/Users/raisaurabh04/OneDrive/GreyAtom/MachineLearning/AirlinesHackathon/input/train_folds.csv
export FOLD=0

python -m src.create_folds
echo create_folds over
python -m src.train