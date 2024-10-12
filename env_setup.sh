#/bin/bash

MODE=$1

rm -rf output
if [ "$MODE" = "train" ];
then
    rm -rf checkpoints
    rm -rf data/train data/test data/val
fi
