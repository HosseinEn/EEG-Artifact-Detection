rm -rf output
rm -rf checkpoints
rm -rf data/train data/test data/val
python3 main.py  --ica --pca --model MLP