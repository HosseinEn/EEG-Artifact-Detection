from torch.utils.data import Dataset
from feature_extraction import extract_features
from datanoise_combiner import DataNoiseCombiner
from numpy import array

class EEGDataset(Dataset):
    def __init__(self, config):
        a = DataNoiseCombiner(file_path=config.datapath, snr_db=config.snr_db)
        clean, noise, X, y = [], [], [], []
        for i, s in enumerate(a):
            clean.append(s[0])
            noise.append(s[1])
            X.append(s[2])
            y.append(s[3])

        y = array(y)

        features = extract_features(X)
        self.features = features
        self.labels = y.flatten()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

