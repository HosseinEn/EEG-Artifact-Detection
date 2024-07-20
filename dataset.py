from torch.utils.data import Dataset
from scipy.io import loadmat
from feature_extraction import extract_features
from data_module import EEGDenoiseDM
from numpy import array

class EEGDataset(Dataset):
    def __init__(self, file_path, config):
        # mat = loadmat(file_path)
        # X = mat['x']
        # labels = mat['y'].flatten()
        a = EEGDenoiseDM(config)

        clean, noise, X, y = [], [], [], []
        for s in a.val:
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

