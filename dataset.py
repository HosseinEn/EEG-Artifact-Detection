from torch.utils.data import Dataset
from scipy.io import loadmat
from feature_extraction import extract_features

class EEGDataset(Dataset):
    def __init__(self, file_path):
        mat = loadmat(file_path)
        X = mat['x']
        labels = mat['y'].flatten()
        features = extract_features(X)
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

