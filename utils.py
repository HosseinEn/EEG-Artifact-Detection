import yaml
import logging
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_file, log_level):
    logging.basicConfig(filename=log_file, level=log_level, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return acc, f1, precision, recall
