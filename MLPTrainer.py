import os
import datetime
import logging
import pickle
from pathlib import Path
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from models import ArtifactDetectionNN,ArtifactDetectionCNN
from dataset import EEGDataset
from datanoise_combiner import DataNoiseCombiner
from utils import setup_logging, EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

run_datetime = datetime.datetime.now()
plt.rcParams.update({'font.size': 14})


class MLPTrainer:
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self._setup_directories()
        self._setup_logging()
        self._init_data_combiner()
        self._load_datasets()
        self._setup_preprocessing()
        self._init_model()
        self._init_training_components()
        self._init_metrics()

    def _setup_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        return device

    def _setup_directories(self):
        os.makedirs(self.config.save_path, exist_ok=True)
        os.makedirs(self.config.outputpath, exist_ok=True)
        os.makedirs(Path(self.config.outputpath) / Path('cnf_matrices'), exist_ok=True)

    def _setup_logging(self):
        setup_logging(self.config.log_file, self.config.log_level)

    def _init_data_combiner(self):
        DataNoiseCombiner(self.config)

    def _load_datasets(self):
        self.train_dataset = EEGDataset(Path(self.config.datapath) / "train")
        self.val_dataset = EEGDataset(Path(self.config.datapath) / "val")
        self.test_dataset = EEGDataset(Path(self.config.datapath) / "test")

    def _load_test_datasets(self, test_dir):
        test_datasets = {}
        for snr_dir in test_dir.iterdir():
            if snr_dir.is_dir():
                snr_value = snr_dir.name.split(' ')[-1]
                test_datasets[snr_value] = EEGDataset(snr_dir)
        return test_datasets

    def _setup_preprocessing(self):
        if self.config.mode == 'train':
            self._preprocess_data()
        self._load_preprocessing()

    def _init_model(self):
        feature_size = len(self.test_dataset.features[0])
        print(f'Feature shape: {feature_size}')
        if self.config.model == 'MLP':
            self.model = ArtifactDetectionNN(feature_size).to(self.device)
        elif self.config.model == 'CNN':
            self.model = ArtifactDetectionCNN(feature_size).to(self.device)


    def _init_training_components(self):
        self.criterion = MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.early_stopping = EarlyStopping(patience=10, min_delta=0)
        self.train_loader, self.val_loader, self.test_loader = self._split_dataset()

    def _init_metrics(self):
        self.metrics = {
            'train': pd.DataFrame(columns=['MSE', 'MAE', 'R2', 'RMSE', 'MAPE']),
            'val': pd.DataFrame(columns=['MSE', 'MAE', 'R2', 'RMSE', 'MAPE']),
        }
        self.best_val_loss = float('inf')

    def _preprocess_data(self):
        self.train_dataset.features, scaler = self._scale_data(self.train_dataset.features)
        self._save_preprocessor(scaler, 'scaler.pkl')
        self.val_dataset.features = scaler.transform(self.val_dataset.features)
        if self.config.pca:
            self.train_dataset.features, pca = self._apply_pca(self.train_dataset.features)
            self._save_preprocessor(pca, 'pca.pkl')
            self.val_dataset.features = pca.transform(self.val_dataset.features)
        if self.config.ica:
            self.train_dataset.features, ica = self._apply_ica(self.train_dataset.features)
            self._save_preprocessor(ica, 'ica.pkl')
            self.val_dataset.features = ica.transform(self.val_dataset.features)


    def _scale_data(self, features):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        return scaled_features, scaler

    def _apply_ica(self, features):
        ica = FastICA(n_components=80, random_state=10)
        ica_features = ica.fit_transform(features)
        return ica_features, ica

    def _apply_pca(self, features):
        pca = PCA(n_components=0.95)
        pca_features = pca.fit_transform(features)
        return pca_features, pca

    def _save_preprocessor(self, preprocessor, filename):
        with open(os.path.join(self.config.save_path, filename), 'wb') as f:
            pickle.dump(preprocessor, f)

    def _load_preprocessing(self):
        scaler = self._load_preprocessor('scaler.pkl')
        self.test_dataset.features = scaler.transform(self.test_dataset.features)
        if self.config.pca:
            pca = self._load_preprocessor('pca.pkl')
            self.test_dataset.features = pca.transform(self.test_dataset.features)
        if self.config.ica:
            ica = self._load_preprocessor('ica.pkl')
            self.test_dataset.features = ica.transform(self.test_dataset.features)


    def _load_preprocessor(self, filename):
        with open(os.path.join(self.config.save_path, filename), 'rb') as f:
            return pickle.load(f)

    def _split_dataset(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss, all_labels, all_preds = 0.0, [], []

        for batch_features, batch_labels in self.train_loader:
            batch_features, batch_labels = batch_features.to(self.device).float(), batch_labels.to(self.device).float()
            self.optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            all_labels.extend(batch_labels.cpu().numpy())
            all_preds.extend(outputs.cpu().detach().numpy())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        mse, mae, r2, rmse, mape = self.calculate_regression_metrics(np.array(all_labels), np.array(all_preds).flatten())
        logging.info(f'Epoch {epoch} - Train Loss: {epoch_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
        new_row = pd.DataFrame([{ 'Epoch': epoch, 'Loss': epoch_loss, 'MSE': mse, 'MAE': mae, 'R2': r2, 'RMSE': rmse, 'MAPE': mape }])
        self.metrics['train'] = pd.concat([self.metrics['train'], new_row], ignore_index=True)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        val_loss, all_val_labels, all_val_preds = 0.0, [], []

        with torch.no_grad():
            for val_features, val_labels in self.val_loader:
                val_features, val_labels = val_features.to(self.device), val_labels.to(self.device)
                val_outputs = self.model(val_features.float())
                loss = self.criterion(val_outputs, val_labels.float())
                val_loss += loss.item()
                all_val_labels.extend(val_labels.cpu().numpy())
                all_val_preds.extend(val_outputs.cpu().detach().numpy())

        epoch_loss = val_loss / len(self.val_loader.dataset)
        mse, mae, r2, rmse, mape = self.calculate_regression_metrics(np.array(all_val_labels), np.array(all_val_preds).flatten())
        logging.info(f'Epoch {epoch} - Val Loss: {epoch_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
        new_row = pd.DataFrame([{'Epoch': epoch, 'Loss': epoch_loss, 'MSE': mse, 'MAE': mae, 'R2': r2, 'RMSE': rmse, 'MAPE': mape}])
        self.metrics['val'] = pd.concat([self.metrics['val'], new_row], ignore_index=True)

        self.scheduler.step(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint()

    def plot_metrics(self):
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        metrics_to_plot = ['Loss', 'MSE', 'MAE', 'R2', 'RMSE', 'MAPE']
        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i // 3, i % 3]
            ax.plot(self.metrics['train']['Epoch'], self.metrics['train'][metric], label='Train')
            ax.plot(self.metrics['val']['Epoch'], self.metrics['val'][metric], label='Validation')
            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
        plt.tight_layout()
        plt.show()

    def calculate_regression_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mse, mae, r2, rmse, mape

    def _save_checkpoint(self):
        checkpoint_path = os.path.join(self.config.save_path, 'best_model.pth')
        torch.save(self.model, checkpoint_path)
        logging.info(f"Model checkpoint saved at {checkpoint_path}")

    def test(self):
        self.model.eval()
        self._load_best_model()
        test_losses, all_test_labels, all_test_preds = [], [], []

        with torch.no_grad():
            for test_features, test_labels in self.test_loader:
                test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)
                test_outputs = self.model(test_features.float())
                test_loss = self.criterion(test_outputs, test_labels.float())
                test_losses.append(test_loss.item())
                all_test_labels.extend(test_labels.cpu().numpy())
                all_test_preds.extend(test_outputs.cpu().detach().numpy())

        test_loss = np.mean(test_losses)

        mse, mae, r2, rmse, mape = self.calculate_regression_metrics(np.array(all_test_labels), np.array(all_test_preds).flatten())
        logging.info(f'Test Loss: {test_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')




    def _load_best_model(self):
        self.model = torch.load(os.path.join(self.config.save_path, 'best_model.pth'))
        self.model.to(self.device)


    def run(self):
        if self.config.mode == 'train':
            self._train()
            self.plot_metrics()
            self.test()
        elif self.config.mode == 'test':
            self.test()

    def _train(self):
        for epoch in tqdm(range(self.config.num_epochs)):
            self.train_one_epoch(epoch)
            self.validate_one_epoch(epoch)
            if self.early_stopping(self.metrics['val']['Loss'].iloc[-1]):
                logging.info("Early stopping")
                print("Early stopping")
                break