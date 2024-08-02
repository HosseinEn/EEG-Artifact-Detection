import datetime

import termcolor
from model import ExtremeLearningMachine
from dataset import EEGDataset
from datanoise_combiner import DataNoiseCombiner
from tqdm import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from utils import *
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

run_datetime = datetime.datetime.now()




class ELMTrainer:
    def __init__(self, config):
        self.config = config
        os.makedirs(config.save_path, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        setup_logging(config.log_file, config.log_level)
        DataNoiseCombiner(config)
        self.train_dataset = EEGDataset(Path(self.config.datapath) / "train")
        self.val_dataset = EEGDataset(Path(self.config.datapath) / "val")
        self.test_datasets = {}
        test_dir = Path(self.config.datapath) / "test"
        for snr_dir in test_dir.iterdir():
            if snr_dir.is_dir():
                snr_value = snr_dir.name.split(' ')[-1]
                self.test_datasets[snr_value] = EEGDataset(snr_dir)
        if self.config.mode == 'train':
            self.preprocess_data()
            self.load_preprocessing()
        else:
            self.load_preprocessing()
        feature_size = next(iter(self.test_datasets.values())).features.shape[1]
        print(f'Feature shape: {feature_size}')
        self.train_loader, self.val_loader = self.split_dataset()
        self.model = ExtremeLearningMachine(feature_size, 128, 3).to(self.device)
        self.train_accuracies = []
        self.val_accuracies = []
        self.criterion = CrossEntropyLoss()
        self.snr_value = self.config.snr_db
        self.mode = config.mode
        os.makedirs(config.outputpath, exist_ok=True)

    def preprocess_data(self):
        scaler = StandardScaler()
        self.train_dataset.features = scaler.fit_transform(self.train_dataset.features)
        with open(os.path.join(self.config.save_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        self.val_dataset.features = scaler.transform(self.val_dataset.features)

        if self.config.ica:
            ica = FastICA(n_components=80, random_state=10)
            self.train_dataset.features = ica.fit_transform(self.train_dataset.features)
            with open(os.path.join(self.config.save_path, 'ica.pkl'), 'wb') as f:
                pickle.dump(ica, f)
            self.val_dataset.features = ica.transform(self.val_dataset.features)

        if self.config.pca:
            pca = PCA(n_components=0.95)
            self.train_dataset.features = pca.fit_transform(self.train_dataset.features)
            with open(os.path.join(self.config.save_path, 'pca.pkl'), 'wb') as f:
                pickle.dump(pca, f)
            self.val_dataset.features = pca.transform(self.val_dataset.features)

    def load_preprocessing(self):
        for snr, test_dataset in self.test_datasets.items():
            with open(os.path.join(self.config.save_path, 'scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)
                test_dataset.features = scaler.transform(test_dataset.features)
            if self.config.ica:
                with open(os.path.join(self.config.save_path, 'ica.pkl'), 'rb') as f:
                    ica = pickle.load(f)
                    test_dataset.features = ica.transform(test_dataset.features)
            if self.config.pca:
                with open(os.path.join(self.config.save_path, 'pca.pkl'), 'rb') as f:
                    pca = pickle.load(f)
                    test_dataset.features = pca.transform(test_dataset.features)

    def split_dataset(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train_loader, val_loader

    def train(self):
        X_train = torch.tensor(self.train_dataset.features, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(self.train_dataset.labels, dtype=torch.float32).to(self.device)

        self.model.train_elm(X_train, y_train)
        model_path = os.path.join(self.config.save_path, 'best_model.pth')
        torch.save(self.model.state_dict(), model_path)

    def validate(self):
        X_val = torch.tensor(self.val_dataset.features, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(self.val_dataset.labels, dtype=torch.float32).to(self.device)

        y_pred = self.model.predict(X_val)

        val_acc = accuracy_score(y_val.cpu().numpy(), y_pred.cpu().numpy())
        val_f1 = f1_score(y_val.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')
        val_precision = precision_score(y_val.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')
        val_recall = recall_score(y_val.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

        r = f"[Validation] Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
        logging.info(r)
        print(r)
        return val_acc, val_f1, val_precision, val_recall

    def test(self):
        test_accuracies = []
        snr_values = []
        model_path = os.path.join(self.config.save_path, 'best_model.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}. Please train the model first.")

        # sort test_datasets by its string snr values as key
        self.test_datasets = dict(sorted(self.test_datasets.items(), key=lambda x: float(x[0])))
        for snr_value, test_dataset in self.test_datasets.items():
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(self.device)
            self.model.eval()
            test_loss = 0.0
            all_test_labels = []
            all_test_preds = []
            with torch.no_grad():
                for test_features, test_labels in test_loader:
                    test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)
                    test_outputs = self.model(test_features.float()).to(self.device)
                    loss = self.criterion(test_outputs, test_labels)
                    test_loss += loss.item()
                    _, test_preds = torch.max(test_outputs, 1)
                    all_test_labels.extend(test_labels.cpu().numpy())
                    all_test_preds.extend(test_preds.cpu().numpy())
            test_acc = accuracy_score(all_test_labels, all_test_preds)
            test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
            test_precision = precision_score(all_test_labels, all_test_preds, average='weighted')
            test_recall = recall_score(all_test_labels, all_test_preds, average='weighted')
            test_accuracies.append(test_acc)
            snr_values.append(snr_value)
            avg_test_loss = test_loss / len(test_loader)
            r = f"[Test] SNR: {snr_value}, Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, " \
                f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}"
            logging.info(r)
            print(r)
            res_path = os.path.join(self.config.outputpath, f'results{run_datetime}.csv')
            with open(res_path, 'a') as f:
                if os.stat(res_path).st_size == 0:
                    f.write('SNR,Accuracy,F1,Precision,Recall\n')
                f.write(f'{snr_value},{test_acc},{test_f1},{test_precision},{test_recall}\n')

        # plot the accuracy based on SNR values
        plt.figure(figsize=(10, 5))
        plt.plot(snr_values, test_accuracies, marker='o', color='b')
        plt.xlabel('SNR [dB]')
        plt.xticks(snr_values)
        plt.ylabel('Test accuracy')
        plt.yticks(np.arange(0.6, 1.05, 0.05))
        plt.title('Relationship between SNR and classification accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(self.config.outputpath, 'snr_accuracy.png'))
        if self.config.no_plot:
            plt.show()

    def plot_metrics(self):
        plt.figure(figsize=(20, 5))
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curves - SNR: {self.snr_value}')
        plt.legend()
        plt.savefig(os.path.join(self.config.outputpath, f'combined_curves-snr{self.snr_value}.png'))
        if self.config.no_plot:
            plt.show()

    def run(self):
        if self.config.mode == 'train':
            self.train()
            self.plot_metrics()
            self.test()
        elif self.config.mode == 'test':
            self.test()
