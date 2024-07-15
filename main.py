import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils import *
from model import ArtifactDetectionNN, EarlyStopping
from dataset import EEGDataset, extract_features


class EEGTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        setup_logging(config['logging']['log_file'], config['logging']['log_level'])
        self.file_path = config['data']['file_path']
        self.dataset = EEGDataset(self.file_path)
        self.train_loader, self.val_loader, self.test_loader = self.split_dataset()
        self.model = ArtifactDetectionNN(self.dataset.features.shape[1]).to(self.device)
        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        os.makedirs(config['checkpoint']['save_path'], exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.early_stopping = EarlyStopping(patience=10, min_delta=0)

    def preprocess_data(self):
        scaler = StandardScaler()
        self.dataset.features = scaler.fit_transform(self.dataset.features)

    def split_dataset(self):
        train_size = int((1 - self.config['data']['test_size'] - self.config['data']['val_size']) * len(self.dataset))
        val_size = int(self.config['data']['val_size'] * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)
        return train_loader, val_loader, test_loader

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        for batch_features, batch_labels in self.train_loader:
            batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_features.float())
            loss = self.criterion(outputs, batch_labels.long())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            all_labels.extend(batch_labels.cpu().numpy())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
        train_acc, train_f1, train_precision, train_recall = calculate_metrics(all_labels, all_preds)
        avg_loss = running_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(train_acc)
        r = f"[Training] Epoch {epoch + 1}/{self.config['training']['num_epochs']}, Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}, " \
            f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}"
        logging.info(r)
        print(r)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_preds = []
        with torch.no_grad():
            for val_features, val_labels in self.val_loader:
                val_features, val_labels = val_features.to(self.device), val_labels.to(self.device)
                val_outputs = self.model(val_features.float())
                loss = self.criterion(val_outputs, val_labels.long())
                val_loss += loss.item()
                all_val_labels.extend(val_labels.cpu().numpy())
                _, val_preds = torch.max(val_outputs, 1)
                all_val_preds.extend(val_preds.cpu().numpy())
        val_acc, val_f1, val_precision, val_recall = calculate_metrics(all_val_labels, all_val_preds)
        avg_val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
        self.val_accuracies.append(val_acc)
        r = f"[Validation] Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, " \
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
        logging.info(r)
        print(r)
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(self.config['checkpoint']['save_path'], 'best_model.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            logging.info(f"Model checkpoint saved at {checkpoint_path}")

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.config['checkpoint']['save_path'], 'best_model.pth')))
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_test_labels = []
        all_test_preds = []
        with torch.no_grad():
            for test_features, test_labels in self.test_loader:
                test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)
                test_outputs = self.model(test_features.float())
                loss = self.criterion(test_outputs, test_labels.long())
                test_loss += loss.item()

                all_test_labels.extend(test_labels.cpu().numpy())
                _, predicted = torch.max(test_outputs.data, 1)
                all_test_preds.extend(predicted.cpu().numpy())
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
        avg_test_loss = test_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        _, test_f1, test_precision, test_recall = calculate_metrics(all_test_labels, all_test_preds)
        r = f"[Test] Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%, F1: {test_f1:.4f}, " \
            f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}"
        logging.info(r)
        print(r)

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.savefig('loss_curves.png')

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.savefig('accuracy_curves.png')
        plt.show()

    def run(self):
        for epoch in tqdm(range(self.config['training']['num_epochs'])):
            self.train_one_epoch(epoch)
            self.validate_one_epoch(epoch)
            self.early_stopping(self.val_losses[-1])
            if self.early_stopping.early_stop:
                logging.info("Early stopping")
                print("Early stopping")
                break
        self.test()
        self.plot_metrics()


if __name__ == "__main__":
    config = load_config()
    trainer = EEGTrainer(config)
    trainer.run()
