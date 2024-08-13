import argparse
from MLPTrainer import MLPTrainer
import warnings
from ELMTrainer import ELMTrainer
warnings.filterwarnings("ignore")


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--outputpath', type=str, default='./output')
    parser.add_argument('--snr_db', type=float, default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lower_snr', type=float, default=-7)
    parser.add_argument('--higher_snr', type=float, default=6.5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--no_plot', default=True, action='store_false')
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--pca', default=False, action='store_true')
    parser.add_argument('--ica', default=False, action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    config = load_config()
    trainer = None
    if config.model == 'MLP':
        trainer = MLPTrainer(config)
    elif config.model == 'ELM':
        trainer = ELMTrainer(config)
    trainer.run()
