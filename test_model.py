from trainer import EEGTrainer
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add arguments
    parser.add_argument('--datapath', type=str, default='./data', help='Path to the data folder.')
    parser.add_argument('--outputpath', type=str, default='./output', help='Path to the output folder.')
    parser.add_argument('--snr_db', type=float, default=None, help='Signal-to-noise ratio in dB.')
    parser.add_argument('--test_size', type=float, default=0.25,
                        help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the validation split.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping.')
    parser.add_argument('--log_file', type=str, default='log.txt', help='File to log training progress.')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level.')
    parser.add_argument('--no_plot', default=True, action='store_false', help='Disable plot display.')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Path to save model checkpoints.')
    parser.add_argument('--mode', type=str, default='test', help='Mode: train or test.')

    args = parser.parse_args()

    acc = list()
    for snr in np.arange(-7, 4.5, 0.5):
        config = argparse.Namespace(
            datapath=args.datapath,
            outputpath=args.outputpath,
            snr_db=snr,
            test_size=args.test_size,
            val_size=args.val_size,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            log_file=args.log_file,
            log_level=args.log_level,
            no_plot=args.no_plot,
            save_path=args.save_path,
            mode=args.mode
        )

        # Initialize dataset and model
        trainer = EEGTrainer(config)
        trainer.run()


    print(acc)
