import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os

pattern = re.compile(r'results.*.csv')

def filter_files(files, labels):
    return [f for f in files if f[0] in labels]

def plot_snr_res(outputpath):
    guide_df = pd.read_csv('./guide.csv')
    labels = guide_df['Description'].tolist()
    files = guide_df['Filename'].tolist()
    files = [(l, f) for l, f in zip(labels, files)]
    plt.figure(figsize=(12, 6))
    for l, file in files:
        df = pd.read_csv(os.path.join(outputpath, file))
        plt.plot(df['SNR'], df['Accuracy'], marker='o', label=l)
    plt.xlabel('SNR (dB)')
    plt.xticks(np.arange(-7, 6.5, 0.5))
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0.6, 1.05, 0.05))
    plt.title('Accuracy vs SNR')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_snr_res('./')