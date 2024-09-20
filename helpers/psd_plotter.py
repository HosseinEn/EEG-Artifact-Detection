import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter
import scipy.io as sio

# /home/hossein/University4022/Project/EEG/workspace/SCEADNN/data/EEG_all_epochs.mat
# /home/hossein/University4022/Project/EEG/workspace/SCEADNN/data/EOG_all_epochs.mat
# /home/hossein/University4022/Project/EEG/workspace/SCEADNN/data/EMG_all_epochs.mat

# show the power of each using psd welch
def power_spectral_density(signal, fs=256):
    freqs, psd = welch(signal, fs=fs)
    return freqs, psd

# show the power of one sample using psd welch

if __name__ == '__main__':
    idx = 1000
    # load the data
    eeg_signals = sio.loadmat('data/EEG_all_epochs')['EEG_all_epochs']
    eog_signals = sio.loadmat('data/EOG_all_epochs.mat')['EOG_all_epochs']
    emg_signals = sio.loadmat('data/filtered80Hz_EMG_all_epochs.mat')['EMG_all_epochs']

    # show the power of each using psd welch
    freqs, psd = power_spectral_density(eeg_signals[idx])
    plt.semilogy(freqs, psd, label='EEG')

    freqs, psd = power_spectral_density(eog_signals[idx])
    plt.semilogy(freqs, psd, label='EOG')

    freqs, psd = power_spectral_density(emg_signals[idx])
    plt.semilogy(freqs, psd, label='EMG')

    # plt.title('Random Samples of Signals in EEGDenoiseNet dataset', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    plt.ylabel('Log PSD (V^2/Hz)', fontsize=14, fontweight='bold')
    plt.xlim([1,85])
    plt.legend(fontsize=14)
    plt.show()
