import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
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
    # load the data
    eeg_signals = sio.loadmat('data/EEG_all_epochs.mat')['EEG_all_epochs']
    eog_signals = sio.loadmat('data/EOG_all_epochs.mat')['EOG_all_epochs']
    emg_signals = sio.loadmat('data/EMG_all_epochs.mat')['EMG_all_epochs']

    # show the power of each using psd welch
    freqs, psd = power_spectral_density(eeg_signals[0])
    plt.semilogy(freqs, psd, label='EEG')

    freqs, psd = power_spectral_density(eog_signals[0])
    plt.semilogy(freqs, psd, label='EOG')

    freqs, psd = power_spectral_density(emg_signals[0])
    plt.semilogy(freqs, psd, label='EMG')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Log PSD')
    plt.legend()
    plt.show()
