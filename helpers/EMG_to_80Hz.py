import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, firwin, lfilter
from scipy.io import savemat

def butterworth_filter(data, cutoff, fs, order=10):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def fir_filter(data, cutoff, fs, numtaps=401):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    fir_coeff = firwin(numtaps, normal_cutoff)
    filtered_data = lfilter(fir_coeff, 1.0, data)
    return filtered_data

def power_spectral_density(signal, fs=256):
    freqs, psd = welch(signal, fs=fs)
    return freqs, psd


if __name__ == '__main__':
    emg_signals = sio.loadmat('data/EMG_all_epochs.mat')['EMG_all_epochs']
    cutoff_frequency = 80  # Hz
    sampling_frequency = 256  # Hz
    filtered_emg_signals = np.zeros_like(emg_signals)

    for i in range(emg_signals.shape[0]):
        filtered_emg_signals[i, :] = fir_filter(emg_signals[i, :], cutoff_frequency, sampling_frequency)

    idx = 10

    freqs, psd_before = power_spectral_density(emg_signals[idx, :], fs=sampling_frequency)
    freqs, psd_after = power_spectral_density(filtered_emg_signals[idx, :], fs=sampling_frequency)

    # plt.figure(figsize=(10, 6))
    # plt.semilogy(freqs, psd_before, label='Before Filtering')
    # plt.semilogy(freqs, psd_after, label='After Filtering')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Spectral Density (V^2/Hz)')
    # plt.title('Power Spectral Density of EMG Signal')
    # plt.legend()
    # plt.show()

    savemat('data/filtered80Hz_EMG_all_epochs.mat', {'EMG_all_epochs': filtered_emg_signals})


