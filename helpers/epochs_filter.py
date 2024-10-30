import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import welch, filtfilt, firwin
from scipy.io import savemat


def custom_bandpass_filter(data, lowcut, highcut, fs,
                           l_trans_bandwidth=0.5,
                           h_trans_bandwidth=0.5,
                           filter_length=101,
                           fir_window='hann',
                           pad_length=100):
    nyquist = 0.5 * fs
    low = (lowcut - l_trans_bandwidth) / nyquist
    high = (highcut + h_trans_bandwidth) / nyquist
    if filter_length % 2 == 0:
        filter_length += 1
    fir_coeff = firwin(filter_length, [low, high], pass_zero=False, window=fir_window)
    padded_data = np.pad(data, (pad_length, pad_length), mode='edge')
    filtered_data = filtfilt(fir_coeff, 1.0, padded_data)
    return filtered_data[pad_length:pad_length + len(data)]

def power_spectral_density(signal, fs=256):
    freqs, psd = welch(signal, fs=fs)
    return freqs, psd

if __name__ == '__main__':
    emg_signals = sio.loadmat('data/EMG_all_epochs.mat')['EMG_all_epochs']
    low_cutoff_frequency = 1
    high_cutoff_frequency = 80
    sampling_frequency = 256
    filtered_emg_signals = np.zeros_like(emg_signals)

    for i in range(emg_signals.shape[0]):
        filtered_emg_signals[i, :] = custom_bandpass_filter(emg_signals[i, :], low_cutoff_frequency, high_cutoff_frequency,
                                                             fs=sampling_frequency)

    savemat('data/filtered80Hz_EMG_all_epochs.mat', {'EMG_all_epochs': filtered_emg_signals})

    idx = 89

    freqs, psd_before = power_spectral_density(emg_signals[idx, :], fs=sampling_frequency)
    freqs, psd_after = power_spectral_density(filtered_emg_signals[idx, :], fs=sampling_frequency)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.semilogy(freqs, psd_before, label='Before filtering')
    plt.semilogy(freqs, psd_after, label='After filtering')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.subplot(1, 2, 2)
    plt.plot(emg_signals[idx, :], label='Before filtering')
    plt.plot(filtered_emg_signals[idx, :], label='After filtering')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()