import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

def power_spectral_density(signal, fs=512):
    freqs, psd = welch(signal, fs=fs)
    return freqs, psd

def data_with_noise(data, labels, noise_no, n):
    index = np.where(labels == noise_no)[0][n]
    return data[index], index

n_comp = 80
index = 100

random_state = 42

for SNR in [-6.0, 6.0]:
    data_snr = np.load(f'data/test/snr {str(SNR)}/X.npy')
    data_snr_label = np.load(f'data/test/snr {str(SNR)}/Y.npy')

    ica = FastICA(n_components=n_comp, random_state=random_state)
    res = ica.fit_transform(data_snr)

    data_clean, real_clean_idx = data_with_noise(data_snr, data_snr_label, 0, index)
    data_eog, real_eog_idx = data_with_noise(data_snr, data_snr_label, 1, index)
    data_emg, real_emg_idx = data_with_noise(data_snr, data_snr_label, 2, index)

    plt.subplot(2, 1, 1)
    plt.title('Signal - EMG Data')
    plt.plot(data_emg, label='Original - SNR: ' + str(SNR))
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('ICA')
    plt.plot(res[real_emg_idx], label=f'n_components={n_comp} - SNR: {str(SNR)}')
    plt.legend()

plt.legend()
plt.show()
