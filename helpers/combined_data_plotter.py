import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

def power_spectral_density(signal, fs=256):
    freqs, psd = welch(signal, fs=fs)
    return freqs, psd

def data_with_noise(data, labels, noise_no, n):
    index = np.where(labels == noise_no)[0][n]
    return data[index], index

n_comp = 80
index = np.random.random_integers(1,700)

random_state = 42

data = np.load('data/test/snr -7.0/X.npy')
data_label = np.load('data/test/snr -7.0/Y.npy')

clean = data_with_noise(data, data_label, 0, index)
eog = data_with_noise(data, data_label, 1, index)
emg = data_with_noise(data, data_label, 2, index)

# calculate the power spectral density
freqs, psd = power_spectral_density(clean[0])
# plt.subplot(2, 1, 1)
plt.title('Log PSD for three signals contaminated with noise, SNR=-7.0', fontsize=14)
plt.semilogy(freqs, psd, label='Clean')
plt.legend()

freqs, psd = power_spectral_density(eog[0])
# plt.subplot(3, 1, 2)
plt.xticks(range(0,256,10))
plt.semilogy(freqs, psd, label='EEG+EOG')
plt.legend()

freqs, psd = power_spectral_density(emg[0])
# plt.subplot(2, 1, 2)
plt.semilogy(freqs, psd, label='EEG+EMG')
plt.xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
plt.ylabel('Log PSD (V^2/Hz)', fontsize=14, fontweight='bold')

# for SNR in [-6.0, 6.0]:
#     data_snr = np.load(f'data/test/snr {str(SNR)}/X.npy')
#     data_snr_label = np.load(f'data/test/snr {str(SNR)}/Y.npy')
#
#     ica = FastICA(n_components=n_comp, random_state=random_state)
#     res = ica.fit_transform(data_snr)
#
#     data_clean, real_clean_idx = data_with_noise(data_snr, data_snr_label, 0, index)
#     data_eog, real_eog_idx = data_with_noise(data_snr, data_snr_label, 1, index)
#     data_emg, real_emg_idx = data_with_noise(data_snr, data_snr_label, 2, index)
#
#     plt.subplot(2, 1, 1)
#     plt.title('Signal - EMG Data')
#     plt.plot(data_emg, label='Original - SNR: ' + str(SNR))
#     plt.legend()
#
#     plt.subplot(2, 1, 2)
#     plt.title('ICA')
#     plt.plot(res[real_emg_idx], label=f'n_components={n_comp} - SNR: {str(SNR)}')
#     plt.legend()

plt.legend(fontsize=14)
plt.show()
