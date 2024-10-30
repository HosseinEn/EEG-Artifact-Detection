import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 6))

def power_spectral_density(signal, fs=256):
    freqs, psd = welch(signal, fs=fs)
    return freqs, psd

def data_with_noise(data, labels, noise_no, n):
    index = np.where(labels == noise_no)[0][n]
    return data[index], index

n_comp = 80
index = np.random.randint(1,700+1)

random_state = 42

data = np.load('data/test/snr -7.0/X.npy')
data_label = np.load('data/test/snr -7.0/Y.npy')

clean = data_with_noise(data, data_label, 0, index)
eog = data_with_noise(data, data_label, 1, index)
emg = data_with_noise(data, data_label, 2, index)

# calculate the power spectral density
freqs, psd = power_spectral_density(clean[0])
# plt.subplot(2, 1, 1)
# plt.title('Log PSD for three signals contaminated with noise, SNR=-7.0', fontsize=14)
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
plt.xlabel('Frequency (Hz)', fontweight='bold')
plt.ylabel('Log PSD (V^2/Hz)', fontweight='bold')
plt.xlim([1,85])
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(clean[0], label='Clean')
plt.plot(eog[0], label='EOG')
plt.plot(emg[0], label='EMG')
plt.legend()
plt.xlabel('Time (ms)', fontweight='bold')
plt.ylabel('Amplitude (V)', fontweight='bold')
plt.title('Original Signal and Noises', fontweight='bold')
plt.show()