import numpy as np
import scipy.io as sio
from scipy.signal import filtfilt, firwin


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


noise_shape = (4514, 512)

noise = np.random.normal(0, 1, noise_shape)

filtered_noise = np.zeros_like(noise)
for i in range(noise_shape[0]):
    filtered_noise[i] = custom_bandpass_filter(noise[i], 1, 80, 256)

sio.savemat('data/white_noise.mat', {'white_noise': filtered_noise})