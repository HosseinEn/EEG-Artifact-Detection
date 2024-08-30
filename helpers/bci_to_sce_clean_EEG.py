import numpy as np
import scipy.io as sio
from scipy.signal import welch
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from mne_icalabel import *

data = sio.loadmat('/home/hossein/Archived University4022/Project/EEG/workspace/SCEADNN/data/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1a_1000Hz.mat')
orig_data = sio.loadmat('/home/hossein/Archived University4022/Project/EEG/workspace/SCEADNN/data/EEG_all_epochs.mat')['EEG_all_epochs']
eeg_signals = data['cnt']

# to uV
# eeg_signals = eeg_signals * 0.1

ch_names = [ch_name[0] for ch_name in data['nfo']['clab'][0][0][0]]
raw = mne.io.RawArray(eeg_signals.T, info=mne.create_info(ch_names=ch_names, sfreq=data['nfo']['fs'][0][0][0], ch_types='eeg'))

raw.pick_channels([ch for ch in raw.ch_names if ch not in ['CFC7', 'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8']])


montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore', match_case=False)
raw.set_eeg_reference('average', projection=True)
raw.resample(256)
raw.filter(1, 120, fir_design='firwin')
raw.notch_filter(np.arange(50, 60, 50), picks='all')

ica = ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
labels = label_components(raw, ica, method='iclabel')
artifact_indices = [i for i, label in enumerate(labels['labels']) if label in ['eye', 'muscle', 'heart', 'line_noise', 'channel_noise', 'other']]
ica.exclude = artifact_indices
ica.apply(raw)

epoch_length = 512
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
data = epochs.get_data()
data = data.reshape(-1, data.shape[2])

data = data[np.random.choice(data.shape[0], 4514, replace=False)]

sio.savemat('/data/filtered120Hz_EEG_all_epochs.mat', {'EEG_all_epochs': data})