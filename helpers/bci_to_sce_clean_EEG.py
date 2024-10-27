import numpy as np
import scipy.io as sio
from scipy.signal import welch
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from mne_icalabel import *
import warnings
warnings.filterwarnings("ignore")

electrode_info = sio.loadmat('./biosemi_template.mat')
data = sio.loadmat('/home/hossein/Archived University4022/Project/EEG/workspace/SCEADNN/data/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1a_1000Hz.mat')

eeg_signals = data['cnt']
ch_names = [ch_name[0] for ch_name in data['nfo']['clab'][0][0][0]]
electrode_data = electrode_info['locs'][0]
labels = [str(entry[0][0]) for entry in electrode_data]
positions = np.array([[float(entry[1]), float(entry[2]), float(entry[3])] for entry in electrode_data])
ch_pos = {labels[idx]: positions[idx] for idx in range(len(labels))}
ch_pos = {ch: pos for ch, pos in ch_pos.items() if ch in ch_names}
missing_channels = [ch for ch in ch_names if ch not in ch_pos]

if missing_channels:
    print(f"Warning: Missing positions for channels {missing_channels}. Removing these channels from data.")
    ch_names = [ch for ch in ch_names if ch in ch_pos]

channel_indices = [i for i, ch in enumerate(data['nfo']['clab'][0][0][0]) if ch[0] in ch_names]
eeg_signals = eeg_signals[:, channel_indices]


raw = mne.io.RawArray(eeg_signals.T, info=mne.create_info(ch_names=ch_names, sfreq=data['nfo']['fs'][0][0][0], ch_types='eeg'))
# raw.pick_channels([ch for ch in raw.ch_names if ch not in ['CFC7', 'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8']])

montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
raw.set_montage(montage, on_missing='ignore', match_case=False)
raw.set_eeg_reference('average', projection=True)
raw.filter(1, 80, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, filter_length='auto', fir_design='firwin', phase='zero-double', fir_window='hann')
raw.notch_filter(60, picks='all')

ica_comp_no = 20
ica = ICA(n_components=ica_comp_no, random_state=97, max_iter='auto', method='fastica')
ica.fit(raw)
labels = label_components(raw, ica, method='iclabel')
# only preserve 'brain' components with high y_pred_proba values (i.e., > 0.8)
ica.exclude = [i for i in range(ica_comp_no) if labels['labels'][i] != 'brain' or labels['y_pred_proba'][i] < 0.9]
ica.apply(raw)

raw.resample(256)

epoch_length = 512
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
data = epochs.get_data()
data = data.reshape(-1, data.shape[2])
data = data[np.random.choice(data.shape[0], 4514, replace=False)]

sio.savemat('data/BCI80Hz_EEG_all_epochs.mat', {'EEG_all_epochs': data})