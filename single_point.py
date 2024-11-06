import os

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
# from xgboost import XGBRegressorv
from feature_extraction import extract_features
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import tensorflow.keras.backend as K
from sklearn.metrics import r2_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




def plot_metric(history, metric, subplot_position, title):
    plt.subplot(2, 2, subplot_position)
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(title)
    plt.ylabel(title.split(' ')[-1])
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

def lr_scheduler(epoch, lr):
    decay_rate = 0.85
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr

X_train = np.load('data/train/X.npy')
y_train = np.load('data/train/Y.npy')
X_train = extract_features(X_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = Sequential()

model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(512, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation=None))

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', keras.metrics.R2Score()])
model.summary()

checkpoint_name = 'weights/Weights-{epoch:03d}--{val_loss:.5f}.keras'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


callbacks_list = [early_stop, LearningRateScheduler(lr_scheduler, verbose=1)]

history = model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1, validation_split=0.2, callbacks=callbacks_list)

plt.figure(figsize=(20, 10))
metrics = [
    ('mse', 'Model MSE'),
    ('mae', 'Model MAE'),
    ('mape', 'Model MAPE'),
    ('r2_score', 'Model R2')
]
for idx, (metric, title) in enumerate(metrics, start=1):
    plot_metric(history, metric, idx, title)
plt.show()

X_test = np.load('data/test/X.npy')
y_test = np.load('data/test/Y.npy')
X_test = extract_features(X_test)
X_test = scaler.transform(X_test)


y_pred = model.predict(X_test)

r2_metric = keras.metrics.R2Score()
r2_metric.update_state(y_true=y_test, y_pred=y_pred)
print(mean_absolute_error(y_test, y_pred), r2_metric.result().numpy())