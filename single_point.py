from keras.callbacks import ModelCheckpoint, EarlyStopping
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
import tensorflow.keras.backend as K

def r2_keras(y_true, y_pred):
    residual = K.sum(K.square(y_true - y_pred))
    total = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - residual / (total + K.epsilon())
    return r2

X_train = np.load('data/train/X.npy')
y_train = np.load('data/train/Y.npy')
X_train = extract_features(X_train)

scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)

model = Sequential()

# model.add(Dense(128, input_dim = X_train.shape[1], activation='relu'))
# model.add(Dropout(0.6))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation=None))

# add convolutional layers
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



model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', r2_keras])
model.summary()

checkpoint_name = 'weights/Weights-{epoch:03d}--{val_loss:.5f}.keras'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

callbacks_list = [earlystop]

history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2, callbacks=callbacks_list)

# Plot training & validation metrics in one image and different axes - MSE, MAE, MAPE, RMSE, R2
plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(2, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(2, 2, 3)
plt.plot(history.history['mape'])
plt.plot(history.history['val_mape'])
plt.title('Model MAPE')
plt.ylabel('MAPE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(2, 3, 4)
plt.plot(history.history['r2_keras'])
plt.plot(history.history['val_r2_keras'])
plt.title('Model R2')
plt.ylabel('R2')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower left')

plt.show()

X_test = np.load('data/test/X.npy')
y_test = np.load('data/test/Y.npy')
X_test = extract_features(X_test)

X_test = scaler.transform(X_test)


y_pred = model.predict(X_test)


print(mean_absolute_error(y_test, y_pred))