import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from keras import Input
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from feature_extraction import extract_features

def lr_scheduler(epoch, lr):
    decay_rate = 0.85
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr

X_train = np.load('data/train/X.npy')
y_train = np.load('data/train/Y.npy')
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)
mse_scores = []
mae_scores = []
r2_scores = []
rmse_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    X_train_fold = extract_features(X_train_fold)
    X_val_fold = extract_features(X_val_fold)
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)
    ica = FastICA(n_components=80)
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_train_fold = pca.fit_transform(X_train_fold)
    X_train_fold = ica.fit_transform(X_train_fold)
    X_val_fold = scaler.transform(X_val_fold)
    X_val_fold = pca.transform(X_val_fold)
    X_val_fold = ica.transform(X_val_fold)
    model = Sequential()
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation=None))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', keras.metrics.R2Score()])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    callbacks_list = [early_stop, LearningRateScheduler(lr_scheduler, verbose=0)]
    history = model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=50, verbose=0,validation_data=(X_val_fold, y_val_fold), callbacks=callbacks_list)
    y_val_pred = model.predict(X_val_fold)
    mse = mean_squared_error(y_val_fold, y_val_pred)
    mae = mean_absolute_error(y_val_fold, y_val_pred)
    r2 = r2_score(y_val_fold, y_val_pred)
    rmse = root_mean_squared_error(y_val_fold, y_val_pred)
    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    print(f"Fold MSE: {mse}, MAE: {mae}, R2: {r2}, RMSE: {rmse}")

print(f"\nAverage MSE over {k} folds: {np.mean(mse_scores)} ± {np.std(mse_scores)}")
print(f"Average MAE over {k} folds: {np.mean(mae_scores)} ± {np.std(mae_scores)}")
print(f"Average R2 over {k} folds: {np.mean(r2_scores)} ± {np.std(r2_scores)}")
print(f"Average RMSE over {k} folds: {np.mean(rmse_scores)} ± {np.std(rmse_scores)}")
scores = pd.DataFrame({'MSE': mse_scores, 'MAE': mae_scores, 'R2': r2_scores, 'RMSE': rmse_scores})
scores = scores.round(3)
scores.to_csv('output/neural_network_scores.csv', index=False)