import os
from keras.src.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Attention, BatchNormalization, \
    Bidirectional, LeakyReLU
from keras.src.models import Model
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import keras
import warnings
import termcolor

from feature_extraction import extract_features

COLS = {
    0: 'EOG',
    1: 'EMG'
}


warnings.filterwarnings("ignore")

keras.utils.set_random_seed(812)


def lr_scheduler(epoch, lr):
    decay_rate = 0.85
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr

def build_cnn_lstm(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv1D(16, 3)(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(32, 3)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 3)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)


    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)

    x = Flatten()(x)
    x = Dense(128, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = LeakyReLU()(x)
    outputs = Dense(2, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

X_train = np.load('data/train/X.npy')
y_train = np.load('data/train/Y.npy')
X_train = extract_features(X_train)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
mse_scores = []
mae_scores = []
r2_scores = []
rmse_scores = []


for train_index, val_index in kf.split(X_train):
    feature_size = X_train.shape[1]
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_val_fold = scaler.transform(X_val_fold)
    X_train_fold = X_train_fold.reshape(-1, feature_size, 1)
    X_val_fold = X_val_fold.reshape(-1, feature_size, 1)

    inputs = Input(shape=(feature_size, 1))

    model = build_cnn_lstm((feature_size, 1))

    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', keras.metrics.R2Score()])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    callbacks_list = [
        early_stop,
        LearningRateScheduler(lr_scheduler, verbose=0),
        ModelCheckpoint('checkpoints/best_model.keras', monitor='val_loss', save_best_only=True)
    ]


    history = model.fit(
        X_train_fold,
        y_train_fold,
        epochs=100,
        batch_size=16,
        verbose=0,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=callbacks_list
    )

    y_val_pred = model.predict(X_val_fold)
    mse = mean_squared_error(y_val_fold, y_val_pred)
    mae = mean_absolute_error(y_val_fold, y_val_pred)
    r2 = r2_score(y_val_fold, y_val_pred)
    rmse = np.sqrt(mse)

    for i in range(2):
        mse_col = mean_squared_error(y_val_fold[:, i], y_val_pred[:, i])
        mae_col = mean_absolute_error(y_val_fold[:, i], y_val_pred[:, i])
        r2_col = r2_score(y_val_fold[:, i], y_val_pred[:, i])
        rmse_col = np.sqrt(mse_col)
        s = f"{COLS[i]} - MSE: {round(mse_col, 3)}, MAE: {round(mae_col, 3)}, R2: {round(r2_col, 3)}, RMSE: {round(rmse_col, 3)}"

        print(termcolor.colored(s, 'blue'))
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_val_fold[:, i], y_val_pred[:, i], alpha=0.6)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label="Ideal")
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        # plt.title(s)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(y_val_fold[:, i], y_val_fold[:, i] - y_val_pred[:, i], alpha=0.6, color='g')
        plt.plot([y_train.min(), y_train.max()], [0, 0], 'r--', label="Ideal")
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.title(f"Residuals for {COLS[i]}")
        plt.legend()
        plt.show()


    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    print(f"Fold MSE: {mse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}, RMSE: {rmse:.3f}")
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()

print(f"\nAverage MSE over {k} folds: {np.mean(mse_scores).round(3)}")
print(f"Average MAE over {k} folds: {np.mean(mae_scores).round(3)}")
print(f"Average R2 over {k} folds: {np.mean(r2_scores).round(3)}")
print(f"Average RMSE over {k} folds: {np.mean(rmse_scores).round(3)}")
scores = pd.DataFrame({'MSE': mse_scores, 'MAE': mae_scores, 'R2': r2_scores, 'RMSE': rmse_scores})
scores = scores.round(3)
scores.to_csv('output/cnn_scores', index=False)