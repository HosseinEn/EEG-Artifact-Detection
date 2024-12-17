import os
import pandas as pd
from keras import Input, Model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import keras
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.src.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Add, \
    GlobalAveragePooling1D, ReLU
import warnings
import matplotlib.pyplot as plt
import termcolor

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

COLS = {
    0: 'EOG',
    1: 'EMG'
}




def conv_block_1d(x, filters, kernel_size, strides=1):
    """1D Convolutional Block"""
    x = Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def identity_block_1d(x, filters, kernel_size, stride=1):
    """Residual Block with Channel Projection (if needed)"""
    shortcut = x
    # If the input and output channels don't match, use a 1x1 Conv1D for the shortcut
    if x.shape[-1] != filters or stride != 1:
        shortcut = Conv1D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv1D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x


def resnet_18_1d(input_shape=(512, 1), num_classes=1):
    inputs = Input(shape=input_shape)

    # Initial Convolutional Layer
    x = conv_block_1d(inputs, 64, kernel_size=7, strides=2)

    # ResNet Blocks
    x = identity_block_1d(x, 64, kernel_size=3)
    x = identity_block_1d(x, 64, kernel_size=3)
    x = identity_block_1d(x, 128, kernel_size=3)
    x = identity_block_1d(x, 128, kernel_size=3)
    x = identity_block_1d(x, 256, kernel_size=3)
    x = identity_block_1d(x, 256, kernel_size=3)
    x = identity_block_1d(x, 512, kernel_size=3)
    x = identity_block_1d(x, 512, kernel_size=3)

    # Global Average Pooling and Dense Layer
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='linear')(x)

    model = Model(inputs, outputs)
    return model



def lr_scheduler(epoch, lr):
    decay_rate = 0.85
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr

X_train = np.load('data/train/X.npy')
y_train = np.load('data/train/Y.npy')
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
mse_scores = []
mae_scores = []
r2_scores = []
rmse_scores = []

input_shape = (X_train.shape[1], 1)

for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_val_fold = scaler.transform(X_val_fold)
    X_train_fold = X_train_fold.reshape(-1, input_shape[0], 1)
    X_val_fold = X_val_fold.reshape(-1, input_shape[0], 1)

    model = resnet_18_1d(input_shape=input_shape, num_classes=2)
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamw",
    )

    model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae', keras.metrics.RootMeanSquaredError()])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=0)
    # reduce_lr = LearningRateScheduler(lr_scheduler)
    checkpoint = ModelCheckpoint(f'checkpoints/best_model_fold{fold}.keras', monitor='val_loss', save_best_only=True, verbose=0)
    callbacks_list = [early_stop, reduce_lr, checkpoint]

    history = model.fit(
        X_train_fold,
        y_train_fold,
        epochs=100,
        batch_size=16,
        verbose=1,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=callbacks_list
    )

    y_val_pred = model.predict(X_val_fold)
    mse = mean_squared_error(y_val_fold, y_val_pred)
    mae = mean_absolute_error(y_val_fold, y_val_pred)
    r2 = r2_score(y_val_fold, y_val_pred)
    rmse = np.sqrt(mse)

    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    rmse_scores.append(rmse)

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

    print(f"Fold {fold} - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}, RMSE: {rmse:.2f}")

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()

print(f"\nAverage MSE over {k} folds: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
print(f"Average MAE over {k} folds: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average R2 over {k} folds: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average RMSE over {k} folds: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

scores = pd.DataFrame({
    'MSE': mse_scores,
    'MAE': mae_scores,
    'R2': r2_scores,
    'RMSE': rmse_scores
})
scores = scores.round(4)
scores.to_csv('output/resnet_scores.csv', index=False)
