import os
import pandas as pd
from keras import Input, Model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import keras
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.src.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Add, GlobalAveragePooling1D
import warnings
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def residual_block(x, filters, kernel_size=3, stride=1, use_conv_shortcut=False):
    shortcut = x
    if use_conv_shortcut:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv1D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters, kernel_size, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def build_resnet(input_shape, num_blocks_per_stage=[2, 2, 2, 2]):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, strides=2, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, strides=2, padding='same')(x)

    filter_sizes = [64, 128, 256, 512]
    for i, filters in enumerate(filter_sizes):
        for j in range(num_blocks_per_stage[i]):
            stride = 1
            use_conv_shortcut = False
            if j == 0 and i != 0:
                stride = 2
                use_conv_shortcut = True
            x = residual_block(x, filters=filters, stride=stride, use_conv_shortcut=use_conv_shortcut)

    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(3, activation='linear')(x)

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

    model = build_resnet(input_shape)
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

    for i in range(3):
        mse_col = mean_squared_error(y_val_fold[:, i], y_val_pred[:, i])
        mae_col = mean_absolute_error(y_val_fold[:, i], y_val_pred[:, i])
        r2_col = r2_score(y_val_fold[:, i], y_val_pred[:, i])
        rmse_col = np.sqrt(mse_col)
        s = f"Column {i} - MSE: {round(mse_col, 3)}, MAE: {round(mae_col, 3)}, R2: {round(r2_col, 3)}, RMSE: {round(rmse_col, 3)}"
        import termcolor

        print(termcolor.colored(s, 'blue'))
        plt.figure(figsize=(6, 6))
        plt.scatter(y_val_fold[:, i], y_val_pred[:, i], alpha=0.6)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label="Ideal")
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(s)
        plt.legend()
        plt.show()

    print(f"Fold {fold} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, RMSE: {rmse:.4f}")

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
