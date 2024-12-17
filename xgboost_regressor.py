import os
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
np.random.seed(812)

COLS = {
    0: 'WN',
    1: 'EOG',
    2: 'EMG'
}

X_train = np.load('data/train/X.npy')
y_train = np.load('data/train/Y.npy')
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

mse_scores = []
mae_scores = []
r2_scores = []
rmse_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_val_fold = scaler.transform(X_val_fold)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        booster='gbtree',
        eta=0.1,
        max_depth=8,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        alpha=0.1,
        lambda_=0.1,
        random_state=42
    )

    model.fit(X_train_fold, y_train_fold)

    y_val_pred = model.predict(X_val_fold)

    mse = mean_squared_error(y_val_fold, y_val_pred)
    mae = mean_absolute_error(y_val_fold, y_val_pred)
    r2 = r2_score(y_val_fold, y_val_pred)
    rmse = np.sqrt(mse)

    for i in range(3):
        mse_col = mean_squared_error(y_val_fold[:, i], y_val_pred[:, i])
        mae_col = mean_absolute_error(y_val_fold[:, i], y_val_pred[:, i])
        r2_col = r2_score(y_val_fold[:, i], y_val_pred[:, i])
        rmse_col = np.sqrt(mse_col)
        s = f"Column {i} - MSE: {round(mse_col, 3)}, MAE: {round(mae_col, 3)}, R2: {round(r2_col, 3)}, RMSE: {round(rmse_col, 3)}"

        print(s)
        plt.figure(figsize=(6, 6))
        plt.scatter(y_val_fold[:, i], y_val_pred[:, i], alpha=0.6)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label="Ideal")
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(s)
        plt.legend()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.scatter(y_val_fold[:, i], y_val_fold[:, i] - y_val_pred[:, i], alpha=0.6)
        plt.plot([y_train.min(), y_train.max()], [0, 0], 'r--', label="Ideal")
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.title(f"Residuals for Column {i}")
        plt.legend()
        plt.show()

    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    print(f"Fold MSE: {mse}, MAE: {mae}, R2: {r2}, RMSE: {rmse}")

# Print average scores
print(f"\nAverage MSE over {k} folds: {np.mean(mse_scores).round(3)}")
print(f"Average MAE over {k} folds: {np.mean(mae_scores).round(3)}")
print(f"Average R2 over {k} folds: {np.mean(r2_scores).round(3)}")
print(f"Average RMSE over {k} folds: {np.mean(rmse_scores).round(3)}")

scores = pd.DataFrame({'MSE': mse_scores, 'MAE': mae_scores, 'R2': r2_scores, 'RMSE': rmse_scores})
scores = scores.round(3)
scores.to_csv('output/xgboost_scores.csv', index=False)
