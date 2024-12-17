from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import termcolor

COLS = {
    0: 'WN',
    1: 'EOG',
    2: 'EMG'
}


k = 10
X_train = np.load('data/train/X.npy')
y_train = np.load('data/train/Y.npy')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

mse_scores, mae_scores, r2_scores = [], [], []
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for train_index, val_index in kf.split(X_scaled):
    X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Define and train SVM model for this fold
    # 'linear', 'poly', 'rbf', 'sigmoid'
    model = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1))
    model.fit(X_train_fold, y_train_fold)

    # Predict and evaluate
    y_val_pred = model.predict(X_val_fold)
    mse_scores.append(mean_squared_error(y_val_fold, y_val_pred))
    mae_scores.append(mean_absolute_error(y_val_fold, y_val_pred))
    r2_scores.append(r2_score(y_val_fold, y_val_pred))

    for i in range(3):
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

    print(f"MSE: {mse_scores[-1]}, MAE: {mae_scores[-1]}, R2: {r2_scores[-1]}")


print(f"Average MSE over {k} folds: {np.mean(mse_scores)} ± {np.std(mse_scores)}")
print(f"Average MAE over {k} folds: {np.mean(mae_scores)} ± {np.std(mae_scores)}")
print(f"Average R2 over {k} folds: {np.mean(r2_scores)} ± {np.std(r2_scores)}")
