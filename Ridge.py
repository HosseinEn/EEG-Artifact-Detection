from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

k = 10
X_train = np.load('data/train/X.npy')
y_train = np.load('data/train/Y.npy')

mse_scores, mae_scores, r2_scores = [], [], []
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Define and train Ridge model for this fold
    model = Ridge(alpha=0.01)
    # model = Lasso(alpha=0.01)
    model.fit(X_train_fold, y_train_fold)

    # Predict and evaluate
    y_val_pred = model.predict(X_val_fold)
    mse_scores.append(mean_squared_error(y_val_fold, y_val_pred))
    mae_scores.append(mean_absolute_error(y_val_fold, y_val_pred))
    r2_scores.append(r2_score(y_val_fold, y_val_pred))
    print(f"MSE: {mse_scores[-1]}, MAE: {mae_scores[-1]}, R2: {r2_scores[-1]}")

print(f"Average MSE over {k} folds: {np.mean(mse_scores)} ± {np.std(mse_scores)}")
print(f"Average MAE over {k} folds: {np.mean(mae_scores)} ± {np.std(mae_scores)}")
print(f"Average R2 over {k} folds: {np.mean(r2_scores)} ± {np.std(r2_scores)}")