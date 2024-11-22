import seaborn as sns
from keras import *
import numpy as np
from matplotlib import pyplot as plt
# from keras.src.losses import *
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error, mean_absolute_error

# Load keras model
model = models.load_model('checkpoints/best_model.keras')

# Load data
X_test = np.load('data/test/X.npy')
y_test = np.load('data/test/Y.npy')

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

# normalize
snr_range = np.max(y_test) - np.min(y_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual')
plt.legend()
plt.show()

y_pred = y_pred.flatten()
y_test = y_test.flatten()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(6, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.show()

print(f'MSE: {mse}, MAE: {mae}, R2: {r2}, RMSE: {rmse}')
