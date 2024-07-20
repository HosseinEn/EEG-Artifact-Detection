import pandas as pd
import matplotlib.pyplot as plt

file_path = './output/results.csv'
data = pd.read_csv(file_path)

# Check the data
print(data.head())

data.columns = data.columns.str.strip()

data_grouped = data.groupby('SNR').mean(numeric_only=True).reset_index()

plt.figure(figsize=(10, 6))

plt.scatter(data_grouped['SNR'], data_grouped['Test Accuracy'])

plt.xlabel('SNR [dB]')
plt.xticks(data_grouped['SNR'])
plt.ylabel('Test accuracy')
plt.yticks(range(0, 101, 5))
plt.title('Relationship between SNR and classification accuracy')

plt.grid(True)
plt.show()
