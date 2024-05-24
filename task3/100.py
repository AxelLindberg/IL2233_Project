import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the dataset
data = pd.read_csv('100.csv')

# Display the first few rows of the dataset
print(data.head())

# Plot the signals MLII and V5
plt.figure(figsize=(12, 6))
plt.plot(data['Elapsed time'], data['MLII'], label='MLII')
plt.plot(data['Elapsed time'], data['V5'], label='V5')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.title('ECG Signals: MLII and V5')
plt.legend()
plt.show()

# Create lag plots for MLII and V5
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
lag_plot(data['MLII'], lag=1)
plt.title('MLII Lag-1 Plot')

plt.subplot(2, 2, 2)
lag_plot(data['MLII'], lag=2)
plt.title('MLII Lag-2 Plot')

plt.subplot(2, 2, 3)
lag_plot(data['V5'], lag=1)
plt.title('V5 Lag-1 Plot')

plt.subplot(2, 2, 4)
lag_plot(data['V5'], lag=2)
plt.title('V5 Lag-2 Plot')

plt.tight_layout()
plt.show()


# Function to create input-output pairs for LSTM
def create_dataset(data, input_size, output_size):
    X, y = [], []
    for i in range(len(data) - input_size - output_size + 1):
        X.append(data[i:i+input_size])
        y.append(data[i+input_size:i+input_size+output_size])
    return np.array(X), np.array(y)

# Split the dataset into training and test sets (80-20 split)
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Define input sizes
input_sizes = [4, 8, 16]
output_size = 1  # Output vector size

# Create and train LSTM models with different input sizes
residuals = []

for input_size in input_sizes:
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(input_size, 1)))
    lstm_model.add(Dense(2))
    lstm_model.compile(optimizer='adam', loss='mse')

    # Create input-output pairs for training
    X_train, y_train = create_dataset(train_data['MLII'], input_size, output_size)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Train the LSTM model
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the LSTM model
    X_test, y_test = create_dataset(test_data['MLII'], input_size, output_size)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    #lstm_loss = lstm_model.evaluate(X_test, y_test)

    y_pred = lstm_model.predict(X_test)

    # Compute residuals
    residuals.append(y_test - y_pred)
       

for i in range(len(residuals)):
    residuals_abs = abs(residuals[i])
    anomaly_threshold = np.percentile(residuals_abs, 99.5)  # 98th percentile for 2% threshold
    print("Anomaly Threshold:", anomaly_threshold)
    anomalies = residuals[i][residuals_abs > anomaly_threshold]
    plt.figure(figsize=(10, 6))
    plt.plot(residuals[i], label='Residuals')
    plt.scatter(anomalies.index, anomalies, color='red', label='Anomalies')
    plt.legend()
    plt.title('Residuals and Anomalies')
    plt.show()  

