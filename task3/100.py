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
mse_scores = []
mape_scores = []

for input_size in input_sizes:
    # Train model to predict next value of MLII
    lstm_model_mlii = Sequential()
    lstm_model_mlii.add(LSTM(64, input_shape=(input_size, 1)))
    lstm_model_mlii.add(Dense(1))
    lstm_model_mlii.compile(optimizer='adam', loss='mse')

    # Create input-output pairs for training MLII
    X_train_mlii, y_train_mlii = create_dataset(train_data['MLII'], input_size, output_size)
    X_train_mlii = X_train_mlii.reshape((X_train_mlii.shape[0], X_train_mlii.shape[1], 1))

    # Train the LSTM model
    lstm_model_mlii.fit(X_train_mlii, y_train_mlii, epochs=10, batch_size=32)

    # Evaluate the LSTM model on MLII test data
    X_test_mlii, y_test_mlii = create_dataset(test_data['MLII'], input_size, output_size)
    X_test_mlii = X_test_mlii.reshape((X_test_mlii.shape[0], X_test_mlii.shape[1], 1))
    
    y_pred_mlii = lstm_model_mlii.predict(X_test_mlii)

    # Compute residuals for MLII
    residuals.append(y_test_mlii.flatten() - y_pred_mlii.flatten())

    mse_mlii = np.mean((y_test_mlii.flatten() - y_pred_mlii.flatten()) ** 2)
    mape_mlii = np.mean(np.abs((y_test_mlii.flatten() - y_pred_mlii.flatten()) / y_test_mlii.flatten())) * 100
    mse_scores.append((input_size, 'MLII', mse_mlii))
    mape_scores.append((input_size, 'MLII', mape_mlii))

    # Train model to predict next value of V5
    lstm_model_v5 = Sequential()
    lstm_model_v5.add(LSTM(64, input_shape=(input_size, 1)))
    lstm_model_v5.add(Dense(1))
    lstm_model_v5.compile(optimizer='adam', loss='mse')

    # Create input-output pairs for training V5
    X_train_v5, y_train_v5 = create_dataset(train_data['V5'], input_size, output_size)
    X_train_v5 = X_train_v5.reshape((X_train_v5.shape[0], X_train_v5.shape[1], 1))

    # Train the LSTM model
    lstm_model_v5.fit(X_train_v5, y_train_v5, epochs=10, batch_size=32)

    # Evaluate the LSTM model on V5 test data
    X_test_v5, y_test_v5 = create_dataset(test_data['V5'], input_size, output_size)
    X_test_v5 = X_test_v5.reshape((X_test_v5.shape[0], X_test_v5.shape[1], 1))
    
    y_pred_v5 = lstm_model_v5.predict(X_test_v5)

    # Compute residuals for V5
    residuals.append(y_test_v5.flatten() - y_pred_v5.flatten())

    mse_v5 = np.mean((y_test_v5.flatten() - y_pred_v5.flatten()) ** 2)
    mape_v5 = np.mean(np.abs((y_test_v5.flatten() - y_pred_v5.flatten()) / y_test_v5.flatten())) * 100
    mse_scores.append((input_size, 'V5', mse_v5))
    mape_scores.append((input_size, 'V5', mape_v5))

    # Train bi-variate model to predict next value of MLII using both MLII and V5
    lstm_model_bivariate = Sequential()
    lstm_model_bivariate.add(LSTM(64, input_shape=(input_size, 2)))
    lstm_model_bivariate.add(Dense(1))
    lstm_model_bivariate.compile(optimizer='adam', loss='mse')

    # Create input-output pairs for training bi-variate series
    train_series = np.column_stack((train_data['MLII'], train_data['V5']))
    X_train_bi, y_train_bi = create_dataset(train_series, input_size, output_size)
    X_train_bi = X_train_bi.reshape((X_train_bi.shape[0], X_train_bi.shape[1], 2))
    y_train_bi = y_train_bi[:, :, 0]  # Predicting the next value of MLII

    # Train the LSTM model
    lstm_model_bivariate.fit(X_train_bi, y_train_bi, epochs=10, batch_size=32)

    # Evaluate the LSTM model on bi-variate test data
    test_series = np.column_stack((test_data['MLII'], test_data['V5']))
    X_test_bi, y_test_bi = create_dataset(test_series, input_size, output_size)
    X_test_bi = X_test_bi.reshape((X_test_bi.shape[0], X_test_bi.shape[1], 2))
    y_test_bi = y_test_bi[:, :, 0]  # Actual next value of MLII
    
    y_pred_bi = lstm_model_bivariate.predict(X_test_bi)

    # Compute residuals for bi-variate series
    residuals.append(y_test_bi.flatten() - y_pred_bi.flatten())

    mse_bi = np.mean((y_test_bi.flatten() - y_pred_bi.flatten()) ** 2)
    mape_bi = np.mean(np.abs((y_test_bi.flatten() - y_pred_bi.flatten()) / y_test_bi.flatten())) * 100
    mse_scores.append((input_size, 'Bi-variate', mse_bi))
    mape_scores.append((input_size, 'Bi-variate', mape_bi))

# Plot residuals and anomalies
for i, input_size in enumerate(input_sizes):
    for j, label in enumerate(['MLII', 'V5', 'Bi-variate']):
        residuals_abs = abs(residuals[i * 3 + j])
        anomaly_threshold = np.percentile(residuals_abs, 98)  # 98th percentile for 2% threshold
        anomalies_indices = np.where(residuals_abs > anomaly_threshold)[0]
        anomalies = residuals[i * 3 + j][anomalies_indices]

        plt.figure(figsize=(10, 6))
        plt.plot(residuals[i * 3 + j], label='Residuals')
        plt.scatter(anomalies_indices, anomalies, color='red', label='Anomalies')
        plt.legend()
        plt.title(f'Residuals and Anomalies for Input Size {input_size} ({label})')
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.show()

# Print MSE and MAPE scores
print("\nMSE Scores:")
for score in mse_scores:
    print(f"Input Size: {score[0]}, Model: {score[1]}, MSE: {score[2]}")

print("\nMAPE Scores:")
for score in mape_scores:
    print(f"Input Size: {score[0]}, Model: {score[1]}, MAPE: {score[2]}%")


