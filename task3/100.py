import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from sklearn.model_selection import train_test_split

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

# Create input-output pairs for each input size
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

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

# Create and train LSTM models for each configuration
for input_size in input_sizes:
    # Univariate series: MLII
    X_train_mlii, y_train_mlii = create_dataset(train_data['MLII'].values, input_size, output_size)
    X_test_mlii, y_test_mlii = create_dataset(test_data['MLII'].values, input_size, output_size)
    
    # Define and compile the LSTM model
    model_mlii = Sequential()
    model_mlii.add(LSTM(50, input_shape=(input_size, 1)))
    model_mlii.add(Dense(output_size))
    model_mlii.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model_mlii.fit(X_train_mlii, y_train_mlii, epochs=100, batch_size=32, verbose=0)
    
    # Evaluate the model
    mse_mlii = model_mlii.evaluate(X_test_mlii, y_test_mlii, verbose=0)
    print(f"MSE (Univariate MLII, Input Size {input_size}): {mse_mlii}")
    
    # Univariate series: V5
    X_train_v5, y_train_v5 = create_dataset(train_data['V5'].values, input_size, output_size)
    X_test_v5, y_test_v5 = create_dataset(test_data['V5'].values, input_size, output_size)
    
    # Define and compile the LSTM model
    model_v5 = Sequential()
    model_v5.add(LSTM(50, input_shape=(input_size, 1)))
    model_v5.add(Dense(output_size))
    model_v5.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model_v5.fit(X_train_v5, y_train_v5, epochs=100, batch_size=32, verbose=0)
    
    # Evaluate the model
    mse_v5, _ = model_v5.evaluate(X_test_v5, y_test_v5, verbose=0)
    print(f"MSE (Univariate V5, Input Size {input_size}): {mse_v5}")
    
    # Bivariate series: combine MLII and V5
    X_train_combined = np.concatenate((X_train_mlii[..., np.newaxis], X_train_v5[..., np.newaxis]), axis=-1)
    y_train_combined = y_train_mlii  # Output size is the same for both MLII and V5
    X_test_combined = np.concatenate((X_test_mlii[..., np.newaxis], X_test_v5[..., np.newaxis]), axis=-1)
    y_test_combined = y_test_mlii
    
    # Define and compile the LSTM model
    model_combined = Sequential()
    model_combined.add(LSTM(50, input_shape=(input_size, 2)))
    model_combined.add(Dense(output_size))
    model_combined.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model_combined.fit(X_train_combined, y_train_combined, epochs=100, batch_size=32, verbose=0)
    
    # Evaluate the model
    mse_combined, _ = model_combined.evaluate(X_test_combined, y_test_combined, verbose=0)
    print(f"MSE (Bivariate Combined MLII and V5, Input Size {input_size}): {mse_combined}")
