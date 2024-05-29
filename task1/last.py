import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Generate Fibonacci series with added noise
def fibonacci_with_noise(length, signal_ratio):
    fibonacci = [0, 1]
    for i in range(2, length):
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    noise = np.random.normal(0, 1, length)
    series = signal_ratio * np.array(fibonacci) + (1 - noise) * (1 - signal_ratio) * np.array(fibonacci)
    return series

length = 50
signal_ratio = 0.8
series = fibonacci_with_noise(length, signal_ratio)

train_size = 40
test_size = length - train_size

# Prepare the data
X = []
y = []
for i in range(len(series) - 4):
    X.append(series[i:i+4])
    y.append(series[i+4])

X = np.array(X)
y = np.array(y)

# Split the data into 80% train and 20% test
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# MLP Model
mlp_model = Sequential([
    Dense(64, input_dim=4, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
mlp_model.compile(optimizer='adam', loss='mse')

# Train the MLP model
mlp_model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=0)

# Make predictions using MLP
mlp_predictions = mlp_model.predict(X_test).flatten()

# Calculate MSE, MAE, and MAPE for MLP
mlp_mse = mean_squared_error(y_test, mlp_predictions)
mlp_mae = mean_absolute_error(y_test, mlp_predictions)
mlp_mape = np.mean(np.abs((y_test - mlp_predictions) / y_test)) * 100

# RNN Model
rnn_model = Sequential([
    SimpleRNN(64, input_shape=(4, 1), activation='relu', return_sequences=True),
    SimpleRNN(64, activation='relu'),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse')

# Train the RNN model
X_train_rnn = X_train.reshape((-1, 4, 1))
X_test_rnn = X_test.reshape((-1, 4, 1))
rnn_model.fit(X_train_rnn, y_train, epochs=100, batch_size=8, validation_data=(X_test_rnn, y_test), verbose=0)

# Make predictions using RNN
rnn_predictions = rnn_model.predict(X_test_rnn).flatten()

# Calculate MSE, MAE, and MAPE for RNN
rnn_mse = mean_squared_error(y_test, rnn_predictions)
rnn_mae = mean_absolute_error(y_test, rnn_predictions)
rnn_mape = np.mean(np.abs((y_test - rnn_predictions) / y_test)) * 100

# LSTM Model
lstm_model = Sequential([
    LSTM(64, input_shape=(4, 1), activation='relu', return_sequences=True),
    LSTM(64, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

# Train the LSTM model
lstm_model.fit(X_train_rnn, y_train, epochs=100, batch_size=8, validation_data=(X_test_rnn, y_test), verbose=0)

# Make predictions using LSTM
lstm_predictions = lstm_model.predict(X_test_rnn).flatten()

# Calculate MSE, MAE, and MAPE for LSTM
lstm_mse = mean_squared_error(y_test, lstm_predictions)
lstm_mae = mean_absolute_error(y_test, lstm_predictions)
lstm_mape = np.mean(np.abs((y_test - lstm_predictions) / y_test)) * 100

# Print MSE, MAE, and MAPE for all models
print("MLP:")
print("MSE:", mlp_mse)
print("MAE:", mlp_mae)
print("MAPE:", mlp_mape)
print("\nRNN:")
print("MSE:", rnn_mse)
print("MAE:", rnn_mae)
print("MAPE:", rnn_mape)
print("\nLSTM:")
print("MSE:", lstm_mse)
print("MAE:", lstm_mae)
print("MAPE:", lstm_mape)

# Plot the entire series with the predicted part at the end
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, len(series)), series, label='Original Series')
plt.plot(np.arange(train_size, len(series)), mlp_predictions, color='red', label='MLP Predicted Part')
plt.plot(np.arange(train_size, len(series)), rnn_predictions, color='green', label='RNN Predicted Part')
plt.plot(np.arange(train_size, len(series)), lstm_predictions, color='blue', label='LSTM Predicted Part')
plt.title('Entire Series with Predicted Parts')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

