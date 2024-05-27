import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Make the data
period = 20  
sample_rate = 100
total_time = 40  
noise_amplitude = 0.1

t = np.arange(0, total_time, 1/sample_rate)
sin_wave = np.sin(2 * np.pi * t / period)
noise = noise_amplitude * np.random.normal(0, 1, len(t))
data = sin_wave + noise

print(f"Generated {len(t)} samples.")

input_size = 20  
X = []
y = []
for i in range(len(data) - input_size - 2):
    X.append(data[i:i+input_size])
    y.append(data[i+input_size:i+input_size+2])

X = np.array(X)
y = np.array(y)

# Split the data into 80% train and 20% test
split_index = int(len(X) * 0.8)
X_train, X_predict = X[:split_index], X[split_index:]
y_train, y_actual = y[:split_index], y[split_index:]

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_predict: {X_predict.shape}")
print(f"Shape of y_actual: {y_actual.shape}")

# Reshape input data for RNN and LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_predict = X_predict.reshape((X_predict.shape[0], X_predict.shape[1], 1))

# Build and train the RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(50, input_shape=(input_size, 1)))
rnn_model.add(Dense(2))
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the RNN model
rnn_loss = rnn_model.evaluate(X_predict, y_actual)
print(f'RNN Test Loss: {rnn_loss}')

# Make predictions with the RNN model
rnn_predictions = rnn_model.predict(X_predict)

# Build and train the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(input_size, 1)))
lstm_model.add(Dense(2))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the LSTM model
lstm_loss = lstm_model.evaluate(X_predict, y_actual)
print(f'LSTM Test Loss: {lstm_loss}')

# Make predictions with the LSTM model
lstm_predictions = lstm_model.predict(X_predict)

# Plot the results
plt.figure(figsize=(14, 7))
# Plot actual values
plt.plot(range(len(data)), data, label='Actual Data')
# Plot RNN predictions
for i in range(len(y_actual)):
    plt.plot(range(split_index + input_size + i, split_index + input_size + i + 2), rnn_predictions[i], 'r--', label='RNN Predictions' if i == 0 else "")
# Plot LSTM predictions
for i in range(len(y_actual)):
    plt.plot(range(split_index + input_size + i, split_index + input_size + i + 2), lstm_predictions[i], 'b--', label='LSTM Predictions' if i == 0 else "")

plt.xlabel('Time Index')
plt.ylabel('Value')
plt.title('Actual Data vs RNN and LSTM Predictions')
plt.legend()
plt.show()

# Example prediction using RNN model
test_input = X_predict[0].reshape(1, input_size, 1)
expected_output = y_actual[0]
predicted_output_rnn = rnn_model.predict(test_input)
print(f'RNN Predicted Output: {predicted_output_rnn[0]}')
print(f'Expected Output: {expected_output}')

# Example prediction using LSTM model
predicted_output_lstm = lstm_model.predict(test_input)
print(f'LSTM Predicted Output: {predicted_output_lstm[0]}')
print(f'Expected Output: {expected_output}')
