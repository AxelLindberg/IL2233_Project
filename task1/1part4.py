import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.layers import LSTM

#make the data
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


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

rnn_model = Sequential()
rnn_model.add(SimpleRNN(50, input_shape=(input_size, 1)))
rnn_model.add(Dense(2))

rnn_model.compile(optimizer='adam', loss='mse')

rnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

rnn_loss = rnn_model.evaluate(X_test, y_test)
print(f'RNN Test Loss: {rnn_loss}')

lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(input_size, 1)))
lstm_model.add(Dense(2))

lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

lstm_loss = lstm_model.evaluate(X_test, y_test)
print(f'LSTM Test Loss: {lstm_loss}')


test_input = X_test[0].reshape(1, input_size, 1)
expected_output = y_test[0]
predicted_output_rnn = rnn_model.predict(test_input)

print(f'RNN Predicted Output: {predicted_output_rnn[0]}')
print(f'Expected Output: {expected_output}')

predicted_output_lstm = lstm_model.predict(test_input)

print(f'LSTM Predicted Output: {predicted_output_lstm[0]}')
print(f'Expected Output: {expected_output}')

