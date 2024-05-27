import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Generate Fibonacci series with added noise
def fibonacci_with_noise(length, signal_ratio):
    fibonacci = [0, 1]
    for i in range(2, length):
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    noise = np.random.normal(0, 1, length)
    series = signal_ratio * np.array(fibonacci) + (1 - signal_ratio) * noise
    return series

length = 50
signal_ratio = 0.95
fibonacci_series = fibonacci_with_noise(length, signal_ratio)

train_size = 40
test_size = length - train_size
train, test = fibonacci_series[:train_size], fibonacci_series[train_size:]

fibonacci_series_log = np.log(fibonacci_series)

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(fibonacci_series_log, lags=15, ax=plt.gca())
plt.subplot(212)
plot_pacf(fibonacci_series_log, lags=15, ax=plt.gca())
plt.show()


model = ARIMA(fibonacci_series_log, order=(2, 1, 0))
model_fit = model.fit()
print(model_fit.summary())


residuals = model_fit.resid

# Plot the residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Residuals')
plt.title('ARIMA Model Residuals')
plt.legend()
plt.show()

# Forecasting
forecast = model_fit.forecast(steps=test_size)
print("Forecasted values:")
print(forecast)

#plot forecast against actual values
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, train_size), fibonacci_series[:train_size], label='Training Data')
plt.plot(np.arange(train_size, length), fibonacci_series[train_size:], label='Test Data')
plt.plot(np.arange(train_size, length), forecast, label='Forecast')
plt.title('Fibonacci Series with ARIMA Forecast')
plt.legend()
plt.show()


'''
# Prepare the training data for MLP model
train_mlp = np.array([train[i:i+5] for i in range(train_size-5)])
train_mlp_labels = train[5:]

# MLP model
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(64, activation='relu'),
    Dense(1)
])
mlp_model.compile(optimizer='adam', loss='mse')
mlp_model.fit(train_mlp, train_mlp_labels, epochs=100, verbose=0)

# Prepare the training data for RNN and LSTM models
train_rnn = train_mlp.reshape(-1, 5, 1)
train_rnn_labels = train_mlp_labels

# RNN model
rnn_model = Sequential([
    SimpleRNN(64, input_shape=(5, 1)),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(train_rnn, train_rnn_labels, epochs=100, verbose=0)

# LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(5, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(train_rnn, train_rnn_labels, epochs=100, verbose=0)

# ARIMA model
# Using ACF and PACF plots to determine p and q
p = 1  # Adjust based on PACF plot
d = 0  # Data is already differenced
q = 2  # Adjust based on ACF plot

arima_model = ARIMA(train_diff, order=(p, d, q))  # Adjust order if needed
arima_model_fit = arima_model.fit()

# Forecasting
forecast_diff = arima_model_fit.forecast(steps=test_size)
forecast = np.concatenate((train[-1:], forecast_diff)).cumsum()

# Ensure forecast length matches test length
forecast = forecast[1:]  # Remove the extra first element

# Function to evaluate model
def evaluate_model_residuals(model, X_test, y_test, input_shape=None):
    if input_shape is None:
        predictions = model.predict(X_test)
    else:
        predictions = model.predict(X_test.reshape(input_shape)).flatten()
    residuals = y_test - predictions
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs(residuals / y_test)) * 100
    return mse, mae, mape, residuals

# Evaluate models
y_test = test[5:]
mlp_mse, mlp_mae, mlp_mape, mlp_residuals = evaluate_model_residuals(mlp_model, X_test, y_test)

test_rnn = test_mlp.reshape(-1, 5, 1)
rnn_mse, rnn_mae, rnn_mape, rnn_residuals = evaluate_model_residuals(rnn_model, test_rnn, (-1, 5, 1))
lstm_mse, lstm_mae, lstm_mape, lstm_residuals = evaluate_model_residuals(lstm_model, test_rnn, (-1, 5, 1))

arima_residuals = test[5:] - forecast[:test_size-5]
arima_mse = mean_squared_error(test[5:], forecast[:test_size-5])
arima_mae = mean_absolute_error(test[5:], forecast[:test_size-5])
arima_mape = np.mean(np.abs(arima_residuals / test[5:])) * 100

print("ARIMA Model:")
print(f"MSE: {arima_mse}, MAE: {arima_mae}, MAPE: {arima_mape}%")
print("MLP Model:")
print(f"MSE: {mlp_mse}, MAE: {mlp_mae}, MAPE: {mlp_mape}%")
print("RNN Model:")
print(f"MSE: {rnn_mse}, MAE: {rnn_mae}, MAPE: {rnn_mape}%")
print("LSTM Model:")
print(f"MSE: {lstm_mse}, MAE: {lstm_mae}, MAPE: {lstm_mape}%")
'''