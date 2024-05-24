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

# Check stationarity
result = adfuller(train)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Difference the series if it's not stationary
if result[1] > 0.05:
    train_diff = np.diff(train)
else:
    train_diff = train

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(train_diff, lags=15, ax=plt.gca())
plt.subplot(212)
plot_pacf(train_diff, lags=15, ax=plt.gca())
plt.show()

# MLP model
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(1)
])
mlp_model.compile(optimizer='adam', loss='mse')
mlp_model.fit(np.arange(train_size), train, epochs=100, verbose=0)

# RNN model
rnn_model = Sequential([
    SimpleRNN(64, input_shape=(1, 1)),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(train.reshape(-1, 1, 1), train, epochs=100, verbose=0)

# LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(1, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(train.reshape(-1, 1, 1), train, epochs=100, verbose=0)

# ARIMA model
# Using ACF and PACF plots to determine p and q
p = 1  # Adjust based on PACF plot
d = 0  # Data is already differenced
q = 3  # Adjust based on ACF plot

arima_model = ARIMA(train_diff, order=(p, d, q))  # Adjust order if needed
arima_model_fit = arima_model.fit()

# Forecasting
forecast_diff = arima_model_fit.forecast(steps=test_size)
forecast = np.concatenate((train[-1:], forecast_diff)).cumsum()

# Ensure forecast length matches test length
forecast = forecast[1:]  # Remove the extra first element

# Function to evaluate model
def evaluate_model_residuals(model, test, input_shape=None):
    if input_shape is None:
        predictions = model.predict(np.arange(len(test)))
    else:
        predictions = model.predict(test.reshape(input_shape)).flatten()
    residuals = test - predictions
    mse = mean_squared_error(test, predictions)
    mae = mean_absolute_error(test, predictions)
    mape = np.mean(np.abs(residuals / test)) * 100
    return mse, mae, mape, residuals

mlp_mse, mlp_mae, mlp_mape, mlp_residuals = evaluate_model_residuals(mlp_model, test)
rnn_mse, rnn_mae, rnn_mape, rnn_residuals = evaluate_model_residuals(rnn_model, test, (-1, 1, 1))
lstm_mse, lstm_mae, lstm_mape, lstm_residuals = evaluate_model_residuals(lstm_model, test, (-1, 1, 1))

arima_residuals = test - forecast
arima_mse = mean_squared_error(test, forecast)
arima_mae = mean_absolute_error(test, forecast)
arima_mape = np.mean(np.abs(arima_residuals / test)) * 100

print("ARIMA Model:")
print(f"MSE: {arima_mse}, MAE: {arima_mae}, MAPE: {arima_mape}%")
print("MLP Model:")
print(f"MSE: {mlp_mse}, MAE: {mlp_mae}, MAPE: {mlp_mape}%")
print("RNN Model:")
print(f"MSE: {rnn_mse}, MAE: {rnn_mae}, MAPE: {rnn_mape}%")
print("LSTM Model:")
print(f"MSE: {lstm_mse}, MAE: {lstm_mae}, MAPE: {lstm_mape}%")
