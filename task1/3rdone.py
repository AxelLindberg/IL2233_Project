import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

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
fibonacci_series = fibonacci_with_noise(length, signal_ratio)

train_size = 40
test_size = length - train_size
train, test = fibonacci_series[:train_size], fibonacci_series[train_size:]

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(fibonacci_series, lags=15, ax=plt.gca())
plt.subplot(212)
plot_pacf(fibonacci_series, lags=15, ax=plt.gca())
plt.show()

# Augmented Dickey-Fuller (ADF) test
result = adfuller(fibonacci_series)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value}')

# Differencing if necessary
if result[1] > 0.05:
    d = 1  # Differencing order
    fibonacci_series_diff = np.diff(fibonacci_series, n=d)
    train_diff, test_diff = fibonacci_series_diff[:train_size-d], fibonacci_series_diff[train_size-d:]
else:
    d = 0
    fibonacci_series_diff = fibonacci_series
    train_diff, test_diff = train, test
# Plot ACF and PACF after differencing
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(fibonacci_series_diff, lags=15, ax=plt.gca())
plt.subplot(212)
plot_pacf(fibonacci_series_diff, lags=15, ax=plt.gca())
plt.show()

# Fit ARIMA model
model_arima = ARIMA(train_diff, order=(2, 3, 1))
model_fit = model_arima.fit()
print(model_fit.summary())

# Forecasting with ARIMA
forecast_diff = model_fit.forecast(steps=test_size)
forecast_arima = np.concatenate((train[-1:], forecast_diff)).cumsum()

print("ARIMA Forecast:", forecast_arima)    
print("Actual Test Data:", test)

# Calculate errors for ARIMA
mse_arima = np.mean((test - forecast_arima[:test_size]) ** 2)
mae_arima = np.mean(np.abs(test - forecast_arima[:test_size]))
mape_arima = np.mean(np.abs((test - forecast_arima[:test_size]) / test)) * 100

print("ARIMA Model:")
print("Mean Squared Error (MSE):", mse_arima)
print("Mean Absolute Error (MAE):", mae_arima)
print("Mean Absolute Percentage Error (MAPE):", mape_arima)

# Plot actual vs. ARIMA forecast
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, train_size), fibonacci_series[:train_size], label='Training Data')
plt.plot(np.arange(train_size, length), test, label='Test Data')
plt.plot(np.arange(train_size, length), forecast_arima[:test_size], label='ARIMA Forecast', color='red')
plt.title('Fibonacci Series with ARIMA Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Prepare data for MLP, RNN, LSTM
window_size = 5  # Adjust the window size as needed
train_mlp = np.array([train_diff[i:i+window_size] for i in range(len(train_diff)-window_size)])
train_rnn = train_mlp.reshape(-1, window_size, 1)
train_lstm = train_rnn


