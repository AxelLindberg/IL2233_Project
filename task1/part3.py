import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split


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
series = fibonacci_with_noise(length, signal_ratio)

train_size = 40
test_size = length - train_size
train, test = series[:train_size], series[train_size:]

# Check stationarity
result = adfuller(train)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')



#train an MLP model
# Prepare the training data for MLP model
X = []
y = []
for i in range(len(series) - 4):
    X.append(series[i:i+4])
    y.append(series[i+4])

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))  
model.add(Dense(64, activation='relu'))  
model.add(Dense(1))  
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
mlp_prediction = model.predict(X_test)
print(f'Predicted next value: {mlp_prediction}')
#print the actual value
print(f'Actual next value: {y_test}')



# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, mlp_prediction)
print(f'Mean Squared Error (MSE): {mse}')

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, mlp_prediction)
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, mlp_prediction.flatten())
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')


p = 1  # Adjust based on PACF plot
d = 0  # Data is already differenced
q = 2  # Adjust based on ACF plot

arima_model = ARIMA(X_train, order=(p, d, q))  # Adjust order if needed
arima_model_fit = arima_model.fit()
