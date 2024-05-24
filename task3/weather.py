import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
df = pd.read_csv('data.csv', delimiter=',', header=2, names=['Year', 'Temperature'])
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

# Select the time series data
series = df['Temperature']['1880':'2020']

# Interpolate missing values
series.interpolate(inplace=True)

# Plot the time series
plt.figure(figsize=(14, 6))
plt.plot(series, label='Land Average Temperature')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Global Land Average Temperature Over Time')
plt.legend()
plt.show()

# Perform Augmented Dickey-Fuller test
result = adfuller(series)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
for key, value in result[4].items():
    print('Critical Values:')
    print(f'   {key}, {value}')

# Difference the series if not stationary
if result[1] > 0.05:
    differenced_series = series.diff().dropna()
    print('Series was differenced to achieve stationarity.')
else:
    differenced_series = series
    print('Series is already stationary.')

# Plot ACF and PACF
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plot_acf(differenced_series, ax=plt.gca(), lags=40, title='Autocorrelation Function (ACF)')
plt.subplot(2, 1, 2)
plot_pacf(differenced_series, ax=plt.gca(), lags=40, title='Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Print statistical summary
print("Statistical Summary:")
mean = differenced_series.mean()
std_dev = differenced_series.std()
print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")

# Fit ARIMA model
p = 3
d = 1
q = 3
model = ARIMA(differenced_series, order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())

# Make predictions
predictions = model_fit.predict(start=0, end=len(differenced_series)-1)

# Calculate residuals
residuals = differenced_series - predictions

# Identify anomalies based on the residuals
residuals_abs = abs(residuals)
anomaly_threshold = np.percentile(residuals_abs, 98)  # 98th percentile for 2% threshold
print("Anomaly Threshold:", anomaly_threshold)
anomalies = residuals[residuals_abs > anomaly_threshold]

# Plot residuals and anomalies
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.scatter(anomalies.index, anomalies, color='red', label='Anomalies')
plt.legend()
plt.title('Residuals and Anomalies')
plt.show()

# Calculate the number of anomalies to match the 2% ratio
num_anomalies = int(0.02 * len(differenced_series))
anomalies = residuals.abs().nlargest(num_anomalies)

print("Anomalies:")
print(anomalies)
