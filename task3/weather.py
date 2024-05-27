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

# Calculate z-scores for residuals
residuals_mean = residuals.mean()
residuals_std = residuals.std()
z_scores = (residuals - residuals_mean) / residuals_std

# Calculate the 98th percentile threshold for z-scores
z_score_threshold = np.percentile(abs(z_scores), 98)
print("Z-score Threshold for Anomalies (98th Percentile):", z_score_threshold)

# Identify anomalies based on z-score threshold
anomalies = residuals[abs(z_scores) > z_score_threshold]

# Plot residuals and anomalies
plt.figure(figsize=(10, 6))
plt.plot(series , label='Series')
plt.scatter(anomalies.index, anomalies, color='red', label='Anomalies')
plt.legend()
plt.title('Residuals and Anomalies')
plt.show()

print("Anomalies (Z-scores):")
print(anomalies)

# Make in-sample predictions
predictions = model_fit.predict(start=0, end=len(differenced_series)-1)

# Calculate residuals (prediction errors)
residuals = differenced_series - predictions

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals (Prediction Errors)')
plt.legend()
plt.show()

