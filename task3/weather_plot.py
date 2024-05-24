import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('data.csv', delimiter=',', header=2, names=['Year', 'Temperature'])
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

# Select the time series data
series = df['Temperature']['1880':'2020']

series.interpolate(inplace=True)

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(series)
plt.title('Line Plot')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
series.hist(bins=30)
plt.title('Histogram')
plt.show()

# Density plot
plt.figure(figsize=(10, 6))
series.plot(kind='kde')
plt.title('Density Plot')
plt.show()

# Heatmap (correlation matrix)
plt.figure(figsize=(10, 6))
plt.imshow(series.to_numpy().reshape(-1, 1).T, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Temperature Anomalies')
plt.title('Time Series Heatmap')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks([])
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=series)
plt.title('Box Plot')
plt.show()

# Lag-1 plot
pd.plotting.lag_plot(series, lag=1)
plt.title('Lag-1 Plot')
plt.show()

# Lag-2 plot
pd.plotting.lag_plot(series, lag=2)
plt.title('Lag-2 Plot')
plt.show()


