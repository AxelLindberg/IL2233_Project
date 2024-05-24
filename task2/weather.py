import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore

df = pd.read_csv('GlobalTemperatures.csv', parse_dates=['dt'])
df.set_index('dt', inplace=True)


series = df['LandAverageTemperature']['1880-01-01':'2020-01-01']

series.interpolate(inplace=True)

series.loc['1994-12-01'] = 20
series.loc["2000-06-01"] = -10

decomposition = seasonal_decompose(series.dropna(), model='additive', period=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(series, label='Original Series')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# get the z-score of each residual
residual_z_scores = pd.Series(zscore(residual.dropna()), index=residual.dropna().index)

# set the threshold to 3
threshold = 2
anomalies = residual_z_scores[abs(residual_z_scores) > 3]

plt.figure(figsize=(10, 6))
plt.plot(series, label='Original Series')
plt.scatter(anomalies.index, anomalies.values, color='red', label='Anomalies')
plt.legend(loc='best')
plt.title('Anomalies in Land Average Temperature')
plt.show()

print("Anomalies detected:")
print(anomalies)


