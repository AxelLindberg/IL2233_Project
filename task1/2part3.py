import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Define ARMA parameters
ar_coefs = [1, -0.75, 0.5]  
ma_coefs = [1, 0.5, 0.3]    

arma_process = ArmaProcess(ar_coefs, ma_coefs)
num_points = 1000
stationary_series = arma_process.generate_sample(nsample=num_points)

input_size = 4
X = []
y = []

for i in range(len(stationary_series) - input_size):
    X.append(stationary_series[i:i+input_size])
    y.append(stationary_series[i+input_size])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = Sequential()
model.add(SimpleRNN(64, input_shape=(input_size, 1), activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

example_input = X_test[0].reshape(1, input_size, 1)
predicted_value = model.predict(example_input)
print(f'Predicted next value: {predicted_value}')
print(f'Actual next value: {y_test[0]}')

prediction_errors = []
for i in range(len(X_test)):
    example_input = X_test[i].reshape(1, input_size, 1)
    predicted_value = model.predict(example_input)
    prediction_errors.append(y_test[i] - predicted_value)

prediction_errors = np.squeeze(prediction_errors)

plt.plot(prediction_errors)
plt.title('Prediction Error Series (White Noise)')
plt.xlabel('Time')
plt.ylabel('Error')
plt.show()
