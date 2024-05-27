import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# make the data
num_points = 1000  
mean = 0  
variance = 1  
std_dev = np.sqrt(variance) 

white_noise = np.random.normal(mean, std_dev, num_points)

input_size = 20
X = []
y = []

for i in range(len(white_noise) - input_size):
    X.append(white_noise[i:i+input_size])
    y.append(white_noise[i+input_size])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape to [samples, timesteps, features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = Sequential()
model.add(SimpleRNN(64, input_shape=(input_size, 1), activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Plotting original series and predicted series
total_points = len(white_noise)
train_size = int(0.8 * total_points)

plt.figure(figsize=(10, 6))

# Plot original series
plt.plot(np.arange(0, train_size), white_noise[:train_size], label='Original Series (Training)', color='blue')
plt.plot(np.arange(train_size, total_points), white_noise[train_size:], label='Original Series (Testing)', color='green')

# Plot predicted series
predicted_series = []

for i in range(train_size, total_points):
    example_input = white_noise[i-input_size:i].reshape(1, input_size, 1)
    predicted_value = model.predict(example_input)
    predicted_series.append(predicted_value[0][0])

plt.plot(np.arange(train_size, total_points), predicted_series, label='Predicted Series', color='red')

plt.title('Original Series and Predicted Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

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


