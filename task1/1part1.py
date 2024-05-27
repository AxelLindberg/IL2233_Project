import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Make the data
series = np.linspace(0, 1, 200, endpoint=False)  # Exclude 1, 200 points

# Prepare the data
X = []
y = []
for i in range(len(series) - 4):
    X.append(series[i:i+4])
    y.append(series[i+4])

X = np.array(X)
y = np.array(y)

# Split the data into 80% train and 20% test
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))  
model.add(Dense(64, activation='relu'))  
model.add(Dense(1))  
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot the original series
plt.figure(figsize=(14, 7))
plt.plot(series, label='Original Series')


# Plot the predictions
test_plot_index = range(split_index + 4, len(series))
plt.plot(test_plot_index, y_pred, color='red', label='Predictions', alpha=0.5)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Original Series, Training Data, and Predictions')
plt.legend()
plt.show()

# Print shapes of the data
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

# Example prediction
example_input = np.array([0.02, 0.025, 0.03, 0.035]).reshape(1, -1)
predicted_value = model.predict(example_input)
print(f'Predicted next value: {predicted_value}')
