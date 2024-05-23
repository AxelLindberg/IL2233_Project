import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Step 1: Generate the series data
series = np.linspace(0, 1, 200, endpoint=False)  # Exclude 1, 200 points

# Add white noise to the series
noise_amplitude = 0.1  # Adjust this to control the amplitude of the noise
noise = np.random.normal(0, 0.25, series.shape) * noise_amplitude
noisy_series = series + noise

# Step 2: Prepare the input and output datasets
X = []
y = []
for i in range(len(noisy_series) - 4):
    X.append(noisy_series[i:i+4])
    y.append(noisy_series[i+4])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Design and train the MLP model
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))  # Input layer with 64 neurons
model.add(Dense(64, activation='relu'))  # Hidden layer with 64 neurons
model.add(Dense(1))  # Output layer with 1 neuron

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Example prediction
example_input = np.array([0.02, 0.025, 0.03, 0.035]).reshape(1, -1)
predicted_value = model.predict(example_input)
print(f'Predicted next value: {predicted_value[0][0]}')
