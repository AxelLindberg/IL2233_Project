import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Make the data
series = np.linspace(0, 1, 200, endpoint=False)  # Exclude 1, 200 points

X = []
y = []
for i in range(len(series) - 4):
    X.append(series[i:i+4])
    y.append(series[i+4])

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Design and train the MLP model
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))  # Input layer
model.add(Dense(64, activation='relu'))  # Hidden layer
model.add(Dense(1))  # Output layer

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Example prediction
example_input = np.array([0.02, 0.025, 0.03, 0.035]).reshape(1, -1)
predicted_value = model.predict(example_input)
print(f'Predicted next value: {predicted_value}')
