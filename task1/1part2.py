import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

series = np.linspace(0, 1, 200, endpoint=False)  # Exclude 1, 200 points

noise_amplitude = 0.1  # Amp adjustment
noise = np.random.normal(0, 1, series.shape) * noise_amplitude
noisy_series = series + noise

X = []
y = []
for i in range(len(noisy_series) - 4):
    X.append(noisy_series[i:i+4])
    y.append(noisy_series[i+4])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))  
model.add(Dense(64, activation='relu')) 
model.add(Dense(1))  

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

example_input = np.array([0.02, 0.025, 0.03, 0.035]).reshape(1, -1)
predicted_value = model.predict(example_input)
print(f'Predicted next value: {predicted_value[0][0]}')
