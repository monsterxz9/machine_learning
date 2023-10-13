import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from data_processing.preprocessing import get_training_set

# # Input data
# X_train = np.array([[3, 5, 8, 11], [2, 1, 9, 3]])
# # Output data with three arguments
# y_train = np.array([[15, 34, 56, 85], [12, 32, 78, 90]])
x_train, y_train = get_training_set(0)

# Define the model
model = Sequential([
  Dense(8, activation='relu', input_shape=(4,)),
  Dense(6, activation='relu'),
  Dense(4)
])

# Compile the model
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())

# Train the model
model.fit(x_train, y_train, epochs=100)

# Predict values using the trained model
y_pred = model.predict(x_train)
# Print the predicted values
print("True values")
print(y_train)
print("Predicted values:")
print(y_pred)
