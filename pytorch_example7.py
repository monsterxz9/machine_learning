import numpy as np
import torch
from torch.autograd import Variable
from data_processing.preprocessing import get_training_set

def polynomial_regression(x_train, y_train, input_dimension, output_dimension, degree, learning_rate, epochs):
  """Trains and evaluates a polynomial regression model using PyTorch.
  Args:
    x_train: A numpy array containing the input features.
    y_train: A numpy array containing the target labels.
    input_dimension: The number of features in the input dataset.
    output_dimension: The number of outputs to predict.
    degree: The degree of the polynomial.
    learning_rate: The learning rate.
    epochs: The number of epochs to train the model.
  Returns:
    A trained polynomial regression model.
  """
  # Check if x_train and y_train are both None
  if all([x_train is None, y_train is None]):
    raise ValueError("x_train and y_train cannot both be None")
  # Define the polynomial regression model class
  class PolynomialRegression(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, degree):
      super().__init__()
      # Define the model parameters
      self.weights = torch.nn.Parameter(torch.randn(degree + 1, input_dimension, output_dimension), requires_grad=True)
      # self.bias = torch.nn.Parameter(torch.randn(output_dimension), requires_grad=True)
    def forward(self, x):
      # Calculate the predicted outputs
      y = torch.zeros(x.shape[0], output_dimension)
      for i in range(degree + 1):
        y += torch.pow(x, i) @ self.weights[i]
      # y += self.bias
      return y

# Initialize the model
  model = PolynomialRegression(input_dimension, output_dimension, degree)
# Define the loss function and optimizer
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Convert the inputs and labels to variables
  inputs = Variable(torch.from_numpy(x_train.astype(np.float32)))
  labels = Variable(torch.from_numpy(y_train.astype(np.float32)))
# Train the model
  for epoch in range(epochs):
    # Clear the gradient buffers
    optimizer.zero_grad()
    # Get the output from the model
    outputs = model(inputs)
    # Calculate the loss
    loss = criterion(outputs, labels)
   # Back propagate the loss
    loss.backward()
    # Update the model parameters
    optimizer.step()
  # Evaluate the model
  with torch.no_grad():
    # Calculate the predicted outputs
    predicted_outputs = model(inputs).cpu().data.numpy()
  return model, predicted_outputs

# Create the x_train and y_train variables
# x_train = np.array([[1, 2, 3, 4], [3, 4, 3, 4], [2, 5, 5, 6]])
# y_train = np.array([[10, 20, 30, 40], [20, 30, 50, 60], [34, 58, 29, 70]])
x_train, y_train = get_training_set(0)
# Train the quadratic polynomial regression model
model, predicted_outputs = polynomial_regression(x_train, y_train, 4, 4, 2, 0.00001, 10000000)
print(model.weights)
print(y_train)
# Print the predicted outputs
print(predicted_outputs)
