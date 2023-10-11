import numpy as np
import torch
from torch.autograd import Variable
from data_processing.preprocessing import get_training_set

def polynomial_regression(x_train, y_train, input_dimension, output_dimension, degree, learning_rate, epochs):
    class PolynomialRegression(torch.nn.Module):
        def __init__(self, input_dimension, output_dimension, degree):
            super().__init__()
            self.degree = degree
            self.weights = torch.nn.Parameter(torch.randn(degree + 1, input_dimension, output_dimension), requires_grad=True)

        def forward(self, x):
            y = torch.zeros(x.shape[0], output_dimension)
            for i in range(self.degree + 1):
                y += torch.pow(x, i) @ self.weights[i]
            return y

    # Initialize the model and move it to the GPU
    model = PolynomialRegression(input_dimension, output_dimension, degree).cuda()
    # Convert the inputs and labels to tensors and move them to the GPU
    inputs = torch.from_numpy(x_train).cuda()
    labels = torch.from_numpy(y_train).cuda()
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    with torch.no_grad():
        predicted_outputs = model(inputs).cpu().data.numpy()

    return model, predicted_outputs

 # Create the x_train and y_train variables (assuming get_training_set is correctly defined)
x_train, y_train = get_training_set(0)

# Train the quadratic polynomial regression model on GPU
model, predicted_outputs = polynomial_regression(x_train, y_train, 4, 4, 2, 0.00001, 10000000)

# Print the model weights, target labels, and predicted outputs
print(model.weights)
print(y_train)
print(predicted_outputs)
