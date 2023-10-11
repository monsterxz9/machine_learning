import numpy as np
import torch
from torch.autograd import Variable
from data_processing.preprocessing import get_training_set


def linear_regression(x, y, input_dimension, output_dimension, learning_rate, epochs):
    class LinearRegression(torch.nn.Module):
        def __init__(self, input_dimension, output_dimension):
            super().__init__()
            self.linear = torch.nn.Linear(input_dimension, output_dimension)

        def forward(self, x):
            return self.linear(x)

    # Initialize the model
    model = LinearRegression(input_dimension, output_dimension)

    # If a GPU is available, move the model to the GPU
    if torch.cuda.is_available():
        model.cuda()

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move inputs and labels to the same device as the model
    inputs = Variable(torch.from_numpy(x.astype(np.float32)))
    labels = Variable(torch.from_numpy(y.astype(np.float32)))

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()

    # Train the model
    for epoch in range(epochs):
        # ... (rest of your training loop remains the same)
        # Convert the inputs and labels to variables

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

# Create dummy data for training
# x_train = np.array([[i, i * 2, i * 3, i * 4] for i in range(11)], dtype=np.float32).reshape(-1, 4)
# y_train = np.array([[i * 10, i * 21, i * 32, i * 44] for i in range(11)], dtype=np.float32).reshape(-1, 4)
x_train, y_train = get_training_set(0)
print(type(x_train))
print(type(y_train))

x_train = x_train.reshape(-1, 4)
y_train = y_train.reshape(-1, 4)
x_train = x_train * 100
y_train = y_train * 100
#Train the linear regression model
model, predicted_outputs = linear_regression(x_train, y_train, 4, 4, 0.00001, 100000)

# Evaluate the model
print('True outputs:')
print(y_train)
print('Predicted outputs:')
print(predicted_outputs)

