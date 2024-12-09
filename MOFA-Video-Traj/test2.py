import torch
import torch.nn as nn

# Define a single-layer neural network
model = nn.Linear(2, 1)  # 2 inputs, 1 output
print("Initial weights:", model.weight)
print("Initial bias:", model.bias)

# Create a sample input and target
x = torch.tensor([[1.0, 2.0]], requires_grad=True)  # Input with requires_grad=True
target = torch.tensor([[5.0]])  # Target value

# Perform a forward pass
output = model(x)
print("\nOutput:", output)

# Define a loss function (Mean Squared Error)
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)
print("\nLoss:", loss)

# Perform backward pass to compute gradients
loss.backward()

# Gradients for weights and bias
print("\nGradients of weights:", model.weight.grad)
print("Gradients of bias:", model.bias.grad)

# Gradients of the input
print("Gradients of input x:", x.grad)
