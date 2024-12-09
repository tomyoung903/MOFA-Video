import torch
import torch.nn as nn

# Define a simple 2-layer neural network
class TwoLayerNN(nn.Module):
    def __init__(self):
        super(TwoLayerNN, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)  # Non-linearity
        x = self.layer2(x)
        return x

model = TwoLayerNN()
input_tensor = torch.randn(1, 10, requires_grad=True)

# Forward pass without torch.no_grad()
output = model(input_tensor)

# Check the grad_fn attribute to see intermediate results' tracking
print("Output grad_fn:", output.grad_fn)  # Shows operations leading to output
print("Activation1 grad_fn (from Layer1):", output.grad_fn.next_functions[0][0])
