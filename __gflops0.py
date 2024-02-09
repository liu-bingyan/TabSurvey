import torch
import torch.nn as nn
from utils import timer

class LinearModel(nn.Module):
    def __init__(self, input_size=100, hidden_dim=100):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.relu(out)
        out = self.linear2(out)
        out = torch.relu(out)
        return out
    
if __name__ == "__main__":
    # Generate random data
    input_size = 100
    hidden_dim = 100
    num_samples = 10000

    x = torch.randn(num_samples, input_size)
    w = torch.randn(input_size, 1)
    y = torch.sin(torch.matmul(x, w)) + torch.randn(num_samples, 1)
    
    # Create an instance of the LinearModel
    model = LinearModel(input_size, hidden_dim)

    # Move the data and model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    y = y.to(device)
    model = model.to(device)
    print(f'device: {device}')

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    total_timer = timer.Timer()
    epoch_timer = timer.Timer()
    # Train the model
    num_epochs = 1000
    total_timer.start()
    for epoch in range(num_epochs):
        epoch_timer.start()
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        epoch_timer.end()

    total_timer.end()

    print(f'total_timer : {total_timer.get_average_time()}')
    print(f'epoch_timer : {epoch_timer.get_average_time()}')