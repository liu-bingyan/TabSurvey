import torch
import torch.nn as nn
import sklearn
from sklearn import datasets
import argparse
from utils import timer

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.relu(out)
        out = self.linear2(out)
        out = torch.relu(out)
        return out


def run(args):
    input_size = 54
    hidden_dim = 100
    num_samples = 10000000
    num_epochs = 300

    num_connections  = hidden_dim*(input_size+hidden_dim)
    GFLO = num_epochs*6*num_samples*num_connections / 1e9
    print(f"GFLO is {GFLO:.2f}")
    print(f"Should take {GFLO/1600:.2f} seconds on a 5 TFLOPS machine")

    # Create dummy data
    x, y = datasets.make_regression(n_samples=num_samples, n_features=input_size, noise=0.001)
    x = torch.tensor(x, dtype=torch.float32)
    y = y.reshape(-1, 1)
    y = torch.tensor(y, dtype=torch.float32)
    # Create an instance of the LinearModel
    model = LinearModel(input_size, hidden_dim)

    # Move the data and model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    y = y.to(device)
    model = model.to(device)
    print(f'device: {device}')

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    total_timer = timer.Timer()
    epoch_timer = timer.Timer()
    # Train the model
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
        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        epoch_timer.end()

    total_timer.end()
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(x)
        test_loss = criterion(test_outputs, y)
        print(f'Test Loss: {test_loss.item():.4f}')

    print(f'total_timer : {total_timer.get_average_time()}')
    print(f'epoch_timer : {epoch_timer.get_average_time()}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with named arguments")
    args = parser.parse_args()
    run(args)