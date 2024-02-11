import torch
import torch.nn as nn
import sklearn
from sklearn import datasets
import argparse
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


def run(args):
    input_size = 54
    hidden_dim = 100
    num_samples = 100000

    x, y = sklearn.datasets.fetch_covtype(return_X_y=True)
    x = x[:num_samples]
    y = y[:num_samples]
    x = torch.from_numpy(x).float()  # Specify float data type
    y = torch.from_numpy(y).long()  # Specify long data type
    #x = torch.randn(num_samples, input_size)
    #w = torch.randn(input_size, 1)
    #y = torch.sin(torch.matmul(x, w)) + torch.randn(num_samples, 1)
    
    # Create an instance of the LinearModel
    model = LinearModel(input_size, hidden_dim)

    # Move the data and model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    y = y.to(device)
    model = model.to(device)
    print(f'device: {device}')

    # Define loss function and optimizer
    #criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_x = torch.randn(100, input_size).to(device)
        test_y = torch.sin(torch.matmul(test_x, w)) + torch.randn(100, 1).to(device)
        test_outputs = model(test_x)
        test_loss = criterion(test_outputs, test_y)
        print(f'Test Loss: {test_loss.item():.4f}')

    print(f'total_timer : {total_timer.get_average_time()}')
    print(f'epoch_timer : {epoch_timer.get_average_time()}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with named arguments")
    args = parser.parse_args()
    run(args)