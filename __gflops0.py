import torch
import torch.nn as nn
import torch.optim as optim
from utils import timer


# Define the neural network class
class MLP(nn.Module):
    def __init__(self, in_features=100, out_features=100):
        super(MLP, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.layer(x)
        x = torch.relu(x)
        return x

if __name__ == "__main__":
    # Create an instance of the neural network

    nepochs = 1000
    nrows = 1000000
    in_features = 100
    out_features = 100

    # Create an instance of the neural network
    net = MLP(in_features, out_features, hidden_dim=68, num_hidden_layers=3)
    train_timer = timer.Timer()


    # Define your training data and labels
    data = torch.randn(nrows, in_features,dtype=torch.float32)
    labels = torch.relu(data)

    # Use GPU for training if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    data = data.to(device)
    labels = labels.to(device)
    print(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Train the neural network
    train_timer.start()
    for epoch in range(nepochs):
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{nepochs}, Loss: {loss.item()}")

    # Calculate the error
    with torch.no_grad():
        predictions = net(data)
        error = criterion(predictions, labels)
        print(f"Error: {error.item()}")

    train_timer.end()
    print(f'Total time spent: {train_timer.get_average_time()}')

    # Output the parameters of the network
    for name, param in net.named_parameters():
        print(f"Parameter name: {name}, Value: {param}")
    print("Finished Training")
