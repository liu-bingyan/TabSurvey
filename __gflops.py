import torch
import torch.nn as nn
import torch.optim as optim
from utils import timer


# Define the neural network class
class MLP(nn.Module):
    def __init__(self, in_features=90, out_features=7, hidden_dim=68, num_hidden_layers=3):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, out_features))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
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
    labels = torch.randn(nrows, out_features,dtype=torch.float32)

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
    
    train_timer.end()
    print(f'total time spent : {train_timer.get_average_time()}')
    print("Finished Training")
