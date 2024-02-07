import torch
import torch.nn as nn
import torch.optim as optim
from utils import timer


# Define the neural network class
class FuckMLP(nn.Module):

    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, task):
        super().__init__()

        self.task = task

        self.layers = nn.ModuleList()

        # Input Layer (= first hidden layer)
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden Layers (number specified by n_layers)
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))

        # Use ReLU as activation for all hidden layers
        for layer in self.layers:
            x = torch.relu(layer(x))

        # No activation function on the output
        x = self.output_layer(x)

        if self.task == "classification":
            x = torch.softmax(x, dim=1)

        return x

def generate_labels(data):
    y = torch.argmax(data, dim=1)
    return y.flatten()  

if __name__ == "__main__":
    # Create an instance of the neural network

    nepochs = 1000
    nrows = 100000
    in_features = 100
    out_features = 1
    print(f"nepochs: {nepochs}, nrows: {nrows}, in_features: {in_features}, out_features: {out_features}")

    # Create an instance of the neural network
    net = FuckMLP(n_layers=3, input_dim=in_features, hidden_dim=68, output_dim=out_features,task='classification')
    train_timer = timer.Timer()


    # Define your training data and labels
    data = torch.randn(nrows, in_features,dtype=torch.float32)
    labels = torch.max(data,dim=1)
    print(data.shape)
    print(labels.shape)

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
    
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{nepochs}, Loss: {loss.item()}")
    
    train_timer.end()
    print(f'total time spent : {train_timer.get_average_time()}')
    print("Finished Training")
