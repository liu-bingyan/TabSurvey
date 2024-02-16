import torch
import torch.nn as nn
import sklearn
from sklearn import datasets
import argparse
import torch.nn.functional as F
from utils import timer
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, in_features=100,hidden_dim=68,out_features=7, num_hidden_layers=3):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, out_features))
        self.layers.append(nn.Softmax(dim=1))

        self.connections = (num_hidden_layers-1)* hidden_dim**2 + hidden_dim*(in_features+out_features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

@profile
def run(args):
    input_size = 54
    hidden_dim = 99
    num_samples = 581012
    num_epochs = 300
    learning_rate = 7.0e-4

    # Create an instance of the LinearModel
   # model = LinearModel(input_size, hidden_dim)
    
    model = MLP(input_size, hidden_dim)

    num_connections  = model.connections
    GFLO = num_epochs*6*num_samples*num_connections / 1e9
    print(f"GFLO is {GFLO:.2f}")
    print(f"Should take {GFLO/1600:.2f} seconds on a 5 TFLOPS machine")

    # Create dummy data
    x, y = datasets.fetch_covtype(return_X_y=True)
    x = torch.tensor(x, dtype=torch.float32)
    y = y-1
    y = torch.tensor(y, dtype=torch.long)
    y = F.one_hot(y, num_classes=7).to(torch.float32)
    # Move the data and model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(type(x))
    x = x.to(device)
    print(type(x))
    y = y.to(device)
    model = model.to(device)
    print(f'device: {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    total_timer = timer.Timer()
    epoch_timer = timer.Timer()
    # Train the model
    total_timer.start()
    
    print('construct dataset')
    # Create a TensorDataset
    dataset = TensorDataset(x, y)

    print('construct dataloader')
    # Create a DataLoader with batch size
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print('start training the model')
    for epoch in range(num_epochs):
        epoch_timer.start()
        
        for i, (batch_x, batch_y) in enumerate(dataloader):
            print('training on batch i:', i)
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=7.0e-4, help="Learning rate for training")
    args = parser.parse_args()
    run(args)