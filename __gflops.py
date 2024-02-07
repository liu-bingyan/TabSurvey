import torch
import torch.nn as nn
import torch.optim as optim
from utils import timer

# Define the neural network class
class SimpleNet(nn.Module):
    def __init__(self, in_features = 1000,out_features=100):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x
    

if __name__ == "__main__":
    nepochs = 100
    nrows = 1000000
    in_features = 1000
    out_features = 100



    # Create an instance of the neural network
    net = SimpleNet(in_features,out_features)
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
    
        print(f"Epoch {epoch+1}/{100}, Loss: {loss.item()}")
    
    train_timer.end()
    print(f'total time spent : {train_timer.get_average_time()}')
    print("Finished Training")
