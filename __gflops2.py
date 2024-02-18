import math
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
    print(args)
    input_size = 54
    hidden_dim = 99
    num_samples = args.num_samples
    num_epochs = 300
    learning_rate = math.sqrt(args.batch_size/num_samples)*0.1

    # Create an instance of the LinearModel
    # model = LinearModel(input_size, hidden_dim)
    
    model = MLP(input_size, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    num_connections  = model.connections
    GFLO = num_epochs*6*num_samples*num_connections / 1e9
    print(f"GFLO is {GFLO:.2f}")
    print(f"Should take {GFLO/(5000 * 0.3):.2f} seconds on a 5 TFLOPS machine")

    # Create dummy data
    x, y = datasets.fetch_covtype(return_X_y=True)
    x = torch.tensor(x, dtype=torch.float32)
    y = y-1
    y = torch.tensor(y, dtype=torch.long)
    y = F.one_hot(y, num_classes=7).to(torch.float32)

    # Move the data and model to GPU if available
    print('cuda is available:', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    x = x.to(device)
    y = y.to(device)

    model = model.to(device)
    criterion = criterion.to(device)
    #optimizer = optimizer.to(device)
    
    total_timer = timer.Timer()
    epoch_timer = timer.Timer()

    # Train the model
    total_timer.start()
    
    # Create a TensorDataset
    dataset = TensorDataset(x, y)

    # Create a DataLoader with batch size
    batch_size = args.batch_size
    print('batch_size:', batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=args.shuffle)#, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)


    #torch.cuda.synchronize()
    print('start training the model')
    for epoch in range(num_epochs):
        epoch_timer.start()
        
        if args.data_loader:
            for i, (batch_x, batch_y) in enumerate(dataloader):
            
                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        elif args.batching:
            for i in range(0, num_samples, batch_size):
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_timer.end()        
        # Print progress
        if (epoch<10)| (epoch % 10 == 0):
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print('avg time for epoch:', epoch_timer.get_average_time())

    #torch.cuda.synchronize()
    total_timer.end()

    # Test the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(x)
        test_loss = criterion(test_outputs, y)
        print(f'Test Loss: {test_loss.item():.4f}')

    print(f'total_timer : {total_timer.get_average_time()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with named arguments")
    parser.add_argument("--batch_size", type=int, default=16384, help="Batch size for training")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle the dataset",action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")  
<<<<<<< HEAD
    parser.add_argument("--pin_memory", type=bool, default=False, help="Pin memory for faster transfer to GPU")
    parser.add_argument("--drop_last", type=bool, default=False, help="Drop the last batch if it is not complete")
    parser.add_argument("--num_samples", type=int, default=581012, help="Number of samples in the dataset")
=======
    parser.add_argument("--pin_memory", type=bool, default=False, help="Pin memory for faster transfer to GPU", action=argparse.BooleanOptionalAction)
    parser.add_argument("--drop_last", type=bool, default=False, help="Drop the last batch if it is not complete", action=argparse.BooleanOptionalAction)
    parser.add_argument("--data_loader", type=bool, default=False, help="Use dataloader or not", action=argparse.BooleanOptionalAction)
    parser.add_argument("--batching", type=bool, default=False, help="Use dataloader or not", action=argparse.BooleanOptionalAction)
>>>>>>> f0a53c79658494f4871210b6a1684a0b0d26833b
    args = parser.parse_args()
    run(args)