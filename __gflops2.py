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
    def __init__(self, in_features,hidden_dim,out_features, num_hidden_layers):
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

def run(args):
    num_epochs = args.num_epochs
    batch_size = args.batch_size    

    print(args)
    x, y = datasets.fetch_covtype(return_X_y=True)
    num_samples = args.sample_portion * x.shape[0]
    learning_rate = math.sqrt(batch_size/num_samples)*0.01
    
    model = MLP(in_features=x.shape[1], hidden_dim=99, out_features=7, num_hidden_layers=3)
    print(f'model in_features: {x.shape[1]}, hidden_dim: 99, out_features: 7, num_hidden_layers: 3, num_samples: {num_samples},leanring_rate: {learning_rate}')
    
    #GFLO
    num_connections  = model.connections
    GFLO = num_epochs*6*num_samples*num_connections / 1e9
    print(f"GFLO is {GFLO:.2f}, should take {GFLO/(5000 * 0.3):.2f} seconds on a 5 TFLOPS machine")

    # load data 
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    y = F.one_hot(y-1, num_classes=7).to(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move data and model to GPU
    x = x.to(device)
    y = y.to(device)
    model = model.to(device)
    



    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    print('start training the model')
    epoch_timer = timer.Timer()
    test_timer = timer.Timer()

    loss = criterion(y*0,y)
    print(f'Initial Loss: {loss.item():.10f}')
    for epoch in range(num_epochs):
        epoch_timer.start()
        if args.data_loader==2:
            for i, (batch_x, batch_y) in enumerate(dataloader):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()              
        elif args.data_loader==1:
            for i in range(0, num_samples, batch_size):
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
                outputs = model(x)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if (epoch<10)| ((epoch+1) % 10 == 0) :
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.10f}')
        epoch_timer.end()        
    print('finished training the model')

    test_timer.start()
    model.eval()
    with torch.no_grad():
        test_outputs = model(x)
        test_loss = criterion(test_outputs, y)
        print(f'Test Loss: {test_loss.item():.10f}')
    test_timer.end()
    print(f'epoch : {epoch_timer.get_average_time()}, total : {epoch_timer.get_total_time()}, test : {test_timer.get_average_time()}')    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with named arguments")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--sample_portion", type=int, default=1, help="Number of samples in the dataset")

    parser.add_argument("--data_loader", type=int, default=False, help="Use dataloader or not")
    parser.add_argument("--batch_size", type=int, default=16384, help="Batch size for training")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle the dataset",action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    run(args)