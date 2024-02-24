import math
import torch
import torch.nn as nn
import sklearn
from sklearn import datasets
import argparse
import torch.nn.functional as F
from utils import timer,fast_tensor_data_loader,fast_tensor_data_loader_2
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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

        self.connections = (num_hidden_layers-1)* hidden_dim**2 + hidden_dim*(in_features+out_features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#@profile
def run(args):
    num_epochs = args.num_epochs
    batch_size = args.batch_size    

    print(args)
    x, y = datasets.fetch_covtype(return_X_y=True)
    num_samples = args.sample_portion * x.shape[0]

    learning_rate = None
    if args.learning_rate is None:
        learning_rate = math.sqrt(batch_size/num_samples)*0.01
    else:
        learning_rate = args.learning_rate
    
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
    #x = x.to(device)
    #y = y.to(device)
    #model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(x, y)
    dataloader2 = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    dataloader3 = fast_tensor_data_loader.FastTensorDataLoader(x, y, batch_size=args.batch_size, shuffle=args.shuffle)
    dataloader5 = fast_tensor_data_loader_2.FastTensorDataLoader(x, y, batch_size=args.batch_size, shuffle=args.shuffle)

    print('start training the model')
    epoch_timer = timer.Timer()
    test_timer = timer.Timer()

    loss = criterion(y*0,y)
    print(f'Initial Loss: {loss.item():.10f}')
    
    for epoch in range(num_epochs):
        epoch_timer.start()
        if args.data_loader==5:
            for i, (batch_x, batch_y) in enumerate(dataloader5):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        elif args.data_loader==3:
            for i, (batch_x, batch_y) in enumerate(dataloader3):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
        elif args.data_loader==2:
            for i, (batch_x, batch_y) in tqdm(enumerate(dataloader2)):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()              
        elif args.data_loader==1:
            if args.shuffle:
                indices = torch.randperm(num_samples)
                x = x[indices]
                y = y[indices]
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
        _, predicted = torch.max(test_outputs, 1)
        total = y.size(0)
        correct = (predicted == torch.argmax(y, dim=1)).sum().item()
        accuracy = correct / total * 100
        print(f'Test Accuracy: {accuracy:.2f}%')        
    test_timer.end()

    print(f'epoch : {epoch_timer.get_average_time()}, total : {epoch_timer.get_total_time()}, test : {test_timer.get_average_time()}')    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with named arguments")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--sample_portion", type=int, default=1, help="Number of samples in the dataset")

    parser.add_argument("--data_loader", type=int, default=0, help="Use dataloader or not")
    parser.add_argument("--batch_size", type=int, default=16384, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=None, help="Use dataloader or not")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle the dataset",action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    run(args)