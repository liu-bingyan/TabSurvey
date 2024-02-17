import torch
import numpy as np

@profile
def shutest():
  x = torch.rand(1000000, 1000)
  batch_x = x[0:1000]
  indices = torch.randperm(x.size(0))
  shuffled_x = x[indices]

if __name__ == "__main__": 
  shutest()