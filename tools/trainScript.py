import os
from torch.utils.data import DataLoader
import dataLoaders
import trainLoop
import torch
from config import *
import sys
import random 
import numpy as np 

## LOAD MODEL HERE
sys.path.append(MODELPATH)
#from models import RadProPoser as Encoder
from models import CNN_LSTM as Encoder
#from models import HRRadarPose as Encoder

CF = Encoder().to(TRAINCONFIG["device"])


## set seed 
def setSeed(seed: int):
    """
    Set the seed for reproducibility in PyTorch across CPU, GPU, NumPy, and Python random.
    
    Parameters:
    seed (int): The seed value to use.
    """
    # Set seed for CPU and GPU (if available)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs.

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy's random functions
    np.random.seed(seed)

    # Ensure deterministic behavior in PyTorch (optional but recommended)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to: {seed}")


# Calculate the number of trainable and non-trainable parameters
totalParams = sum(p.numel() for p in CF.parameters())
trainableParams = sum(p.numel() for p in CF.parameters() if p.requires_grad)
nonTrainableParams = totalParams - trainableParams

print(f"Total parameters: {totalParams}")
print(f"Trainable parameters: {trainableParams}")
print(f"Non-trainable parameters: {nonTrainableParams}")


# get dataLoaders 
datasetTrain = dataLoaders.RadarData("train", PATHRAW, PATHRAW, SEQLEN)
dataTrain = DataLoader(datasetTrain, TRAINCONFIG["batchSize"], shuffle = True, num_workers = 1)

datasetVal = dataLoaders.RadarData("val", PATHRAW, PATHRAW,  SEQLEN)
dataVal = DataLoader(datasetVal, TRAINCONFIG["batchSize"], shuffle = True, num_workers = 1)


# loss functions
MSE = torch.nn.MSELoss()


if __name__ == "__main__":
    #setSeed(42)
    trainLoop.trainLoop(dataTrain,
              dataVal, 
              CF,
              False, 
              MODELNAME,
              TRAINCONFIG,  
              True, 
              TRAINCONFIG["device"], 
              MSE)
