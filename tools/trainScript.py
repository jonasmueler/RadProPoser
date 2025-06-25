import os
from torch.utils.data import DataLoader
import dataLoaders
import trainLoop
import torch
from config import *
import sys
import random 
import numpy as np 
import torch.nn as nn

## LOAD MODEL HERE
sys.path.append(MODELPATH)
#from evidential_pose_regression import RadProPoserEvidential as Encoder
from vae_lstm_ho import RadProPoserPad as Encoder
#from models import CNN_LSTM as Encoder
#from models import HRRadarPose as Encoder

CF = Encoder().to(TRAINCONFIG["device"])

def loadCheckpoint(model: nn.Module, 
                   optimizer: torch.optim.Adam, 
                   path: str)->torch.nn.Module:
    """
    Loads a trained model and/or optimizer state from a checkpoint file.

    Args:
        model (nn.Module): The model instance to load the saved state into.
        optimizer (torch.optim.Optimizer): The optimizer instance to load the saved state into. 
                                           If None, only the model state will be loaded.
        path (str): Path to the checkpoint file.

    Returns:
        torch.nn.Module: The model with the loaded state.
        If an optimizer is provided, returns a list [model, optimizer] with both loaded states.
    """

    if optimizer != None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("checkpoint loaded")
        return [model, optimizer]
    elif optimizer == None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        print("checkpoint loaded")
        return model

# load model 
#CF = loadCheckpoint(CF, None, "/home/jonas/code/RadProPoser/trainedModels/humanPoseEstimation/RadProPoserEvidential1")


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
    setSeed(42)
    trainLoop.trainLoop(dataTrain,
              dataVal, 
              CF,
              False, 
              MODELNAME,
              TRAINCONFIG,  
              True, 
              TRAINCONFIG["device"], 
              MSE)
