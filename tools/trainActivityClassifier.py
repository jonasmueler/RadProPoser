import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from config import *
import random

class Dataset(Dataset):
    def __init__(self, mode: str):
        self.data = os.listdir(os.path.join(PATHLATENT, "dataTrain", "X"))
        self.labels = os.listdir(os.path.join(PATHLATENT, "dataTrain", "targets"))
        
        # Shuffle both lists 
        self.data, self.labels = zip(*random.sample(list(zip(self.data, self.labels)), len(self.data)))

        # Convert back
        self.data = list(self.data)
        self.labels = list(self.labels)
        
        
        # split in train and validation
        criterion = round(0.9*len(self.data))
        
        if mode == "train":
            self.data = self.data[0:criterion]
            self.labels = self.labels[0:criterion]
        
        if mode == "val":
            self.data = self.data[criterion:]
            self.labels = self.labels[criterion:]
            
    def __len__(self):
        return len(self.data)
    
    def padMatrixToRows(self, 
                    matrix: torch.Tensor, 
                    targetRows: int) -> torch.Tensor:
        """
        Pads a single 2D matrix (tensor) to have the specified number of rows.

        Args:
            matrix (torch.Tensor): 2D tensor of shape [rows, cols].
            targetRows (int): The target number of rows for the matrix.

        Returns:
            torch.Tensor: The padded 2D tensor with `targetRows` rows.
        """
        rows, cols = matrix.shape
        if rows < targetRows:
            padding = (0, 0, 0, targetRows - rows)  # Pad only rows (left, right, top, bottom)
            padded_matrix = torch.nn.functional.pad(matrix, padding, mode='constant', value=0)
        else:
            padded_matrix = matrix  # If already the target size or larger, return unchanged
        
        return padded_matrix

    def __getitem__(self, idx):
        # load data 
        x = self.padMatrixToRows(torch.load(os.path.join(PATHLATENT, "dataTrain", "X", self.data[idx])).squeeze(), 600)
        y = torch.load(os.path.join(PATHLATENT, "dataTrain", "targets", self.labels[idx]))
        return x.float(), y.float()

# Model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=6, stride=6) 
        self.out = nn.Linear(64, 9)
                                    

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = torch.mean(x.permute(0, 2, 1), dim = 1)
        x = self.out(x)
        
        return x

# Training 
def trainLoop(model: nn.Module, 
              dataloaderTrain: DataLoader,
              dataloaderVal: DataLoader,  
              criterion: torch.nn.MSELoss, 
              optimizer: torch.optim.Adam, 
              device: str,
              epochs: int = 5000):
    
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for inputs, labels in dataloaderTrain:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # evaluation loop
        valLossFull = torch.tensor(0.0)
        counter = 1
        with torch.no_grad():
            model.eval()
            for inputs, labels in dataloaderVal:
                inputs, labels = inputs.to(device), labels.to(device)
                valLoss = criterion(model(inputs).squeeze(), labels)
                valLossFull += valLoss
                counter += 1
            
            valLossFull = valLossFull.detach().cpu().item()/counter
            print("validation loss: ", valLossFull)
                


def main():
    # test model
    model = Classifier()
    testD = torch.rand((5, 600, 256))
    print(model(testD).size())

    # Dataset and DataLoader
    datasetTrain = Dataset("train")
    datasetVal = Dataset("val")
    dataloaderTrain = DataLoader(datasetTrain, batch_size=CLASSIFIERCONFIG["batch_size"], shuffle=True)
    dataloaderVal = DataLoader(datasetVal, batch_size=CLASSIFIERCONFIG["batch_size"], shuffle=True)

    # Model, Loss, and Optimizer
    model = Classifier()
    criterion = nn.CrossEntropyLoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=CLASSIFIERCONFIG["learning_rate"])

    # Train the model
    trainLoop(model, dataloaderTrain, dataloaderVal, criterion, optimizer, CLASSIFIERCONFIG["device"], epochs=CLASSIFIERCONFIG["num_epochs"])


if __name__ == "__main__":
    main()
    
