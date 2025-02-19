import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from config import *
import random
from trainLoop import *
from trainActivityClassifier import *
from sklearn.metrics import accuracy_score, f1_score, recall_score

class DatasetTest(Dataset):
    def __init__(self):
        self.data = os.listdir(os.path.join(PATHLATENT, "dataTest", "X"))
        self.labels = os.listdir(os.path.join(PATHLATENT, "dataTest", "targets"))
            
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
        x = self.padMatrixToRows(torch.load(os.path.join(PATHLATENT, "dataTest", "X", self.data[idx])).squeeze(), 600)
        y = torch.load(os.path.join(PATHLATENT, "dataTest", "targets", self.labels[idx]))
        return x.float(), y.float()




def computeMetrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    Computes accuracy, precision, recall, and F1 score for multi-class classification using PyTorch tensors.

    Args:
        preds (torch.Tensor): Model predictions of shape [N, C] (probabilities or logits).
        labels (torch.Tensor): Ground truth labels of shape [N] or [N, C] (one-hot encoded).

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score (all as scalar values).
    """
    # Get predicted classes (highest probability)
    preds_classes = torch.argmax(preds, dim=1)

    # If labels are one-hot encoded, convert to class indices
    if labels.dim() == 2:  # Check if one-hot encoded
        labels = torch.argmax(labels, dim=1)

    # Accuracy
    accuracy = (preds_classes == labels).float().mean()

    # Initialize true positives (TP), false positives (FP), and false negatives (FN)
    num_classes = preds.size(1)
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)

    # Compute TP, FP, FN for each class
    for cls in range(num_classes):
        TP[cls] = ((preds_classes == cls) & (labels == cls)).sum().float()
        FP[cls] = ((preds_classes == cls) & (labels != cls)).sum().float()
        FN[cls] = ((preds_classes != cls) & (labels == cls)).sum().float()

    # Compute precision, recall, and F1 for each class
    precision = TP / (TP + FP + 1e-8)  # Avoid division by zero
    recall = TP / (TP + FN + 1e-8)
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Macro-averaged metrics
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_score = f1_per_class.mean()

    return {
        "accuracy": accuracy.item(),
        "precision": precision_macro.item(),
        "recall": recall_macro.item(),
        "f1_score": f1_score.item()
    }


if __name__ == "__main__":
    # load model 
    model = Classifier()
    model = loadCheckpoint(model, None, os.path.join(ACTIVITYCLASSIFICATIONCKPT, None)) # add trained classifier model name here
    print("model loaded correctly")
    
    # get data loader 
    dataset = DatasetTest()
    dataloaderTest = DataLoader(dataset, batch_size=CLASSIFIERCONFIG["batch_size"], shuffle=True, drop_last=True)
    
    # get predictions and ground truth data
    preds = []
    Labels = []
    with torch.no_grad():
        model.eval()
        for inputs, labels in dataloaderTest:
            inputs, labels = inputs.to(CLASSIFIERCONFIG["device"]), labels.to(CLASSIFIERCONFIG["device"])
            pred = torch.nn.functional.softmax(model(inputs), dim = 1)
            preds.append(pred)
            Labels.append(labels)
            
    preds = torch.cat(preds, dim = 0)
    labels = torch.cat(Labels, dim = 0)
    
    
    print(computeMetrics(preds, labels))
    
            

