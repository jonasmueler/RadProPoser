import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from config import *
import random

def saveCheckpoint(model: nn.Module, 
                   optimizer: torch.optim.Adam, 
                   filename: str):
    """
    Saves the model and optimizer states to a checkpoint file.

    Args:
        model (nn.Module): The model whose state needs to be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state needs to be saved.
        filename (str): The name of the file to save the checkpoint to.

    """

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(PATHORIGIN, filename))
    print("checkpoint saved")
    return




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
        criterion = round(0.8*len(self.data))

        # load data into ram 
        self.dataR = []
        self.labelsR = []
        for i in range(len(self.data)):
            # load data 
            x = self.padMatrixToRows(torch.load(os.path.join(PATHLATENT, "dataTrain", "X", self.data[i])).squeeze(), 600)

            #x = torch.load(os.path.join(PATHLATENT, "dataTrain", "X", self.data[idx])).squeeze()
            y = torch.load(os.path.join(PATHLATENT, "dataTrain", "targets", self.labels[i]))

            self.dataR.append(x)
            self.labelsR.append(y)

        self.data = self.dataR
        self.labels = self.labelsR

        print("data loaded into ram")
        
        if mode == "train":
            self.data = self.data[0:criterion]
            self.labels = self.labels[0:criterion]
            
        
        if mode == "val":
            self.data = self.data[criterion:]
            self.labels = self.labels[criterion:]
            
    def __len__(self):
        return len(self.data)
    
    def load_directory_into_ram(self, directory_path: str):
        """"
        Load all .pth files in a directory into RAM in order.

        Args:
            directory_path (str): Path to the directory containing .pth files.

        Returns:
            dict: A dictionary where keys are file indices (e.g., 0, 1, 2) and values are the loaded tensors.
        """
        # Get all .pth files in the directory, sorted by numeric order
        file_names = sorted(
            [f for f in os.listdir(directory_path) if f.endswith(".pth")],
            key=lambda x: int(os.path.splitext(x)[0])  # Sort by numeric value of the file name
        )
        
        # Load all files into RAM
        tensor_dict = {}
        for file_name in file_names:
            file_path = os.path.join(directory_path, file_name)
            file_index = int(os.path.splitext(file_name)[0])  # Extract the numeric index
            #print(f"Loading {file_name} into RAM...")
            tensor_dict[file_index] = torch.load(file_path)

        print(f"Loaded {len(tensor_dict)} files into RAM.")
        return tensor_dict
    
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
        #x = self.padMatrixToRows(torch.load(os.path.join(PATHLATENT, "dataTrain", "X", self.data[idx])).squeeze(), 600)
        x = self.data[idx]
        
        # augment 
        flip = random.choice([True, False])

        if flip:
            std = torch.std(x, dim=0, keepdim=True)  # Keep dimensions for broadcasting
            noise = torch.normal(0, std.expand_as(x))  # Expand std to match x's shape
            x = x + noise

        #x = torch.load(os.path.join(PATHLATENT, "dataTrain", "X", self.data[idx])).squeeze()
        #y = torch.load(os.path.join(PATHLATENT, "dataTrain", "targets", self.labels[idx]))
        y = self.labels[idx]

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
    

class LSTMModel(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim= 32, output_dim=9, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        output = self.fc(lstm_out[:, -1, :])
        return output
    

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=601):
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim

        # Create positional encodings
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, hidden_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=12, output_dim=9, num_layers=1, nhead=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.embedding = nn.Linear(input_dim, hidden_dim)  # Input embedding
        self.positional_encoding = PositionalEncoding(hidden_dim)  # Positional encoding
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)

        # Transform input to hidden_dim
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, hidden_dim)

        # Add the [CLS] token to the beginning of each sequence
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)  # Shape: (1, batch_size, hidden_dim)
        x = torch.cat((cls_tokens, x.permute(1, 0, 2)), dim=0)  # Shape: (sequence_length+1, batch_size, hidden_dim)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through transformer
        transformer_out = self.transformer(x)  # Shape: (sequence_length+1, batch_size, hidden_dim)

        # Use the [CLS] token's output for classification
        cls_output = transformer_out[0]  # Shape: (batch_size, hidden_dim)
        output = self.fc(cls_output)  # Shape: (batch_size, output_dim)

        return output
    
class TwoLayerPerceptron(nn.Module):
    def __init__(self, input_dim=2, sequence_length=600, hidden_dim=64, output_dim=9):
        """
        Two-layer perceptron model.

        Args:
            input_dim (int): Number of features per time step.
            sequence_length (int): Length of the input sequence.
            hidden_dim (int): Number of hidden units in the hidden layer.
            output_dim (int): Size of the output.
        """
        super(TwoLayerPerceptron, self).__init__()
        self.input_size = input_dim * sequence_length  # Flattened input size
        self.hidden_dim = hidden_dim

        # Define the two fully connected layers
        self.fc1 = nn.Linear(self.input_size, hidden_dim)  # First layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second layer

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        batch_size, seq_len, input_dim = x.size()

        # Flatten the input: (batch_size, sequence_length, input_dim) -> (batch_size, input_size)
        x = x.view(batch_size, -1)

        # Pass through the two fully connected layers
        x = torch.relu(self.fc1(x))  # First layer with ReLU activation
        output = self.fc2(x)  # Second layer

        return output


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
        model.train()
        for inputs, labels in dataloaderTrain:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            #print(loss)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # evaluation loop
        valLossFull = torch.tensor(0.0).cuda()
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
            
            # save checkpoint
            saveCheckpoint(model, optimizer, os.path.join(ACTIVITYCLASSIFICATIONCKPT, "model" + ".pth"))
                


def main():
    # test model
    model = TwoLayerPerceptron().to(CLASSIFIERCONFIG["device"])
    testD = torch.rand((5, 600, 2)).to(CLASSIFIERCONFIG["device"])
    print(model(testD).size())

    # Dataset and DataLoader
    datasetTrain = Dataset("train")
    datasetVal = Dataset("val")
    dataloaderTrain = DataLoader(datasetTrain, batch_size=CLASSIFIERCONFIG["batch_size"], shuffle=True)
    dataloaderVal = DataLoader(datasetVal, batch_size=CLASSIFIERCONFIG["batch_size"], shuffle=True)

    # Model, Loss, and Optimizer
    criterion = nn.CrossEntropyLoss()  # Binary Cross-Entropy Loss
    optimizer = optim.RMSprop(model.parameters(), lr=CLASSIFIERCONFIG["learning_rate"])

    # Train the model
    trainLoop(model, dataloaderTrain, dataloaderVal, criterion, optimizer, CLASSIFIERCONFIG["device"], epochs=CLASSIFIERCONFIG["num_epochs"])


if __name__ == "__main__":
    main()
    
