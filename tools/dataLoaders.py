import torch
import os
from torch.utils.data import Dataset
from torch import is_tensor
from torch.utils.data import DataLoader
import numpy as np 
import random
import torchvision.transforms.functional as TF
from tqdm import tqdm


class RadarData(Dataset):
    def __init__(self, 
                 trainMode: str, 
                 rootPath: str,
                 trainDataPath: str, 
                 seqLen: int):
        """
        Initializes the RadarData dataset class for loading horizontal and vertical radar data.

        Args:
            trainMode (str): Indicates whether the mode is 'train' or 'val'.
            rootPath (str): Root path where the radar and skeleton data are stored.
            trainDataPath (str): Path to the specific training data files.
            seqLen (int): Length of the sequence to be loaded for training or validation.

        """
        self.seqLen = seqLen
        self.mode = trainMode
        self.rootPath = rootPath
        self.trainPath = trainDataPath
        
        if self.mode == "train": 
            self.participants = ['p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', "p11"] 
            self.angles = ["an0", "an1" , "an2"]
            self.actions = ["ac1", "ac2", "ac3","ac4","ac5","ac6","ac7","ac8", "ac9"] 
            self.recording = ["r0", "r1"]
            
            
        if self.mode == "val":
            self.participants = ["p3"]
            self.angles = ["an1", "an1" , "an2"]
            self.actions = ["ac1", "ac2", "ac3","ac4","ac5","ac6","ac7","ac8","ac9"] 
            self.recording = ["r0", "r1"]
            
    
        # generate combos for epoch 
        
        # Generate the list of lists where all nested parts are mixed once in random order
        trials = []
        for participant in self.participants:
            for angle in self.angles:
                for action in self.actions:
                    for recording in self.recording:
                        trials.append([participant, angle, action, recording])

        # Shuffle the combined list to get random order
        random.shuffle(trials)
        self.trials = trials 
        
        if not os.path.exists(os.path.join(self.trainPath, self.mode)):
            # generate trainData
            counter = 0
            for i in tqdm(range(len(self.trials))):
                combination = self.trials[i]
                
                # check if file exists 
                path = os.path.join(self.rootPath, "radar",  "data_cube_parsed" + "_" + combination[0] + "_" + combination[1]  +  "_" + combination[2]+ "_" + combination[3] + ".npz")
                if os.path.exists(path):
                    print("processing path ", path)
                    # get radar 
                    data = np.load(path)
                    radar = data.files[0]
                    radar = data[radar]
                    
                    # get gt
                    pathTarget = os.path.join(self.rootPath, "skeletons",  "skeleton" + "_" + combination[0]  + "_" + combination[1]  +  "_" + combination[2] + "_" + combination[3] + ".npy")
                    dataTarget = np.load(pathTarget)
                    dataTarget = dataTarget.reshape(dataTarget.shape[0], 26, 3)
                    
                    
                    for b in range(radar.shape[0] - self.seqLen): # <--  use for overlapping sequences
                    #for b in range(0, radar.shape[0] - self.seqLen, int(self.seqLen/2)):
                        radarOut = radar[b:b + self.seqLen]
                        radarTorch = torch.from_numpy(radarOut)
                        radarTorch = radarTorch.to(torch.complex64)
                        
                        # save radar
                        os.makedirs(os.path.join(self.trainPath, self.mode, "radar"), exist_ok=True)
                        os.makedirs(os.path.join(self.trainPath, self.mode, "target"), exist_ok= True)
                        torch.save(radarTorch, os.path.join(self.trainPath, self.mode, "radar", str(counter) + ".pth"))
                        
                        # save target
                        # get gt
                        gt = torch.from_numpy(dataTarget[b + self.seqLen])
                        torch.save(gt, os.path.join(self.trainPath, self.mode, "target", str(counter) + ".pth"))


                        counter += 1
                        
                        ## print
                        if counter % 200 == 0:
                            print("created ", counter, " sequences")
                else: 
                    pass
            
            print("data generated!")
            self.counter = counter
        else:
            self.counter = len(os.listdir(os.path.join(self.trainPath, self.mode, "radar")))

                           
    def __len__(self):
        return self.counter

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ method to get train and validation data

        Args:
            idx (int): indx of data 

        Returns:
            torch.Tensor: model input and target
        """
        
        # get radar 
        radar = torch.load(os.path.join(os.path.join(self.rootPath, self.mode, "radar", str(idx) + ".pth")), weights_only=True)

        
        gt = torch.load(os.path.join(os.path.join(self.rootPath, self.mode, "target", str(idx) + ".pth")), weights_only=True)
        gt = gt.flatten()
        return radar, gt
    


   







