import torch
import os
from torch.utils.data import Dataset
from torch import is_tensor
from torch.utils.data import DataLoader
import numpy as np 
import random
import torchvision.transforms.functional as TF
from tqdm import tqdm


# Define your processing function
def preProcess(
                   x: torch.Tensor):
        x = x.unsqueeze(dim = 0)
        x = x - torch.mean(x, dim = -1, keepdim = True)
        #x = torch.fft.fftn(x, dim=(0, 1, 2, 3), norm="forward")
        x = torch.fft.fft(torch.fft.fft(torch.fft.fft(torch.fft.fft(x ,dim = -1,  norm = "forward"), dim = -2,  norm = "forward"), dim = -3,  norm = "forward"), dim = -4,  norm = "forward")
        #x = torch.fft.fftshift(x, dim=(-4, -3, -2, -1))

        #x = torch.fft.fftshift(torch.fft.fft2(x,norm='forward'), dim=-1)
        #x = torch.fft.fftshift(torch.fft.fft2(x, dim=(1,2),norm='forward'),dim=(1,2))  

        x = x.permute(0, 1, 5, 2, 3, 4) 
        return x.squeeze()

def preProcess_numpy(x: np.ndarray):
    # Add new axis (equivalent to `unsqueeze(dim=0)`)
    x = np.expand_dims(x, axis=0)

    # Subtract mean along the last dimension
    x = x - np.mean(x, axis=-1, keepdims=True)

    # Apply sequential 1D FFT along multiple axes with "forward" normalization
    x = np.fft.fft(x, axis=-1, norm="forward") 
    x = np.fft.fft(x, axis=-2, norm="forward") 
    x = np.fft.fft(x, axis=-3, norm="forward") 
    x = np.fft.fft(x, axis=-4, norm="forward") 

    # Permute dimensions (equivalent to `permute(0, 1, 5, 2, 3, 4)`)
    x = np.moveaxis(x, [5, 2, 3, 4], [2, 3, 4, 5])  # Swap dimensions

    return np.squeeze(x)  # Remove the added dimension


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
        # genral 
        self.seqLen = seqLen
        self.mode = trainMode
        self.rootPath = rootPath
        self.trainPath = trainDataPath

        # augmentation
        self.augment = False
        self.augment_prob = 0.3  # Percentage of samples that get augmented
        self.mask_ratio = 0.2
        self.augmentations = [
            self.random_zero_mask,
            self.add_gaussian_noise,
            self.random_time_masking
        ]  # List of available augmentations
        
        if self.mode == "train": 
            self.participants = ["p3", 'p4', "p3" 'p5', 'p6', 'p7', 'p8', "p9", 'p10', "p11"] #['p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', "p11"]
            self.angles = ["an0", "an1" , "an2"]
            self.actions = ["ac1", "ac2", "ac3","ac4","ac5","ac6","ac7","ac8", "ac9"]#, "ac10", "ac11"]#, "ac10", "ac11"]#, "ac10","ac11"]  #["ac0", "ac1" "ac2", "ac3","ac4","ac5","ac6","ac7","ac8","ac9","ac10","ac11"] 
            self.recording = ["r0", "r1"]

             
            
        if self.mode == "val":
            self.participants = ["p12"]
            self.angles = ["an0", "an1", "an2"]
            self.actions = ["ac1", "ac2", "ac3","ac4","ac5","ac6","ac7","ac8","ac9"] #, "ac10", "ac11"] #, "ac10", "ac11"]#, "ac10","ac11"] #["ac0", "ac1", "ac2", "ac3","ac4","ac5","ac6","ac7","ac8","ac9", "ac10","ac11"]
            self.recording = ["r0", "r1"]
            
            
        if self.mode == "test":
            self.participants = ["p1", "p2", "p12"] # 1, 2, 12
            self.angles = ["an0"] #, "an1", "an2"]
            self.actions = ["ac0","ac1","ac2","ac3","ac4","ac5","ac6","ac7","ac8","ac9"] #,"ac10","ac11"]
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
                    
                    
                    for b in range(radar.shape[0]- self.seqLen): # <--  use for overlapping sequences
                    #for b in range(0, radar.shape[0] - self.seqLen, int(self.seqLen/2)):
                        radarOut = radar[b:b + self.seqLen]
                        #radarOut = radar[b]
                        radarTorch = torch.from_numpy(radarOut)
                        radarTorch = radarTorch.to(torch.complex64)
                        #radarTorch = preProcess(radarTorch)
                        
                        # save radar
                        os.makedirs(os.path.join(self.trainPath, self.mode, "radar"), exist_ok=True)
                        os.makedirs(os.path.join(self.trainPath, self.mode, "target"), exist_ok= True)
                        torch.save(radarTorch, os.path.join(self.trainPath, self.mode, "radar", str(counter) + ".pth"))
                        
                        # save target
                        # get gt
                        gt = torch.from_numpy(dataTarget[b + self.seqLen])
                        #gt = torch.from_numpy(dataTarget[b])
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
        
        # load gt 
        if self.mode == "train":
            # load gt into ram 
            self.gt = self.load_directory_into_ram(os.path.join(self.trainPath, "train", "target"))
        
        if self.mode == "val":
            self.gt = self.load_directory_into_ram(os.path.join(self.trainPath, "val", "target"))

                           
    def __len__(self):
        return self.counter
    
    def load_directory_into_ram(self, directory_path: str):
        """
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
    
    def random_zero_mask(self, radar):
        """Randomly masks parts of the tensor with zeros while keeping the original size."""
        D, A, E, R, T = radar.shape
        d_mask = int(D * self.mask_ratio)
        a_mask = int(A * self.mask_ratio)
        e_mask = int(E * self.mask_ratio)
        r_mask = int(R * self.mask_ratio)

        d_start = random.randint(0, max(0, D - d_mask)) if d_mask > 0 else 0
        a_start = random.randint(0, max(0, A - a_mask)) if a_mask > 0 else 0
        e_start = random.randint(0, max(0, E - e_mask)) if e_mask > 0 else 0
        r_start = random.randint(0, max(0, R - r_mask)) if r_mask > 0 else 0

        mask = torch.ones_like(radar)
        mask[d_start:d_start + d_mask, a_start:a_start + a_mask, 
             e_start:e_start + e_mask, r_start:r_start + r_mask, :] = 0

        return radar * mask  # Apply masking

    def add_gaussian_noise(self, radar, noise_std=0.05):
        """Adds Gaussian noise (size remains unchanged)."""
        noise = torch.randn_like(radar) * noise_std
        return radar + noise

    def random_time_masking(self, radar, mask_prob=0.2):
        """Randomly masks some time steps (size remains unchanged)."""
        mask = (torch.rand(radar.shape[-1]) > mask_prob).float()
        return radar * mask.view(1, 1, 1, 1, -1)  # Apply across all dimensions


    def __getitem__(self, idx: int) -> torch.Tensor:
        """ method to get train and validation data

        Args:
            idx (int): indx of data 

        Returns:
            torch.Tensor: model input and target
        """
        
        # get radar 
        radar = torch.load(os.path.join(os.path.join(self.rootPath, self.mode, "radar", str(idx) + ".pth")), weights_only=True)


        # Apply augmentation only with probability `augment_prob`
        if self.augment and random.random() < self.augment_prob and self.mode == "train":
            augmentation = random.choice(self.augmentations)  # Select one augmentation randomly
            radar = augmentation(radar)  # Apply the selected augmentation

        
        #gt = torch.load(os.path.join(os.path.join(self.rootPath, self.mode, "target", str(idx) + ".pth")), weights_only=True)
        gt = self.gt[idx]
        gt = gt.flatten()
        return radar, gt