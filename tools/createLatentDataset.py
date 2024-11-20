import torch 
import numpy as np
import trainLoop
import os 
from tqdm import tqdm
from config import *
import sys


def filterFilesBySubstrings(directory: str,
                            substrings: str):
    """
    Filters files in the specified directory that contain all the given substrings in their filenames.

    :param directory: The directory where the files are located.
    :param substrings: A list of substrings that should be present in the filenames.
    :return: A list of file paths that match the criteria.
    """
    matchingFiles = []
    
    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if all substrings are in the filename
        if all(substring in filename for substring in substrings):
            matchingFiles.append(os.path.join(directory, filename))
    
    return matchingFiles

# helper 
def oneHotEncodeAction(action: str):
    """
    Converts an action string into a one-hot encoded vector.

    Parameters:
    - action (str): The action string to encode. Must be one of "ac0" through "ac11".

    Returns:
    - torch.Tensor: A 1D tensor of size 12, where the index corresponding to the input
      action is set to 1.0, and all other indices are set to 0.0.

    """
    # Initialize a zero tensor of length equal to the number of actions
    oneHotVector = torch.zeros(9, dtype=torch.float32)
    
    # Map each action string to a one-hot encoded vector
    if action == "ac0":
        oneHotVector[0] = 1.0
    elif action == "ac1":
        oneHotVector[1] = 1.0
    elif action == "ac2":
        oneHotVector[2] = 1.0
    elif action == "ac3":
        oneHotVector[3] = 1.0
    elif action == "ac4":
        oneHotVector[4] = 1.0
    elif action == "ac5":
        oneHotVector[5] = 1.0
    elif action == "ac6":
        oneHotVector[6] = 1.0
    elif action == "ac7":
        oneHotVector[7] = 1.0
    elif action == "ac8":
        oneHotVector[8] = 1.0
    elif action == "ac9":
        oneHotVector[9] = 1.0
    elif action == "ac10":
        oneHotVector[10] = 1.0
    elif action == "ac11":
        oneHotVector[11] = 1.0
    else:
        raise ValueError(f"Unknown action: {action}")

    return oneHotVector



def dataGenerator(alphas: np.ndarray, 
                  reps: int, 
                  train: bool = True):
    """
    Generates labeled latent data samples by processing radar data through a pre-trained model and 
    sampling from the latent space.

    Parameters:
    - alphas (np.ndarray): A numpy array of scaling factors used for sampling in the latent space.
    - reps (int): Number of repetitions for sampling for each combination of inputs.
    - train (bool, optional): Indicates whether the function is generating training data (True) 
      or testing data (False). Defaults to True.

    Workflow:
    1. Load the pre-trained model.
    2. Select participants, angles, actions, and recordings based on the `train` flag.
    3. Check for the existence of radar data files and collect valid combinations.
    4. Process radar data to extract latent space parameters (mu, sigma) using the model.
    5. Generate samples from the latent space for each combination, scaled by `alphas`.
    6. Create one-hot encoded labels for the action associated with each combination.
    7. Save the generated samples and their labels to designated train/test directories.

    """
    ## LOAD MODEL HERE
    sys.path.append(MODELPATH)
    from models import RadProPoser as Encoder
    #from models import CNN_LSTM as Encoder
    #from models import HRRadarPose as Encoder
    CF = Encoder().to(TRAINCONFIG["device"])

    # load weights 
    #CF = trainLoop.loadCheckpoint(CF, None, MODELCKTPT)


    if train == True:
        participants = [ 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', "p11"]
        angles = ["an0", "an1", "an2"]
        actions = ["ac1", "ac2", "ac3","ac4","ac5","ac6","ac7","ac8", "ac9"]
        recordings = ["r0", "r1"]
    else: 
        participants = ["p1", "p2", "p12"]
        angles = ["an0", "an1", "an2"]
        actions = ["ac1", "ac2", "ac3","ac4","ac5","ac6","ac7","ac8", "ac9"] 
        recordings = ["r0", "r1"]


    trials = []
    counter = 0
    for participant in participants:
        for angle in angles:
            for action in actions:
                for recording in recordings:
                    path = os.path.join(PATHRAW, "radar",  "data_cube_parsed" + "_" + participant + "_" + angle  +  "_" + action+ "_" + recording + ".npz")
                    if os.path.exists(path):
                        counter += 1
                    trials.append([participant, angle, action, recording])
                    
    # start generating data                
    counter = 0
    for i in tqdm(range(len(trials))):
        combination = trials[i]

        # check if file exists 
        path = os.path.join(PATHRAW, "radar",  "data_cube_parsed" + "_" + combination[0] + "_" + combination[1]  +  "_" + combination[2]+ "_" + combination[3] + ".npz")
        #print(path)
        if os.path.exists(path):
            # get radar 
            data = np.load(path)
            radar = data.files[0]
            radar = data[radar]
            
            radar = torch.from_numpy(radar).to(TRAINCONFIG["device"]).to(torch.complex64) # debug [0:10, ...]

            # get prediction
            print("start predicting ", combination)
            mus = []
            sigmas = []
            with torch.no_grad():
                CF.eval()
                for i in range(radar.size(0) - SEQLEN):
                    inpt = radar[i:i+SEQLEN].unsqueeze(dim = 0)
                    mu, sigma = CF.getLatent(inpt)
                    mus.append(mu)
                    sigmas.append(sigma)
            
            # start sampling data
            for a in alphas:
                for r in range(reps):
                    samples = []
                    for i in range(radar.size(0) - SEQLEN):
                        sample = CF.sample(mus[i], sigmas[i], a)
                        samples.append(sample)
                    
                    # samples 
                    samples = torch.stack(samples, dim = 0)
                    
                    # get gt 
                    label = torch.tensor(oneHotEncodeAction(combination[2]))
                    
                    counter += 1

                    # save data
                    if train == True: 
                        ## create folders 
                        os.makedirs(os.path.join(PATHLATENT, "dataTrain", "X"), exist_ok=True)
                        os.makedirs(os.path.join(PATHLATENT, "dataTrain", "targets"), exist_ok=True)

                        torch.save(samples, os.path.join(PATHLATENT, "dataTrain", "X", str(counter) + ".pth"))
                        torch.save(label, os.path.join(PATHLATENT, "dataTrain", "targets", str(counter) + ".pth"))
                    else:
                        ## create folders 
                        os.makedirs(os.path.join(PATHLATENT, "dataTest", "X"), exist_ok=True)
                        os.makedirs(os.path.join(PATHLATENT, "dataTest", "targets"), exist_ok=True)

                        torch.save(samples, os.path.join(PATHLATENT, "dataTest", "X", str(counter) + ".pth"))
                        torch.save(label, os.path.join(PATHLATENT, "dataTest", "targets", str(counter) + ".pth"))


if __name__ == "__main__":
    alphas = list(np.linspace(0.0128 - 0.01, 0.0128 + 0.01, 10))
    alphas.append(0.0128)
    dataGenerator(alphas, 100, False)