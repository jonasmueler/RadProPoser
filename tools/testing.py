import torch
import torch.nn as nn 
import numpy as np
import os
import trainLoop
from tqdm import tqdm
from config import *
import sys

def MPJPE(preds: torch.Tensor, 
          targets: torch.Tensor, 
          keypoints = False):
    """
    Calculate the Mean Per Joint Position Error (MPJPE).

    Args:
        preds (torch.Tensor): Predicted joint positions, 
            shape (batch_size, num_joints, 3). Each joint position is a 3D 
            coordinate (x, y, z).
        targets (torch.Tensor): Ground truth joint positions, 
            shape (batch_size, num_joints, 3). Must match the shape of `preds`.
        keypoints (bool): If False, calculates and returns the overall MPJPE 
            for all joints combined. If True, calculates MPJPE per keypoint 
            separately.

    Returns:
        Tuple[float, float or torch.Tensor]: 
            - mpjpe: The mean per joint position error. If `keypoints` is False, 
              this is a single float value. If `keypoints` is True, it is a 
              tensor containing errors for each keypoint.
            - std: The standard deviation of the error, providing an uncertainty 
              measure. If `keypoints` is False, this is a single float value. If 
              `keypoints` is True, it is a tensor of standard deviations per keypoint.
    """
    assert preds.shape == targets.shape, "Predictions and targets must have the same shape"
    
    # Calculate the mean per joint position error
    if keypoints == False:
        # Calculate the Euclidean distance between predicted and target joint positions
        preds = preds.reshape(preds.size(0), 26, 3)
        #preds = preds - preds[:, 3, :].unsqueeze(dim = 1)

        targets = targets.reshape(preds.size(0), 26, 3)
        #targets = targets - targets[:, 3, :].unsqueeze(dim = 1)


        diff = preds - targets
        dist = torch.norm(diff, dim=-1)

        mpjpe = dist.mean()
        std = dist.std()
    if keypoints == True:
        # Calculate the Euclidean distance between predicted and target joint positions
        diff = preds.reshape(preds.size(0), 26, 3) - targets.reshape(preds.size(0), 26, 3)
        
        dist = torch.norm(diff, dim=-1)

        mpjpe = dist.mean(dim = 0)
        std = dist.std(dim = 0)
    
    return mpjpe, std

def testSeq(combination: list[str], # part, ang, act, rec
            rootPath: str, 
            model: nn.Module,
            seqLen: int = SEQLEN, 
            device: str = TRAINCONFIG["device"], 
            keypointsFlag: bool = False, 
            returnPreds: bool = False) -> torch.Tensor:
    """
    Evaluate a sequence of radar data using a specified model to predict keypoints 
    and compute error metrics.

    Parameters:
    - combination (list[str]): A list of identifiers that specify the part, angle, 
      activity, and recording to locate the corresponding radar and skeleton data.
    - rootPath (str): The root directory path containing radar and skeleton data files.
    - model (nn.Module): The PyTorch model used to predict keypoints from radar data.
    - seqLen (int, optional): The sequence length to use for predictions. Defaults to SEQLEN.
    - device (str, optional): The device (e.g., 'cuda' or 'cpu') for computation. Defaults to TRAINCONFIG["device"].
    - keypointsFlag (bool, optional): Flag to indicate whether to use specific keypoint-based error metrics. Defaults to False.
    - returnPreds (bool, optional): If True, return predictions along with error metrics. Defaults to False.

    Returns:
    - torch.Tensor: If returnPreds is False, returns a tuple (error, std, sigmas) where:
        - error: Mean Per Joint Position Error (MPJPE) between predicted and ground truth keypoints.
        - std: Standard deviation of errors.
        - sigmas: Learned variance estimates from the model.
      If returnPreds is True, also returns:
        - preds: Predicted keypoints.
    """
    
    # get radar data
    path = os.path.join(rootPath, "radar",  "data_cube_parsed" + "_" + combination[0] + "_" + combination[1]  +  "_" + combination[2]+ "_" + combination[3] + ".npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    radar = data.files[0]
    radar = torch.from_numpy(data[radar]).to(device).to(torch.complex64)

    # get prediction
    print("start predicting ", combination)
    with torch.no_grad():
        preds = []
        sigmas = []
        model.eval()
        for i in tqdm(range(radar.size(0) - seqLen)):
            inpt = radar[i:i+seqLen].unsqueeze(dim = 0) 
            _,_,_,pred, sigma = model(inpt)
            preds.append(pred)
            sigmas.append(sigma)

    preds = torch.stack(preds, dim = 0).squeeze()
    sigmas = torch.stack(sigmas, dim = 0).squeeze()
    sigmas = torch.sum(sigmas.reshape(sigmas.size(0), 26, 3), dim = 2).squeeze()

    print("predictions done")

    # get gt 
    pathTarget = os.path.join(rootPath, "skeletons",  "skeleton" + "_" + combination[0]  + "_" + combination[1]  +  "_" + combination[2] + "_" + combination[3] + ".npy")
    if not os.path.exists(pathTarget):
        return None, None, None
    dataTarget = np.load(pathTarget)
    dataTarget = torch.from_numpy(dataTarget.reshape(dataTarget.shape[0], 26, 3)) * 100
    gt = dataTarget[seqLen:, :, :].flatten(start_dim = 1, end_dim = -1).cuda() # change to 5 again TODO

    
    # get MPJPE
    error, std = MPJPE(preds, gt, keypointsFlag)
    

    if returnPreds:
        return error, std, sigmas, preds
    else:
        return error, std, sigmas # MPJPE, std of errors, learned vars
    
    

def testLoss(testSetList: list[list[str]], 
             rootPath: str, 
             model: nn.Module,
             keypointsFlag: bool) -> float:
    """
    Calculate the average test loss, standard deviation, and learned variances over a test dataset.

    Parameters:
    - testSetList (list[list[str]]): A list of combinations, where each combination specifies the 
      part, angle, activity, and recording for locating the corresponding radar and skeleton data.
    - rootPath (str): The root directory path containing radar and skeleton data files.
    - model (nn.Module): The PyTorch model used for predicting keypoints from radar data.
    - keypointsFlag (bool): Flag to indicate whether to use specific keypoint-based error metrics.

    Returns:
    - float: Returns a tuple (error, std, sigmas) where:
        - error: The mean of the Mean Per Joint Position Errors (MPJPE) across all test sequences.
        - std: The mean of the standard deviations of errors across all test sequences.
        - sigmas: The mean of the learned variances across all test sequences.
    """
    
    losses = []
    stds = []
    sigmas = []
    for comb in testSetList:
        mean, std, sigma = testSeq(comb, rootPath, model, keypointsFlag=keypointsFlag)
        losses.append(mean)
        stds.append(std)
        sigmas.append(sigma)

    removeNone = lambda lst: [item for item in lst if item is not None]

    losses = removeNone(losses)
    stds = removeNone(stds)
    sigmas = removeNone(sigmas)


    error = torch.mean(torch.stack(losses, dim = 0), dim = 0)
    std = torch.mean(torch.stack(stds, dim = 0), dim = 0)
    sigmas = torch.mean(torch.stack(sigmas, dim = 0), dim = 0)

    return error, std, sigmas


def testingMainKeypoints():
    ## LOAD MODEL HERE
    sys.path.append(MODELPATH)
    from models import RadProPoser as Encoder
    #from models import CNN_LSTM as Encoder
    #from models import HRRadarPose as Encoder
    CF = Encoder().to(TRAINCONFIG["device"])

    # load weights 
    CF = trainLoop.loadCheckpoint(CF, None, os.path.join(HPECKPT, None)) # add trained HPE model name here 

    # conditions
    testPart = ["p12", "p2", "p1"]
    angles = ["an0", "an1", "an2"]
    actions = ["ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9"]
    reps = ["r0", "r1"]


    # prepare test run 
    resDict = {"p12": {"an0": [], "an1": [], "an2": []}, 
               "p2": {"an0": [], "an1": [], "an2": []}, 
               "p1": {"an0": [], "an1": [], "an2": []}} 
    
    sigmaDict = {"p12": {"an0": [], "an1": [], "an2": []}, 
               "p2": {"an0": [], "an1": [], "an2": []}, 
               "p1": {"an0": [], "an1": [], "an2": []}} 


    # testing
    for part in testPart:
        for an in angles:
            combos = []
            for ac in actions:
                for r in reps:
                    combos.append([part, an, ac, r])
            out, _, sigmas = testLoss(combos,
                                PATHORIGIN,
                                CF, 
                                keypointsFlag = False)
            
            # save in dictionary
            resDict[part][an] = out
            sigmaDict[part][an] = sigmas
    
    print("errors ", resDict)
    print("uncertainties ", sigmaDict)


if __name__ == "__main__":
    testingMainKeypoints()
    
                        
