import torch
import torch.nn as nn 
import numpy as np
import os
import trainLoop
from tqdm import tqdm
from config import *
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd


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
        mpjpe = dist
        #mpjpe = dist.mean(dim = 0)
        std = dist.std(dim = 0)
    
    return mpjpe#, std

def testSeq(combination: list[str], # part, ang, act, rec
            rootPath: str, 
            model: nn.Module,
            seqLen: int = SEQLEN, 
            device: str = TRAINCONFIG["device"], 
            keypointsFlag: bool = True, 
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
        return None, None
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
    error  = MPJPE(preds, gt, keypointsFlag)
    
    return error, sigmas # MPJPE, std of errors, learned vars
    
    

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
    sigmas = []
    for comb in testSetList:
        mean, sigma = testSeq(comb, rootPath, model, keypointsFlag=keypointsFlag)
        losses.append(mean)
        sigmas.append(sigma)

    #removeNone = lambda lst: [item for item in lst if item is not None]

    #losses = removeNone(losses)
    #sigmas = removeNone(sigmas)


    #error = torch.mean(torch.stack(losses, dim = 0), dim = 0)
    #sigmas = torch.mean(torch.stack(sigmas, dim = 0), dim = 0)

    return losses, sigmas


def testingMainKeypoints():
    ## LOAD MODEL HERE
    sys.path.append(MODELPATH)
    from models import RadProPoser as Encoder
    #from models import CNN_LSTM as Encoder
    #from models import HRRadarPose as Encoder
    CF = Encoder().to(TRAINCONFIG["device"])

    # load weights 
    CF = trainLoop.loadCheckpoint(CF, None, os.path.join(HPECKPT, "correct")) # add trained HPE model name here 

    # conditions
    testPart = ["p12", "p2", "p1"]
    angles = ["an0", "an1", "an2"]
    actions = ["ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9"]
    reps = ["r0", "r1"]


    # testing
    combos = []
    for part in testPart:
        for an in angles:
            for ac in actions:
                for r in reps:
                    combos.append([part, an, ac, r])

                    # start saving preictions and uncertainties


    res = []
    for i, elem in enumerate(combos):                  
        errors, sigmas = testSeq(elem,
                            PATHRAW,
                            CF)
        
        #assert errors.size(0) == sigmas.size(0)
        
        res.append([combos[i], errors, sigmas]) # output is list of lists for correlation analysis

                 
    # Save as a pickle file
    with open("correlationAnalysis/res.pckl", "wb") as f:
        pickle.dump(res, f)


def plot_error_uncertainty_correlation(errors, uncertainties, title="Correlation Between Error and Uncertainty"):
    """
    Plots a scatterplot of errors vs. uncertainties with an optional trend line.
    
    Parameters:
    - errors: List or NumPy array of error values (y-axis)
    - uncertainties: List or NumPy array of uncertainty values (x-axis)
    - title: Title of the plot (default is "Correlation Between Error and Uncertainty")
    """
    plt.figure(figsize=(8, 6))

    # Set Times New Roman font
    plt.rcParams["font.family"] = "serif"
    
    # Scatter plot
    sns.scatterplot(x=uncertainties, y=errors, alpha=0.5, edgecolor=None, s= 20)
    
    # Regression trend line
    sns.regplot(x=uncertainties, y=errors, scatter=False, color="red", ci=None, line_kws={'linewidth': 2})
    
    # Labels and title
    plt.xlabel("Uncertainty", fontsize = 16)
    plt.ylabel("MPJPE (cm)", fontsize = 16)
    #plt.title(title)
    
    # Grid
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Show plot
    plt.show()


def plot_histogram_with_density(data1, data2, data3, bins=30, alpha=0.5, title="Distribution with KDE Fit"):
    """
    Plots three histograms with KDE density curves and transparency for overlapping distributions.
    
    Parameters:
    - data1: First NumPy array (e.g., errors)
    - data2: Second NumPy array (e.g., uncertainties)
    - data3: Third NumPy array (e.g., another variable)
    - bins: Number of bins for histograms (default: 30)
    - alpha: Transparency level (default: 0.5)
    - title: Plot title (default: "Distribution with KDE Fit")
    """
    plt.figure(figsize=(8, 6))

    # Set distinct colors
    colors = sns.color_palette("Set1", n_colors=3)

    # Plot histogram & KDE for data1
    sns.histplot(data1, bins=bins, kde=True, color=colors[0], alpha=alpha, label="Data 1", edgecolor=None)

    # Plot histogram & KDE for data2
    sns.histplot(data2, bins=bins, kde=True, color=colors[1], alpha=alpha, label="Data 2", edgecolor=None)

    # Plot histogram & KDE for data3
    sns.histplot(data3, bins=bins, kde=True, color=colors[2], alpha=alpha, label="Data 3", edgecolor=None)

    # Labels and title
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(title, fontsize=16)

    # Legend
    plt.legend()

    # Show plot
    plt.show()



def exponential_fit(x, a, b):
    """Exponential function y = a * exp(b * x)"""
    return a * np.exp(b * x)

def power_law_fit(x, a, b):
    return a * x**b

def log_fit(x, a, b):
    return a + b * np.log(x)

def plot_exponential_regression(errors, uncertainties, title="Exponential Regression Between Error and Uncertainty"):
    plt.figure(figsize=(8, 6))

    # Set smaller scatter points
    sns.scatterplot(x=uncertainties, y=errors, color = "green", alpha=0.01, edgecolor="black", s=20)

    # Fit an exponential function
    popt, _ = curve_fit(exponential_fit, uncertainties, errors, maxfev=100000)

    # Compute predicted values for given uncertainties
    y_pred = exponential_fit(uncertainties, *popt)

    # Compute R² score
    r2 = r2_score(errors, y_pred)

    print(r2)

    # Generate fitted values
    x_fit = np.linspace(min(uncertainties), max(uncertainties), 100)
    y_fit = exponential_fit(x_fit, *popt)

    # Plot the exponential regression line
    #plt.plot(x_fit, y_fit, color="black", linewidth=2) #, label=f"Exp Fit: {popt[0]:.2f} * exp({popt[1]:.2f}x)")

    # Labels and title
    plt.xlabel("Uncertainty", fontsize=14, fontname="Serif")
    plt.ylabel("Error", fontsize=14, fontname="Serif")
    #plt.title(title, fontsize=16, fontname="Serif")

    # Grid
    plt.grid(True, linestyle="--", alpha=0.6)
    #plt.legend()

    # Show plot
    plt.show()


def plotCorrelation():
    
    # load pickled data
    with open("correlationAnalysis/res.pckl", "rb") as f:
        res = pickle.load(f)
    

    print(len(res[1][0]))
    print(len(res))
    errors = []
    sigmas = []

    for i, elem in enumerate(res):
        if elem[1] == None:
            continue
        else:
            errors.append(elem[1])
            sigmas.append(elem[2])

    # create two lnog arrays
    errors = torch.cat(errors, dim = 0).flatten().detach().cpu().numpy()
    sigmas = torch.cat(sigmas, dim = 0).flatten().detach().cpu().numpy()
    
    #plot_error_uncertainty_correlation(errors, sigmas)
    plot_exponential_regression(errors, sigmas)

def plotDistributions():
    # load pickled data
    with open("correlationAnalysis/res.pckl", "rb") as f:
        res = pickle.load(f)
    
    
    sigmas0 = []
    sigmas1 = []
    sigmas2 = []

    for i, elem in enumerate(res):
        
        if elem[1] == None:
            continue
        else:
            if res[i][0][i][1] == "an0":
                sigmas0.append(elem[2])
            elif res[i][0][i][1] == "an1":
                sigmas1.append(elem[2])
            elif res[i][0][i][1] == "an2":
                sigmas2.append(elem[2])
            

    # create two lnog arrays
    sigmas0 = torch.cat(sigmas0, dim = 0).flatten().detach().cpu().numpy()
    sigmas1 = torch.cat(sigmas1, dim = 0).flatten().detach().cpu().numpy()
    sigmas2 = torch.cat(sigmas2, dim = 0).flatten().detach().cpu().numpy()

    anova_with_posthoc(sigmas0, sigmas1, sigmas2)

    plot_histogram_with_density(sigmas0, sigmas1, sigmas2)

    
def anova_with_posthoc(data1, data2, data3):
    """
    Performs ANOVA and Tukey's HSD post-hoc test on three NumPy arrays.
    
    Parameters:
    - data1, data2, data3: NumPy arrays (each representing a group)
    
    Returns:
    - Prints ANOVA results (F-statistic, p-value)
    - Prints Tukey's HSD post-hoc test results
    """
    # Combine data into a Pandas DataFrame
    df = pd.DataFrame({
        "value": np.concatenate([data1, data2, data3]),
        "group": ["Group1"] * len(data1) + ["Group2"] * len(data2) + ["Group3"] * len(data3)
    })

    # ✅ 1. Perform One-Way ANOVA
    f_stat, p_value = stats.f_oneway(data1, data2, data3)
    print(f"🔹 ANOVA Results: F-statistic = {f_stat:.4f}, P-value = {p_value:.4f}")

    # ✅ 2. Perform Tukey's HSD post-hoc test if ANOVA is significant
    if p_value < 0.05:
        print("\n✅ Significant difference found! Performing post-hoc Tukey's HSD test...")
        tukey = pairwise_tukeyhsd(df["value"], df["group"])
        print(tukey)
    else:
        print("\n❌ No significant difference found, no need for post-hoc test.")

if __name__ == "__main__":
    #testingMainKeypoints()
    plotCorrelation()
    #plotDistributions()
                        
