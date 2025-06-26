import torch
import wandb
import os
import torch.nn as nn
import numpy as np
from config import *
import torch.optim as optim
import torch.nn.functional as F
from edl_pytorch import evidential_regression
from calibration_sharpness import *
from torch.optim.lr_scheduler import SequentialLR, LambdaLR


def MPJPE(preds: torch.Tensor, 
          targets: torch.Tensor)->torch.Tensor:
    """
    Calculate the Mean Per Joint Position Error (MPJPE).
    
    Args:
    preds (torch.Tensor): Predicted joint positions, shape (batch_size, num_joints, 3).
    targets (torch.Tensor): Ground truth joint positions, shape (batch_size, num_joints, 3).
    
    Returns:
    float: The MPJPE for the batch.
    """
    assert preds.shape == targets.shape, "Predictions and targets must have the same shape"
    
    # Calculate the Euclidean distance between predicted and target joint positions
    #diff = preds.reshape(preds.size(0), 14, 3) - targets.reshape(preds.size(0), 14, 3)
    diff = preds.reshape(preds.size(0), 26, 3) - targets.reshape(preds.size(0), 26, 3)
    dist = torch.norm(diff, dim=-1)
    
    # Calculate the mean per joint position error
    mpjpe = dist.mean()
    
    return mpjpe

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
    
def lrLambda(epoch: int):
    return TRAINCONFIG["lrDecay"]

def KLLoss(mu: torch.Tensor, 
           logvar: torch.Tensor)->torch.Tensor:
    klloss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return klloss

def nllLoss(gt: torch.Tensor, 
            mu: torch.Tensor, 
            var: torch.Tensor) -> torch.Tensor:
    """
    Simplified Negative Log-Likelihood (NLL) Loss Function.

    Args:
        gt (torch.Tensor): Ground truth tensor.
        mu (torch.Tensor): Predicted mean tensor.
        var (torch.Tensor): Predicted variance tensor.

    Returns:
        torch.Tensor: Computed NLL loss.
    """
    # Ensure variance is positive to avoid numerical instability
    var = torch.clamp(var, min=1e-6)

    # Compute the NLL loss
    pen = TRAINCONFIG["gamma"] * var.mean()
    loss = ((gt - mu) ** 2 / var).mean() + pen

    return loss, pen

def nllLoss_precision(gt: torch.Tensor, 
                        mu: torch.Tensor, 
                        log_variance: torch.Tensor) -> torch.Tensor:
    """
    NLL loss using predicted log variance, internally converting
    to log precision and using the precision-based Gaussian NLL.

    Args:
        gt (torch.Tensor): Ground truth tensor.
        mu (torch.Tensor): Predicted mean tensor.
        log_variance (torch.Tensor): Predicted log variance (log(sigma^2)).

    Returns:
        torch.Tensor: Scalar NLL loss.
    """
    # Step 1: Convert to variance
    #variance = torch.exp(log_variance)
    variance = log_variance

    # Step 2: Compute precision
    precision = 1.0 / variance

    # Step 3: Compute log precision
    log_precision = torch.log(precision)

    # penalty 
    pen = -(TRAINCONFIG["gamma"] *log_precision.mean())

    # Step 4: Final NLL computation
    nll = (pen + precision * (gt - mu) ** 2).mean() 
    return nll, pen

def evidential_uncertainty(v: torch.Tensor, 
                           alpha: torch.Tensor, 
                           beta: torch.Tensor) -> torch.Tensor:
    # Clamp for numerical stability
    #v = torch.clamp(v, min=1.0)
    #alpha = torch.clamp(alpha, min=1.1)
    #beta = torch.clamp(beta, max=1e6)

    aleatoric = beta / (alpha - 1)
    epistemic = beta / (v * (alpha - 1))

    #return aleatoric + epistemic
    return epistemic 

def trainLoop(trainLoader: torch.utils.data.DataLoader, 
              valLoader: torch.utils.data.DataLoader, 
              model: nn.Module,
              loadModel: bool, 
              modelName: str, 
              params: dict,  
              WandB: bool, 
              device: str, 
              criterion: nn.Module):
        
    """
    Wrapper function to train a PyTorch nn.Module model with support for checkpoint loading, 
    optimizer and scheduler setup, and logging using Weights & Biases.

    Args:
        trainLoader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        valLoader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        model (nn.Module): The neural network model to be trained.
        loadModel (bool): Flag indicating whether to load a previously saved model checkpoint.
        modelName (str): The name of the model used for saving checkpoints and logging.
        params (dict): Dictionary containing training parameters, including:
                    - "optimizer": Type of optimizer ("adam" or "RMSProp").
                    - "learningRate": Learning rate for the optimizer.
                    - "betas": Betas for the Adam optimizer.
                    - "weightDecay": Weight decay (L2 regularization) factor.
                    - "epochs": Number of training epochs.
        WandB (bool): Flag indicating whether to use Weights & Biases for logging.
        device (str): Device to use for training (e.g., "cpu" or "cuda").
        criterion (nn.Module): Loss function to use when TRAINCONFIG["nll"] is False.

    Functionality:
        - Initializes Weights & Biases (WandB) for experiment tracking if enabled.
        - Sets up the optimizer and learning rate scheduler.
        - Loads a model and optimizer from a checkpoint if loadModel is True.
        - Performs training over the specified number of epochs:
            - Computes the forward pass and loss, using a custom NLL loss if TRAINCONFIG["nll"] is True,
            or a provided criterion otherwise.
            - Applies gradient clipping and updates the model parameters.
            - Logs the training loss to WandB if enabled.
        - Performs validation at the end of each epoch, logging the validation loss to WandB.
        - Saves model and optimizer checkpoints after each epoch.
        - Saves the final model and optimizer state after training is complete.
    """

    # WandB
    if WandB:
        wandb.init(
            # set the wandb project where this run will be logged
            project= modelName,

            # track hyperparameters and run metadata
            config=params
        )

    # get optimizer
    if params["optimizer"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr = params["learningRate"],
                                      betas = params["betas"],
                                      weight_decay= params["weightDecay"])
    if params["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                      lr = params["learningRate"],
                                      betas = params["betas"],
                                      weight_decay= params["weightDecay"])

    # load model and optimizer from checkpoint
    if loadModel:
        # get into folder
        os.chdir(PATHORIGIN + "/models")
        lastState = loadCheckpoint(model, optimizer, PATHORIGIN + "/trainedModels/" + modelName)
        model = lastState[0]
        optimizer = lastState[1]

    # initilaize LR sheduling
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lrLambda)

    ###################### start training #############################
    for b in range(params["epochs"]):
        trainCounter = 0
        for radar,  gt in trainLoader:
            model.train()

            # move data to device and preprocess
            radar = radar.to(device).to(torch.complex64)
            #radar2 = radar.to(device).float() # .to(torch.complex64)
            gt = gt.to(device).float() * 100 # convert to cm
    
            
            # zero the parameter gradients
            optimizer.zero_grad()


            # loss functions
            if TRAINCONFIG["nll"] == True:
                # generator loss
                preds, mu, logVar, muOut, varOut = model.forward(radar)
                KLloss = KLLoss(mu, logVar) 
                nLL, pen = nllLoss(gt, muOut, varOut)
                loss = nLL + TRAINCONFIG["beta"] * KLloss

            elif TRAINCONFIG["nf"] == True:
                    mu, var, kld = model.forward(radar, inference = False)

                    nll, pen = nllLoss_precision(gt, mu, var) 
                    loss = nll + TRAINCONFIG["beta"] * kld
            
            elif TRAINCONFIG["evd"] == True:
                # generator loss
                pred_nig, pred = model(radar)

                loss_uncertainty = evidential_regression(
                        pred_nig,      # predicted Normal Inverse Gamma parameters
                        gt,             # target labels
                        lamb=TRAINCONFIG["lambda"],    # regularization coefficient 
                    )
                
                MSE = criterion(pred, gt)

                loss = 10 * loss_uncertainty + MSE

            else:
                preds = model.forward(radar)
                loss = criterion(preds, gt)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            trainCounter += 1

            # logging 
            if TRAINCONFIG["nll"] == True:
                if WandB:
                        wandb.log({"nll": nLL.detach().cpu().item(), 
                                   "KLLOss": KLloss.detach().cpu().item(), 
                                   "loss": loss.detach().cpu().item(), 
                                   "varPenalty": pen.detach().cpu().item()})
                        
            elif TRAINCONFIG["nf"] == True:
                if WandB:
                        wandb.log({"nll": nll.detach().cpu().item(), 
                                   "loss": loss.detach().cpu().item(), 
                                   "varPen": pen.detach().cpu().item(), 
                                   "KL": kld.detach().cpu().item()})
                        
            elif TRAINCONFIG["evd"] == True:
                if WandB:
                        wandb.log({"evidential_loss": loss_uncertainty, 
                                   "loss": loss, 
                                   "v": pred_nig[1].mean(), 
                                   "alpha": pred_nig[2].mean(), 
                                   "beta": pred_nig[3].mean(), 
                                   "MSE": MSE.detach().cpu().item()})

            else:
                if WandB:
                        wandb.log({"MSE": loss.detach().cpu().item()})

        scheduler.step()
        ########################### Validation Loop ############################
        with torch.no_grad():
            print("start validating model")
            valLossMean = 0
            predictions, variances, gts = [], [], []
            counter = 0
            for x,  y in valLoader:
                model.eval()
                x = x.to(device).to(device).to(torch.complex64)
                y = y.to(device).float() * 100

                if TRAINCONFIG["nll"] == True:
                    _, _, _, preds, logvar = model.forward(x)

                    var = torch.exp(logvar)

                    predictions.append(preds.detach().cpu())
                    variances.append(var.detach().cpu())
                    gts.append(y.detach().cpu())

                
                elif TRAINCONFIG["nf"] == True:
                    preds, var, kld = model.forward(x, inference = True)

                    predictions.append(preds.detach().cpu())
                    variances.append(var.detach().cpu())
                    gts.append(y.detach().cpu())
                
                elif TRAINCONFIG["evd"] == True:
                    # generator loss
                    pred_nig, preds = model(x) # (mu, v, alpha, beta)
                    var = evidential_uncertainty(pred_nig[1], pred_nig[2], pred_nig[3]) # epistemic uncertainty

                    predictions.append(preds.detach().cpu())
                    variances.append(var.detach().cpu())
                    gts.append(y.detach().cpu())
                    

                else:
                    preds = model.forward(x)

                # calculate validation loss
                
                valLoss = MPJPE(preds, y)
                valLossMean += valLoss

                counter += 1
                
                        
            valLosses = valLossMean/counter + 1 

            # calibration scores 
            #mu = torch.cat(predictions, dim = 0)
            #var  = torch.cat(variances, dim = 0)
            #gt = mu = torch.cat(gts, dim = 0)
            #calLoss, sharpness, _ = compute_calibration_and_sharpness(mu, var, gt, 500)

            if WandB:
                    wandb.log({"valLoss": valLosses.detach().cpu().item(), 
                               #"calibration": calLoss, 
                               #"sharpness": sharpness}
                               })
            print("valLoss: ", valLosses.detach().cpu().item())

            # save model and optimizer checkpoint
            path = os.path.join(HPECKPT)
            os.chdir(path)
            saveCheckpoint(model, optimizer, os.path.join(path, modelName + str(b)))

    ## save model state
    saveCheckpoint(model, optimizer, PATHORIGIN + "/" + "trainedModels/" + modelName + str(b))
    print("Model saved!")
    print("Training done!")

