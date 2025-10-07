import os

## globals
SEQLEN = 8
MODELNAME = None # set the name of the model you want to train  here

############################################### HPE #######################################################################
## model paramas

if MODELNAME in ("RPPgaussianGaussian", "RPPgaussianGaussianCov"):
  # 20, 5 gaussian, 1,1 laplace, laplace gaussian 1, 5
  TRAINCONFIG = {"learningRate": 0.001, 
            "weightDecay": 0.0001,
            "epochs": 24, 
            "batchSize": 16, 
          "optimizer": "adam", 
            "device": "cuda", 
            "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
            "lrDecay": 0.99, 
          "beta": 20, 
          "gamma": 5,  
            "nll": True,
          "evd": False,
            "nf": False
            }

if MODELNAME in ("RPPlaplaceLaplace"):
  TRAINCONFIG = {"learningRate": 0.001, 
            "weightDecay": 0.0001,
            "epochs": 24, 
            "batchSize": 16, 
          "optimizer": "adam", 
            "device": "cuda", 
            "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
            "lrDecay": 0.99, 
          "beta": 1, 
          "gamma": 1, 
            "nll": True,
          "evd": False,
            "nf": False
            }
  
if MODELNAME in ("RPPlaplaceGaussian"):
  TRAINCONFIG = {"learningRate": 0.001, 
            "weightDecay": 0.0001,
            "epochs": 24, 
            "batchSize": 16, 
          "optimizer": "adam", 
            "device": "cuda", 
            "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
            "lrDecay": 0.99, 
          "beta": 1, 
          "gamma": 5, 
            "nll": True,
          "evd": False,
            "nf": False
            }

        
if MODELNAME in ("RPPevidential"):
  TRAINCONFIG = {"learningRate": 0.0001, 
            "weightDecay": 0.0001,
            "epochs": 24, 
            "batchSize": 16, 
            "optimizer": "adam", 
            "device": "cuda", 
            "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
            "lrDecay": 0.99, 
            "lambda":0.001, # 0.001
            "nll": False,
            "evd": True,
            "nf": False
            }


if MODELNAME in ("RPPnormalizingFlow"):
  TRAINCONFIG = {"learningRate": 0.0001, 
            "weightDecay": 0.0001,
            "epochs": 24, 
            "batchSize": 32, 
            "optimizer": "adam", 
            "device": "cuda", 
            "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
            "lrDecay": 0.99, 
            "gamma": 1,
            "beta": 1,
            "nll": False,
            "evd": False, 
            "nf": True
            }

if MODELNAME in ("HoEtAlBaseline"):
  TRAINCONFIG = {"learningRate": 0.0001, 
            "weightDecay": 0.0001,
            "epochs": 40, 
            "batchSize": 16, #16 #64 
            "optimizer": "adam", 
            "device": "cuda", 
            "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
            "lrDecay": 0.99, 
            "nll": False,
            "evd": False,
            "nf": False, 
            "HR_SINGLE": True
            }



# paths
PATHORIGIN = None
MODELPATH = os.path.join(PATHORIGIN, "models")
PATHRAW = None
HPECKPT = os.path.join(PATHORIGIN, "trainedModels", "humanPoseEstimation")
