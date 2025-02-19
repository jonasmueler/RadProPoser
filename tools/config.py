import os

## globals
SEQLEN = 8
MODELNAME = "RadProPoser"

############################################### HPE #######################################################################
## model paramas
#RadProPoser
# define hyperparameters
TRAINCONFIG = {"learningRate": 0.001, 
          "weightDecay": 0.0001,
          "epochs": 12, 
          "batchSize": 16, 
          "optimizer": "adam", 
          "device": "cuda", 
          "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
          "lrDecay": 0.99, 
          "beta": 20, 
          "gamma": 5, 
          "nll": True,
          }



## CNN-LSTM 
# define hyperparameters
"""
TRAINCONFIG = {"learningRate": 0.0001, 
          "weightDecay": 0.0001,
          "epochs": 40, 
          "batchSize": 16, #16 #64 
          "optimizer": "adam", 
          "device": "cuda", 
          "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
          "lrDecay": 0.99, 
          "nll": False,
          }
"""
## Ho et al
# define hyperparameters

"""
TRAINCONFIG = {"learningRate": 0.0001, 
          "weightDecay": 0.0001,
          "epochs": 40, 
          "batchSize": 16, #16 #64 
          "optimizer": "adam", 
          "device": "cuda", 
          "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
          "lrDecay": 0.99, 
          "nll": False,
          }
"""

############################################################## Activity Classification #########################################################
CLASSIFIERCONFIG = {"batch_size": 32, 
                    "learning_rate": 0.001, 
                    "num_epochs": 50, 
                    "device": "cpu"
          }

# paths
PATHORIGIN = None
MODELPATH = os.path.join(PATHORIGIN, "models")
PATHLATENT = os.path.join(PATHORIGIN, "data", "latentData")
PATHRAW = os.path.join(PATHORIGIN, "data", "raw_radar_data")
ACTIVITYCLASSIFICATIONCKPT = os.path.join(PATHORIGIN, "trainedModels", "activityClassification")
HPECKPT = os.path.join(PATHORIGIN, "trainedModels", "humanPoseEstimation")
