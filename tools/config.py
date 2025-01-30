import os

## globals
SEQLEN = 8
MODELNAME = "RadProPoser"

############################################### HPE #######################################################################
## model paramas
#RadProPoser
# define hyperparameters

TRAINCONFIG = {"learningRate": 0.001, # 0.001 
          "weightDecay": 0.0001,
          "epochs": 300, #12
          "batchSize": 16, #16
          "optimizer": "adam", 
          "device": "cuda", 
          "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
          "lrDecay": 0.99, 
          "beta": 80, #80
          "gamma": 5, #5
          "delta": 1,
          "nll": True,
          }




## CNN-LSTM 
# define hyperparameters

#TRAINCONFIG = {"learningRate": 0.0001, 
#          "weightDecay": 0.0001,
#          "epochs": 40, 
#          "batchSize": 16, #16 #64 
#          "optimizer": "adam", 
#          "device": "cuda", 
#          "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
#          "lrDecay": 0.99, 
#          "nll": False,
#          }

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
                    "learning_rate": 0.0001, 
                    "num_epochs": 5000, 
                    "device": "cuda"
          }

# paths
PATHORIGIN = "/home/jonas/code/bioMechRadar/CVPR2025Replication"
MODELPATH = os.path.join(PATHORIGIN, "models")
PATHLATENT = os.path.join(PATHORIGIN, "data", "latentData")
PATHRAW = "/home/jonas/data/radarPose/radar_raw_data/" #"/home/jonas/data/radarPose/raw_radar_data" #os.path.join(PATHORIGIN, "data", "raw_radar_data")
ACTIVITYCLASSIFICATIONCKPT = os.path.join(PATHORIGIN, "trainedModels", "activityClassification")
HPECKPT = os.path.join(PATHORIGIN, "trainedModels", "humanPoseEstimation")
