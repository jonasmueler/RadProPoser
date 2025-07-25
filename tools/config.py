import os

## globals
SEQLEN = 8
MODELNAME = "nf"

############################################### HPE #######################################################################
## model paramas
#RadProPoserVAE
# define hyperparameters


"""
# 20, 5 gaussian, 1,1 laplace, laplace gaussian 1, 5
TRAINCONFIG = {"learningRate": 0.001, 
          "weightDecay": 0.0001,
          "epochs": 24, 
          "batchSize": 16, 
         "optimizer": "adam", 
          "device": "cuda", 
          "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
          "lrDecay": 0.99, 
         "beta": 20, #20
         "gamma": 5, # 5 
          "nll": True,
        "evd": False,
          "nf": False
          }
"""
          
"""
#RadProPoserEvidential
# define hyperparameters
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
"""

# radproposer noermalizing flow 
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
          "evd": False,
          "nf": False
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
          "evd": False,
          "nf": False, 
          "HR_SINGLE": True
          }
"""

############################################################## Activity Classification #########################################################
CLASSIFIERCONFIG = {"batch_size": 32, 
                    "learning_rate": 0.001, 
                    "num_epochs": 50, 
                    "device": "cpu"
          }

# paths
PATHORIGIN = "/home/jonas/code/RadProPoser"
MODELPATH = os.path.join(PATHORIGIN, "models")
PATHLATENT = os.path.join(PATHORIGIN, "data", "latentData")
PATHRAW = "/home/jonas/data/radarPose/mmRadPose_rawdata"
ACTIVITYCLASSIFICATIONCKPT = os.path.join(PATHORIGIN, "trainedModels", "activityClassification")
HPECKPT = os.path.join(PATHORIGIN, "trainedModels", "humanPoseEstimation")
