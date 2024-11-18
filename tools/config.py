## globals
SEQLEN = 8
MODELNAME = "RadProPoser"

## model paramas
#RadProPoser
# define hyperparameters
TRAINCONFIG = {"learningRate": 0.001, 
          "weightDecay": 0.0001,
          "epochs": 4000, 
          "batchSize": 16, 
          "optimizer": "adam", 
          "device": "cpu", 
          "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
          "lrDecay": 0.99, 
          "beta": 80, 
          "gamma": 5, 
          "nll": True,
          }



## CNN-LSTM 
# define hyperparameters
"""
TRAINCONFIG = {"learningRate": 0.0001, 
          "weightDecay": 0.0001,
          "epochs": 4000, 
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
          "epochs": 4000, 
          "batchSize": 16, #16 #64 
          "optimizer": "adam", 
          "device": "cuda", 
          "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
          "lrDecay": 0.99, 
          "nll": False,
          }
"""


# paths
MODELPATH = "/home/jonas/CVPR2025/code/codeFinal/models"
MODELCKTPT = None # path to model for checking
PATHORIGIN = "/home/jonas/Dokumente/raw_radar_data"


