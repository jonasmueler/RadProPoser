## globals
SEQLEN = 8
MODELNAME = "RadProPoser"

############################################### HPE #######################################################################
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

############################################################## Activity Classification #########################################################
CLASSIFIERCONFIG = {"batch_size": 32, 
                    "learning_rate": 0.0001, 
                    "num_epochs": 5000, 
                    "device": "cpu"
          }

# paths
MODELPATH = "/home/jonas/CVPR2025/code/codeFinal/src/models"
MODELCKTPT = None # path to model for testing
PATHORIGIN = "/home/jonas/Dokumente/raw_radar_data"
PATHLATENT = "/home/jonas/Dokumente"


