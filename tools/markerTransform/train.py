import os
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.space import Continuous, Categorical
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import BayesianRidge
import numpy as np
import visualCheck
from sklearn_genetic import GASearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn_genetic.space import Integer
import time

def extract_pattern_number(filename):
    try:
        parts = filename.split('_')
        # Find the parts that match the pattern
        p = next(int(part[1:]) for part in parts if part.startswith('p'))
        an = next(int(part[2:]) for part in parts if part.startswith('an'))
        ac = next(int(part[2:]) for part in parts if part.startswith('ac'))
        r = next(int(part[1:]) for part in parts if part.startswith('r'))
        return (p, an, ac, r)
    except (ValueError, StopIteration):
        # Fallback to alphabetical sorting if pattern can't be parsed
        return filename
    
def load_and_concat_npy_files(directory, sub = "ac0"):
    """
    Load all .npy files from a directory and concatenate them along the first dimension.

    Args:
        directory (str): Path to the directory containing .npy files.

    Returns:
        numpy.ndarray: Concatenated array from all .npy files.
    """
    # List all .npy files in the directory that contain 'ac0' in their name
    npy_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy') and sub in f and "an0" in f],
        key=extract_pattern_number
    )
    
    if not npy_files:
        raise ValueError("No .npy files found in the directory.")
    
    # Load each .npy file into a list of arrays
    arrays = [np.load(file) for file in npy_files]
    
    # Concatenate all arrays along the first dimension
    concatenated_array = np.concatenate(arrays, axis=0)
    
    return concatenated_array


def load_and_concat_npy_files_first(directory):
    """
    Load all .npy files from a directory and concatenate them along the first dimension.

    Args:
        directory (str): Path to the directory containing .npy files.

    Returns:
        numpy.ndarray: Concatenated array from all .npy files.
    """
    # List all .npy files in the directory
    #npy_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    npy_files = sorted(
    [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')],
    key=extract_pattern_number
)
   
    
    if not npy_files:
        raise ValueError("No .npy files found in the directory.")
    
    # Load each .npy file into a list of arrays
    arrays = [np.load(file) for file in npy_files]
    
    # Concatenate all arrays along the first dimension
    concatenated_array = np.concatenate(arrays, axis=0)
    
    return concatenated_array


def fit_bayesian_ridge(X_train, y_train, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, 
                       fit_intercept=True, compute_score=False, max_iter=300, tol=0.001):
    """
    Fit a BayesianRidge model to the training data with specified hyperparameters.
    
    Args:
        X_train (numpy.ndarray): Training features (shape: [n_samples, n_features]).
        y_train (numpy.ndarray): Training targets (shape: [n_samples, n_targets] or [n_samples]).
        alpha_1 (float): Hyperparameter for the precision of the weights' prior (Gamma distribution).
        alpha_2 (float): Hyperparameter for the precision of the noise's prior (Gamma distribution).
        lambda_1 (float): Hyperparameter for the precision of the weights' prior (Gamma distribution).
        lambda_2 (float): Hyperparameter for the precision of the noise's prior (Gamma distribution).
        fit_intercept (bool): Whether to calculate the intercept for the model.
        compute_score (bool): Whether to compute the objective function at each iteration.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
    
    Returns:
        BayesianRidge: The trained BayesianRidge model.
    """
    # Initialize the BayesianRidge model with the given hyperparameters
    """
    model = BayesianRidge(
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        fit_intercept=fit_intercept,
        compute_score=compute_score,
        max_iter=max_iter,
        tol=tol
    )
    """

    model = MultiOutputRegressor(BayesianRidge())
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    return model


def genetic_optimize_bayesian_ridge(X, y):
    """
    Use GASearchCV to optimize hyperparameters for BayesianRidge with cross-validation.

    Args:
        X (numpy.ndarray): Features for training (shape: [n_samples, n_features]).
        y (numpy.ndarray): Target values (shape: [n_samples]).

    Returns:
        dict: Dictionary containing the best hyperparameters, the corresponding model, 
              and the genetic search object for further analysis.
    """
    # Define the hyperparameter search space
    param_grid = {
    'estimator__alpha_1': Continuous(1e-10, 1e0, distribution='log-uniform'),
    'estimator__alpha_2': Continuous(1e-10, 1e0, distribution='log-uniform'),
    'estimator__lambda_1': Continuous(1e-10, 1e0, distribution='log-uniform'),
    'estimator__lambda_2': Continuous(1e-10, 1e0, distribution='log-uniform'),
    'estimator__tol': Continuous(1e-8, 1e-1, distribution='log-uniform'),
        }
    

    # Define cross-validation strategy
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # Initialize BayesianRidge model
    bayesian_ridge = MultiOutputRegressor(BayesianRidge(), n_jobs = 8)
   

    # Initialize the GASearchCV optimizer
    
    evolved_estimator = GASearchCV(
        estimator=bayesian_ridge,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        population_size=10,          # Number of individuals in each generation
        generations=1,              # Number of generations to evolve
        tournament_size=3,           # Tournament size for selection
        elitism=True,                # Preserve the best individuals
        crossover_probability=0.8,   # Probability of crossover
        mutation_probability=0.1,    # Probability of mutation
        criteria='max',              # Maximize the fitness score
        algorithm='eaMuPlusLambda',  # Evolutionary algorithm
        n_jobs=4,                   # Parallelize the evaluation
        verbose=True,                # Enable detailed output
        keep_top_k=4,                # Keep the top 4 solutions
    )
    

    # Fit the GASearchCV optimizer to the data
    evolved_estimator.fit(X, y)

    # Retrieve the best model and parameters
    best_model = evolved_estimator.best_estimator_
    best_params = evolved_estimator.best_params_

    # Return results
    return {
        'best_model': best_model,
        'best_params': best_params,
        'evolved_estimator': evolved_estimator
    }
    

def train():
    # load X and y 
    X = load_and_concat_npy_files("/home/jonas/data/radarPose/radar_raw_data/matched_skeletons", "ac0")
    y = load_and_concat_npy_files("/home/jonas/data/radarPose/radar_raw_data/matched_markers", "ac0")
    X = X.reshape(X.shape[0], -1)#[0:500]
    y = y.reshape(y.shape[0], -1)#[0:500]

    print(X.shape[0])
    assert X.shape[0] == y.shape[0]

    # get train and test
    criterion = 0.8
    l = X.shape[0]
    Xtrain = X[0:int(criterion*l)]
    ytrain = y[0:int(criterion*l)]
    Xtest = X[int(criterion*l):]
    ytest = y[int(criterion*l):]

    start_time = time.time()
    res = genetic_optimize_bayesian_ridge(Xtrain, ytrain)
    #res = fit_bayesian_ridge(Xtrain, ytrain)
    #inferenceModel = res
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")


    inferenceModel = res["best_model"]
    preds = inferenceModel.predict(Xtest)

    #plot = visualCheck.plot_skeleton_3d_with_points_time(Xtest.reshape(Xtest.shape[0], 26, 3), preds.reshape(preds.shape[0], 39, 3), save_path = os.getcwd() + "/test.mp4")

    return inferenceModel

def test(model):
    # load X and y 
    X = load_and_concat_npy_files("/home/jonas/data/radarPose/radar_raw_data/matched_skeletons", 
                                  "ac8")
    y = load_and_concat_npy_files("/home/jonas/data/radarPose/radar_raw_data/matched_markers", 
                                  "ac8")
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    
    preds = model.predict(X)
    plot = visualCheck.plot_skeleton_3d_with_points_time(X.reshape(X.shape[0], 26, 3), preds.reshape(preds.shape[0], 39, 3), save_path = os.getcwd() + "/test.mp4")


if __name__ == "__main__":
    model = train()
    test(model)

