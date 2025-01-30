# load model
import os
import torch
from config import *
import sys
import random 
import numpy as np 
import torch.nn as nn
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import pickle
from umap import UMAP

## LOAD MODEL HERE
sys.path.append(MODELPATH)
from models import RadProPoser4 as Encoder
#from models import CNN_LSTM as Encoder
#from models import HRRadarPose as Encoder



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


def pca(data: torch.Tensor, n_components: int):
    """
    Perform PCA on the given data tensor.

    Args:
        data (torch.Tensor): Input data of shape (n_samples, n_features).
        n_components (int): Number of principal components to retain.

    Returns:
        torch.Tensor: Transformed data of shape (n_samples, n_components).
        torch.Tensor: Principal components (eigenvectors) of shape (n_features, n_components).
        torch.Tensor: Explained variances (percent) of shape (n_components,).
    """
    # Step 1: Center the data by subtracting the mean
    mean = data.mean(dim=0)
    centered_data = data - mean

    # Step 2: Perform low-rank PCA
    U, S, V = torch.pca_lowrank(centered_data, q=n_components)

    # Step 3: Project the data onto the principal components
    transformed_data = torch.mm(centered_data, V[:, :n_components])

    # Step 4: Compute explained variances
    total_variance = (S ** 2).sum() / (data.size(0) - 1)
    explained_variances = (S[:n_components] ** 2) / (data.size(0) - 1)
    percent_explained_variance = explained_variances / total_variance * 100

    return transformed_data, V[:, :n_components], percent_explained_variance


def pca(data: torch.Tensor, n_components: int):
    """
    Perform PCA on the given data tensor.

    Args:
        data (torch.Tensor): Input data of shape (n_samples, n_features).
        n_components (int): Number of principal components to retain.

    Returns:
        torch.Tensor: Transformed data of shape (n_samples, n_components).
        torch.Tensor: Principal components (eigenvectors) of shape (n_features, n_components).
        torch.Tensor: Explained variances (percent) of shape (n_components,).
    """
    if data == None:
        return None
    
    else:
        # Step 1: Center the data by subtracting the mean
        mean = data.mean(dim=0)
        centered_data = data - mean

        # Step 2: Perform low-rank PCA
        U, S, V = torch.pca_lowrank(centered_data, q=n_components)

        # Step 3: Project the data onto the principal components
        transformed_data = torch.mm(centered_data, V[:, :n_components])

        # Step 4: Compute explained variances
        total_variance = (S ** 2).sum() / (data.size(0) - 1)
        explained_variances = (S[:n_components] ** 2) / (data.size(0) - 1)
        percent_explained_variance = explained_variances / total_variance * 100

        return transformed_data #, V[:, :n_components], percent_explained_variance

def tsne(data: torch.Tensor, n_components: int, perplexity: float = 50, random_state: int = 42):
    """
    Perform t-SNE on the given data tensor.

    Args:
        data (torch.Tensor): Input data of shape (n_samples, n_features).
        n_components (int): Number of dimensions to project the data into (e.g., 2 for visualization).
        perplexity (float): Perplexity parameter for t-SNE (default: 30.0).
        random_state (int): Random state for reproducibility (default: 42).

    Returns:
        torch.Tensor: Transformed data of shape (n_samples, n_components).
    """
    if data == None:
        return None

    else:
        data_np = data.detach().cpu().numpy()
        tsne_model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        transformed_data = tsne_model.fit_transform(data_np)

        return torch.tensor(transformed_data)

def umap_projection(data: torch.Tensor, n_components: int, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42): # 15, 0.1
    """
    Perform UMAP on the given data tensor.

    Args:
        data (torch.Tensor): Input data of shape (n_samples, n_features).
        n_components (int): Number of dimensions to project the data into (e.g., 2 for visualization).
        n_neighbors (int): Number of neighbors considered for UMAP (default: 15).
        min_dist (float): Minimum distance between points in the low-dimensional space (default: 0.1).
        random_state (int): Random state for reproducibility (default: 42).

    Returns:
        torch.Tensor: Transformed data of shape (n_samples, n_components).
    """
    data_np = data.detach().cpu().numpy()
    umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, n_jobs = -1)
    transformed_data = umap_model.fit_transform(data_np)

    return torch.tensor(transformed_data)

def plot_2d_latent_space(latent_vectors_list, labels_list=None):
    """
    Plot 2D latent space data for multiple exercises, each with a different color.

    Args:
        latent_vectors_list (list of torch.Tensor): List of latent space data, where each tensor has shape (n_samples, 2).
        labels_list (list of str, optional): List of labels corresponding to each latent vector set (for legend).
    """
    plt.figure(figsize=(8, 6))

    # Plot each set of latent vectors with a different color
    for i, latent_vectors in enumerate(latent_vectors_list):
        latent_vectors = latent_vectors.detach().cpu().numpy()
        label = labels_list[i] if labels_list is not None else f"Exercise {i+1}"
        plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], label=label, s=10)

    plt.title('2D Latent Space Visualization')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid()
    plt.legend(title="Exercises")
    plt.show()

def plot_2d_latent_space_subplot_orig(latent_vectors_list, labels_list=None):
    """
    Plot 2D latent space data for multiple exercises, each in a separate subplot, arranged in two rows.

    Args:
        latent_vectors_list (list of torch.Tensor): List of latent space data, where each tensor has shape (n_samples, 2).
        labels_list (list of str, optional): List of labels corresponding to each latent vector set.
    """
    num_exercises = len(latent_vectors_list)
    n_cols = 5  # Number of columns
    n_rows = (num_exercises + 1) // n_cols  # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()  # Flatten to iterate easily

    for i, latent_vectors in enumerate(latent_vectors_list):
        latent_vectors = latent_vectors.detach().cpu().numpy()
        label = labels_list[i] if labels_list is not None else f"Exercise {i+1}"
        axes[i].scatter(latent_vectors[:, 0], latent_vectors[:, 1], s=10)
        axes[i].set_title(label)
        axes[i].set_xlabel('PC1')
        axes[i].set_ylabel('PC2')
        axes[i].grid()

    # Hide any unused subplots
    for j in range(num_exercises, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("latentSpace.pdf", dpi = 1000)
    plt.show()



def plot_2d_latent_space_subplot(latent_vectors_list, participant_names, exercise_names):
    """
    Plot 2D latent space data for multiple exercises, each in a separate subplot, arranged in three rows.

    Args:
        latent_vectors_list (list of lists of torch.Tensor or None): List of latent space data for three participants, 
                                                                     where each participant's data is a list of tensors 
                                                                     (one tensor per exercise), or None if missing.
        participant_names (list of str): List of participant names (e.g., ["Participant 1", "Participant 2", "Participant 3"]).
        exercise_names (list of str): List of exercise names for subplot headers.
    """
    n_exercises = len(latent_vectors_list[0])
    n_cols = 5  # Fixed number of columns for layout
    n_rows = (n_exercises + 1) // n_cols  # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()  # Flatten axes for easier indexing

    for exercise_idx in range(n_exercises):
        ax = axes[exercise_idx]
        valid_data = False  # To check if any valid data is present for this exercise
        for participant_idx, latent_vectors in enumerate(latent_vectors_list):
            if latent_vectors[exercise_idx] is None:
                continue  # Skip if the latent vector is None
            vectors = latent_vectors[exercise_idx].detach().cpu().numpy()
            ax.scatter(vectors[:, 0], vectors[:, 1], label=participant_names[participant_idx], s=10)
            valid_data = True
        if valid_data:
            ax.set_title(exercise_names[exercise_idx], fontsize = 20)
            ax.set_xlabel("dim 1", fontsize = 18)
            ax.set_ylabel("dim 2", fontsize = 18)
            ax.grid()
            ax.legend()
        else:
            ax.axis("off")  # Hide subplot if no valid data

    # Hide any unused subplots
    for j in range(n_exercises, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("participantLatents.pdf", dpi = 1000)
    plt.show()

def plot_2d_latent_space_subplot_scaled(latent_vectors_list, participant_names, exercise_names):
    """
    Plot 2D latent space data for multiple exercises, each in a separate subplot, arranged in three rows.
    Both axes are scaled to the maximum range across all subplots, with increased tick sizes globally.

    Args:
        latent_vectors_list (list of lists of torch.Tensor or None): List of latent space data for three participants, 
                                                                     where each participant's data is a list of tensors 
                                                                     (one tensor per exercise), or None if missing.
        participant_names (list of str): List of participant names (e.g., ["Participant 1", "Participant 2", "Participant 3"]).
        exercise_names (list of str): List of exercise names for subplot headers.
    """
    # Update global font size for tick labels
    plt.rcParams.update({
        'xtick.labelsize': 16,  # Font size for x-axis ticks
        'ytick.labelsize': 16   # Font size for y-axis ticks
    })

    n_exercises = len(latent_vectors_list[0])
    n_cols = 5  # Fixed number of columns for layout
    n_rows = (n_exercises + n_cols - 1) // n_cols  # Calculate rows needed (ceiling division)

    # Compute global min and max for both dimensions across all latent vectors
    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')

    for participant_latents in latent_vectors_list:
        for latent_vectors in participant_latents:
            if latent_vectors is None:
                continue
            vectors = latent_vectors.detach().cpu().numpy()
            global_min_x = min(global_min_x, vectors[:, 0].min())
            global_max_x = max(global_max_x, vectors[:, 0].max())
            global_min_y = min(global_min_y, vectors[:, 1].min())
            global_max_y = max(global_max_y, vectors[:, 1].max())

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()  # Flatten axes for easier indexing
    counter = 0
    for exercise_idx in range(n_exercises):
        ax = axes[exercise_idx]
        valid_data = False  # To check if any valid data is present for this exercise
        for participant_idx, latent_vectors in enumerate(latent_vectors_list):
            if latent_vectors[exercise_idx] is None:
                continue  # Skip if the latent vector is None
            vectors = latent_vectors[exercise_idx].detach().cpu().numpy()
            ax.scatter(vectors[:, 0], vectors[:, 1], label=participant_names[participant_idx], s=10)
            valid_data = True
        if valid_data:
            ax.set_title(exercise_names[exercise_idx], fontsize=20)
            ax.set_xlabel("dim 1", fontsize=18)
            ax.set_ylabel("dim 2", fontsize=18)
            ax.grid()

            if counter == 0:
                ax.legend(fontsize=18)
            # Apply global limits
            ax.set_xlim(global_min_x, global_max_x)
            ax.set_ylim(global_min_y, global_max_y)
        else:
            ax.axis("off")  # Hide subplot if no valid data
        counter += 1

    # Hide any unused subplots
    for j in range(n_exercises, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("an1.pdf", dpi=1000)
    plt.show()

def predict_exercise_sequence(model: torch.nn.Module, path: str) -> torch.tensor:
    if not os.path.exists(path):
        return None

    else:
        # load data
        data = np.load(path)
        radar = data.files[0]
        radar = data[radar]
        radar = torch.from_numpy(radar).to(TRAINCONFIG["device"]).to(torch.complex64) # debug [0:10, ...]

        # get prediction
        print("start predicting ")
        mus = []
        sigmas = []
        with torch.no_grad():
            model.eval()
            for i in range(radar.size(0) - SEQLEN):
                inpt = radar[i:i+SEQLEN].unsqueeze(dim = 0)
                mu, sigma = model.getLatent(inpt)
                mus.append(mu)
                sigmas.append(sigma)
        mus = torch.stack(mus, dim = 0).squeeze()
        print(path, ", done")
        return mus


def test():
    # Load weights
    CF = Encoder().to(TRAINCONFIG["device"])
    path = "/home/jonas/code/bioMechRadar/CVPR2025Replication/trainedModels/humanPoseEstimation/RadProPoser4"
    model = loadCheckpoint(CF, None, path).cuda()

    # Predict sequence and get latent samples
    part = "p3"
    angle = "an1"
    ex = predict_exercise_sequence(
        model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_" + angle + "_ac8_r0.npz"
    )
    samples = ex.squeeze().detach().cpu().numpy().T  # Extract samples from the exercise sequence, and transpose cluster over features

    # Apply Agglomerative Clustering
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN


    # grid search 
    distances = np.linspace(0.01, 50, 1000)
    res = np.zeros((len(distances), 2))
    for i in range(len(distances)): 
        clusterer = DBSCAN(eps=distances[i])
        
        #AgglomerativeClustering(
        #    n_clusters=None,         # Automatically determine number of clusters
        #    linkage="ward",          # Use Ward's method for linkage
        #    distance_threshold=distances[i]   # Set threshold for cluster granularity
        #)
        clusters = clusterer.fit_predict(samples)

        # Evaluate Clustering Performance with Silhouette Score
        if len(set(clusters)) > 1:  # Silhouette score is not defined for 1 cluster
            silhouette = silhouette_score(samples, clusters)
            #print("Number of clusters found:", len(set(clusters)))
            #print("Silhouette Score:", silhouette)
            res[i, 0] = silhouette
            res[i, 1] = distances[i] 
        else:
            print("Only one cluster found, silhouette score cannot be computed.")

    best = np.argmax([res[:, 0]])

    clusterer = DBSCAN(eps=distances[best])
    
    #AgglomerativeClustering(
    #        n_clusters=None,         # Automatically determine number of clusters
    #        linkage="ward",          # Use Ward's method for linkage
    #        distance_threshold=distances[best]   # Set threshold for cluster granularity
    #    )
    clusters = clusterer.fit_predict(samples)

    unique_clusters, counts = np.unique(clusters, return_counts=True)
    print("Cluster labels:", unique_clusters)
    print("Samples per cluster:", counts)
    #print(res)
    print("best combination = ", res[best])

def compare_distributions_means_covs(samples1, samples2, metric="js"):
    """
    Compare two multivariate distributions using their mean vectors and covariance matrices.

    Parameters:
        samples1 (np.ndarray): Samples from the first distribution (N1 x D).
        samples2 (np.ndarray): Samples from the second distribution (N2 x D).
        metric (str): Metric to use ("kl" for Kullback-Leibler, "js" for Jensen-Shannon).

    Returns:
        float: Divergence between the two distributions.
    """
    # Calculate mean vectors and covariance matrices
    mean1, cov1 = np.mean(samples1, axis=0), np.cov(samples1, rowvar=False)
    mean2, cov2 = np.mean(samples2, axis=0), np.cov(samples2, rowvar=False)

    # Compute metrics
    if metric == "kl":
        # KL Divergence for multivariate Gaussians
        kl_div = 0.5 * (
            np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
            - len(mean1)
            + np.trace(np.linalg.solve(cov2, cov1))
            + (mean2 - mean1).T @ np.linalg.solve(cov2, (mean2 - mean1))
        )
        return kl_div

    elif metric == "js":
        
        # JS Divergence for multivariate Gaussians
        mean_avg = 0.5 * (mean1 + mean2)
        cov_avg = 0.5 * (cov1 + cov2)

        # KL Divergences
        kl_div1 = 0.5 * (
            np.log(np.linalg.det(cov_avg) / np.linalg.det(cov1))
            - len(mean1)
            + np.trace(np.linalg.solve(cov_avg, cov1))
            + (mean_avg - mean1).T @ np.linalg.solve(cov_avg, (mean_avg - mean1))
        )

        kl_div2 = 0.5 * (
            np.log(np.linalg.det(cov_avg) / np.linalg.det(cov2))
            - len(mean2)
            + np.trace(np.linalg.solve(cov_avg, cov2))
            + (mean_avg - mean2).T @ np.linalg.solve(cov_avg, (mean_avg - mean2))
        )

        # JS Divergence
        js_div = 0.5 * kl_div1 + 0.5 * kl_div2
        
        return js_div

    else:
        raise ValueError("Invalid metric. Choose 'kl' or 'js'.")


def get_embeddings_participant(part: str,
                                model: nn.Module, 
                                angle = "an1"):

    if not os.path.exists("/home/jonas/code/bioMechRadar/bioMechEstimation/tools/latentPreds/testset/" + part + "_" + angle + ".pkl"):
        # get all 9 exercsies from one participant
        sequences = []
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac1_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac2_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac3_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac4_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac5_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac6_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac7_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac8_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac9_r0.npz")
        sequences.append(ex)

        # Save the list to a file
        with open("/home/jonas/code/bioMechRadar/bioMechEstimation/tools/latentPreds/testset/" + part + "_" + angle + ".pkl", 'wb') as f:
            pickle.dump(sequences, f)
    

    else:
        with open("/home/jonas/code/bioMechRadar/bioMechEstimation/tools/latentPreds/testset/" + part + "_" + angle + ".pkl", 'rb') as f:
            sequences = pickle.load(f)

    reduced = [tsne(sequences[i], 2) for i in range(len(sequences))]

    return reduced


def get_latents(part: str,
                model: nn.Module, 
                angle = "an1"):

    if not os.path.exists("/home/jonas/code/bioMechRadar/bioMechEstimation/tools/latentPreds/" + angle + ".pkl"):
        # get all 9 exercsies from one participant
        sequences = []
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac1_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac2_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac3_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac4_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac5_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac6_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac7_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac8_r0.npz")
        sequences.append(ex)
        ex = predict_exercise_sequence(model, "/home/jonas/data/radarPose/radar_raw_data/radar/data_cube_parsed_" + part + "_"  + angle +  "_ac9_r0.npz")
        sequences.append(ex)

        # Save the list to a file
        with open(angle + ".pkl", 'wb') as f:
            pickle.dump(sequences, f)
    

    else:
        with open("/home/jonas/code/bioMechRadar/bioMechEstimation/tools/latentPreds/" + angle + ".pkl", 'rb') as f:
            sequences = pickle.load(f)


    return sequences


def main():
    # load weights
    CF = Encoder().to(TRAINCONFIG["device"])
    path = "/home/jonas/code/bioMechRadar/CVPR2025Replication/trainedModels/humanPoseEstimation/RadProPoser7"
    model = loadCheckpoint(CF, None, path).cuda()

    reduced_participants = []

    for part in ["p1", "p2", "p12"]:
        reduced = get_embeddings_participant(part, model, "an1")
        reduced_participants.append(reduced)
        print(part, " done")
    


    
    # plot
    plot_2d_latent_space_subplot_scaled(reduced_participants, ["p1", "p2", "p12"], ["left upper limb extensions",
                                                                            "right upper limb extension", 
                                                                            "bilateral upper limb extension", 
                                                                            "bicep curls", 
                                                                            "front arm rotation", 
                                                                            "torso forward bending", 
                                                                            "left front lunge", 
                                                                            "right front lunge", 
                                                                         "squats"])


def mainDistTest():
    # load weights
    CF = Encoder().to(TRAINCONFIG["device"])
    path = "/home/jonas/code/bioMechRadar/CVPR2025Replication/trainedModels/humanPoseEstimation/RadProPoser4"
    model = loadCheckpoint(CF, None, path).cuda()

    participantDist = []

    for an in ["an0", "an1", "an2"]:
        latents = get_embeddings_participant("p3", model, an)
        participantDist.append(latents)
        print(an, " done")

    for i in range(len(participantDist[0])):
        res = compare_distributions_means_covs(participantDist[1][i].squeeze().detach().cpu().numpy(), 
                                               participantDist[2][i].squeeze().detach().cpu().numpy())
        print(res)
    


    
    

if __name__ == "__main__":
    main()
    #test()
    #mainDistTest()



