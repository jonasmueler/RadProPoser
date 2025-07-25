import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import sys
from config import *
import trainLoop
import joblib
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
from scipy.interpolate import interp1d
import glob
from torch.optim import Adam
import wandb

from torch.utils.data import Dataset, DataLoader


# import packages
sys.path.append(MODELPATH)
from vae_lstm_ho import RadProPoserVAE



class RadPoser:
    def __init__(self,  device: str = "cuda"):

        self.device = device
        self.root_path = PATHRAW
        self.model = RadProPoserVAE().to(self.device)
        self.model = trainLoop.loadCheckpoint(self.model, 
                                              None,
                                              os.path.join(HPECKPT, 
                                              "/home/jonas/code/RadProPoser/trainedModels/humanPoseEstimation/RPP_gaussian_gaussian/correct")).cuda()
        self.model.eval()
        self.models = None
        self.models = {
            "isotonic": [IsotonicRegression(out_of_bounds="clip") for _ in range(78)],
        }
        self.load_models("/home/jonas/code/RadProPoser/tools/calibrated_models")

    def load_models(self, directory: str):
        """
        Load each model-type’s .pkl file from `directory/` back into self.models.
        Expects files like `isotonic.pkl`, `rf100.pkl`, etc.
        """
        for model_type in list(self.models.keys()):
            path = os.path.join(directory, f"{model_type}.pkl")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Could not find calibration file '{path}'")
            self.models[model_type] = joblib.load(path)
        print(f"Loaded {len(self.models)} recalibration models types from '{directory}'")

    def load_sequence(self, combination: list[str]):
        radar_path = os.path.join(self.root_path, "radar", f"data_cube_parsed_{'_'.join(combination)}.npz")
        skeleton_path = os.path.join(self.root_path, "skeletons", f"skeleton_{'_'.join(combination)}.npy")

        if not os.path.exists(radar_path) or not os.path.exists(skeleton_path):
            return None, None

        radar_npz = np.load(radar_path)
        radar_tensor = torch.from_numpy(radar_npz[radar_npz.files[0]]).to(self.device).to(torch.complex64)

        skeleton_tensor = torch.from_numpy(np.load(skeleton_path)).float()
        skeleton_tensor = skeleton_tensor.view(skeleton_tensor.shape[0], 26, 3) * 100

        return radar_tensor, skeleton_tensor

    def predict_sequence(self, combo):
        # Load and run your VAE model
        radar, gt = self.load_sequence(combo)
        if radar is None:
            return None, None, None

        # Ensure inputs are torch tensors on device
        radar = radar.to(self.device)

        preds, vars_ = [], []
        with torch.no_grad():
            print("start predicting from radar")
            batch_size = 32  # adjust as needed or import from config
            # batch slide windows of length SEQLEN
            total = radar.size(0) - SEQLEN
            for start in tqdm(range(0, total, batch_size), desc="Predicting sequence in batches"):
                end = min(start + batch_size, total)
                # build input batch of shape (end-start, SEQLEN, ...)
                batch = torch.stack([radar[i:i+SEQLEN] for i in range(start, end)], dim=0)
                _, _, _,mu_batch, var_batch = self.model(batch)
                # convert to numpy and flatten dims
                mu_np = mu_batch.cpu().numpy().reshape(mu_batch.size(0), -1)
                var_np = var_batch.cpu().numpy().reshape(var_batch.size(0), -1)
                preds.extend(mu_np)
                vars_.extend(var_np)

        # Stack into (T,78) numpy arrays
        preds_np = np.stack(preds, axis=0)
        vars_np  = np.stack(vars_, axis=0)
        T, D = preds_np.shape


        """
        # Prepare output arrays
        recal_means = np.zeros_like(preds_np)
        recal_vars  = np.zeros_like(vars_np)

        # Compute recalibrated moments per sample & dim
        print("start recalibrating")
        for i in tqdm(range(T)):
            for d, iso_model in enumerate(self.models['isotonic']):
                mu_i    = preds_np[i, d]
                sigma_i = np.sqrt(vars_np[i, d])
                # y-grid
                y = np.linspace(mu_i - 4*sigma_i, mu_i + 4*sigma_i, 200)
                # original CDF
                u = norm.cdf((y - mu_i) / sigma_i)
                # recalibrated CDF
                F_cal = np.clip(iso_model.predict(u.reshape(-1,1)), 0.0, 1.0)
                # inverse CDF
                Q = interp1d(F_cal.flatten(), y, bounds_error=False, fill_value=(y[0], y[-1]))
                # quantile integration
                p = np.linspace(0.005, 0.995, 200)
                q = Q(p)
                m_i = np.trapz(q, p)
                v_i = np.trapz((q - m_i)**2, p)
                recal_means[i, d] = m_i
                recal_vars[i, d] = v_i

        # Reshape back to (T,26,3) numpy arrays and return
        recal_means = recal_means.reshape(T, 26, 3)
        recal_vars  = recal_vars.reshape(T, 26, 3)

        """

        #return preds_np, recal_vars, gt[SEQLEN:]
        return preds_np, vars_np, gt[SEQLEN:]
    
    def run_on_sequences(self, sequences: list[list[str]], out_dir: str = "results"):
            """
            Runs predict_sequence over multiple combos and saves outputs.
            Returns dict mapping combo -> (means, vars).
            """
            os.makedirs(out_dir, exist_ok=True)
            results = {}
            for combo in sequences:
                print("start with ", combo)
                recal_means, recal_vars, gt = self.predict_sequence(combo)
                if recal_means is None:
                    print(f"Skipping {combo}: no data found")
                    continue
                key = tuple(combo)
                results[key] = (recal_means, recal_vars)
                fname = '_'.join(combo)
                np.save(os.path.join(out_dir, f"{fname}_means.npy"), recal_means)
                np.save(os.path.join(out_dir, f"{fname}_vars.npy"), recal_vars)
                np.save(os.path.join(out_dir, f"{fname}_gt.npy"), gt)
                print(f"Saved {fname} to {out_dir}")
            return results

def create_data():
    parts   = ["p3", "p4", "p5", "p6"]#["p1", "p2", "p12"]
    angles  = ["an0", "an1", "an2"]
    actions = ["ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9"]
    reps    = ["r0", "r1"]

    sequences = [[part, angle, act, rep]
                 for part in parts
                 for angle in angles
                 for act in actions
                 for rep in reps]

    rp = RadPoser(device="cuda")
    results = rp.run_on_sequences(sequences, out_dir="results_recalibrated_test")
    print("Batch processing complete. Processed", len(results), "sequences.")


class ParticipantDataLoader:
    """
    Loads saved predictions (means), variances, and ground truths
    for a given participant from disk, only if all three files are present.
    """
    def __init__(self, data_dir: str = "results"):
        self.data_dir = data_dir

    def load_saved_participant_data(self, participant: str):
        """
        For a given participant ID (e.g. "p1"), load all saved
        *_means.npy, *_vars.npy, *_gt.npy files that exist together.

        Returns:
            all_means (list of np.ndarray): shape (T,26,3)
            all_vars  (list of np.ndarray): shape (T,26,3)
            all_gts   (list of np.ndarray): shape (T,26,3)

        Raises:
            RuntimeError if no complete triplets are found.
        """
        pattern = os.path.join(self.data_dir, f"{participant}_*_*_r?_means.npy")
        mean_paths = sorted(glob.glob(pattern))

        all_means, all_vars, all_gts = [], [], []
        for mean_path in mean_paths:
            base = mean_path[:-10]            # strip off '_means.npy'
            vars_path = base + "_vars.npy"
            gt_path   = base + "_gt.npy"

            # only load if all three exist
            if os.path.isfile(vars_path) and os.path.isfile(gt_path):
                try:
                    means = np.load(mean_path)
                    vars_ = np.load(vars_path)
                    gt    = np.load(gt_path)
                except Exception as e:
                    # skip if any file is corrupt
                    print(f"Warning: failed to load {base}: {e}")
                    continue

                all_means.append(means)
                all_vars.append(vars_)
                all_gts.append(gt)

        if not all_means:
            raise RuntimeError(
                f"No complete mean/var/gt triplets found for '{participant}' in '{self.data_dir}'"
            )

        return all_means, all_vars, all_gts

class SequencePredictionDataset_single(Dataset):
    """
    PyTorch Dataset for loading saved prediction means, variances, and ground truths
    from .npy files, one sequence at a time.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Pattern matches files like "p1_an0_ac1_r0_means.npy"
        pattern = os.path.join(data_dir, "*_r?_means.npy")
        self.samples = []
        for mean_path in sorted(glob.glob(pattern)):
            base = mean_path[:-10]  # strip '_means.npy'
            vars_path = base + "_vars.npy"
            gt_path   = base + "_gt.npy"
            if os.path.isfile(vars_path) and os.path.isfile(gt_path):
                self.samples.append((mean_path, vars_path, gt_path))
        if not self.samples:
            raise RuntimeError(f"No complete mean/var/gt triplets found in '{data_dir}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        mean_path, vars_path, gt_path = self.samples[idx]
        # Load numpy arrays
        means_np = np.load(mean_path)   # shape (T, 26, 3)
        vars_np  = np.load(vars_path)
        gt_np    = np.load(gt_path)
        # Convert to torch tensors
        means = torch.from_numpy(means_np).float()
        vars_ = torch.from_numpy(vars_np).float()
        gt    = torch.from_numpy(gt_np).float()
        return means, vars_, gt
    
class SequentialPairTrainDataset(Dataset):
    """
    Dataset returning sequential (t-1, t) pairs for participants 'p2' and 'p12'.
    """
    def __init__(self, data_dir: str, participants: list[str] = ["p3", "p4", "p5", "p6"]):
        self.data = []

        for participant in participants:
            pattern = os.path.join(data_dir, f"{participant}_*_r?_means.npy")
            for mean_path in sorted(glob.glob(pattern)):
                base = mean_path[:-10]
                vars_path = base + "_vars.npy"
                gt_path   = base + "_gt.npy"
                if not os.path.isfile(vars_path) or not os.path.isfile(gt_path):
                    continue

                means = np.load(mean_path)  # (T, 26, 3)
                vars_ = np.load(vars_path)
                gt    = np.load(gt_path)

                means = means.reshape(len(means), -1)
                vars_ = vars_.reshape(len(vars_), -1)
                gt    = gt.reshape(len(gt), -1)

                for t in range(1, len(means)):
                    self.data.append((
                        means[t-1], vars_[t-1],
                        means[t],   vars_[t],
                        gt[t]
                    ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        m_tm1, v_tm1, m_t, v_t, gt_t = self.data[idx]
        return (
            torch.from_numpy(m_tm1).float(),
            torch.from_numpy(v_tm1).float(),
            torch.from_numpy(m_t).float(),
            torch.from_numpy(v_t).float(),
            torch.from_numpy(gt_t).float(),
        )
    

class SequentialPairValidationDataset(Dataset):
    """
    Dataset returning sequential (t-1, t) pairs for participants 'p2' and 'p12'.
    """
    def __init__(self, data_dir: str, participants: list[str] = ["p2", "p12"]):
        self.data = []

        for participant in participants:
            pattern = os.path.join(data_dir, f"{participant}_*_r?_means.npy")
            for mean_path in sorted(glob.glob(pattern)):
                base = mean_path[:-10]
                vars_path = base + "_vars.npy"
                gt_path   = base + "_gt.npy"
                if not os.path.isfile(vars_path) or not os.path.isfile(gt_path):
                    continue

                means = np.load(mean_path)  # (T, 26, 3)
                vars_ = np.load(vars_path)
                gt    = np.load(gt_path)

                means = means.reshape(len(means), -1)
                vars_ = vars_.reshape(len(vars_), -1)
                gt    = gt.reshape(len(gt), -1)

                for t in range(1, len(means)):
                    self.data.append((
                        means[t-1], vars_[t-1],
                        means[t],   vars_[t],
                        gt[t]
                    ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        m_tm1, v_tm1, m_t, v_t, gt_t = self.data[idx]
        return (
            torch.from_numpy(m_tm1).float(),
            torch.from_numpy(v_tm1).float(),
            torch.from_numpy(m_t).float(),
            torch.from_numpy(v_t).float(),
            torch.from_numpy(gt_t).float(),
        )


def compute_mpjpe(preds: torch.Tensor, targets: torch.Tensor, keypoints: bool = False):
        preds = preds.view(preds.size(0), 26, 3)
        targets = targets.view(targets.size(0), 26, 3)

        diff = preds - targets
        dist = torch.norm(diff, dim=-1)

        if keypoints:
            return dist.mean(dim=0), dist.std(dim=0)
        return dist.mean(), dist.std()


class ResidualMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class BayesianUpdateStep(nn.Module):
    """
    Neural Bayesian update step (computationally stable).

    Predicts prior mean and full prior covariance (diagonal + off-diagonal) from last timestep (t-1),
    and performs a Bayesian update using the measurement at timestep t.

    Inputs:
        - mean_tm1: (B, D) prior mean at t-1
        - var_tm1:  (B, D) prior variance at t-1
        - mean_t:   (B, D) measurement mean at t
        - var_t:    (B, D) measurement variance at t

    Outputs:
        - post_mean: (B, D) updated posterior mean
        - post_cov:  (B, D, D) updated posterior covariance
    """
    def __init__(self, dim: int, hidden_dim: int = 78*4):
        super().__init__()
        self.dim = dim

        # Predict prior mean from (mean_tm1, var_tm1)
        self.prior_mean_net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            #ResidualMLP(hidden_dim),
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim)
        )

        # Predict prior variance (diagonal) from (mean_tm1, var_tm1)
        self.prior_var_net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            #ResidualMLP(hidden_dim),
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
            nn.Softplus()  # ensures positive variance
        )

        # Predict prior off-diagonal from (mean_tm1, var_tm1)
        self.prior_offdiag_net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            #ResidualMLP(hidden_dim),
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim * (dim - 1) // 2)
        )

    def forward(self, mean_tm1, var_tm1, mean_t, var_t):
        B, D = mean_tm1.shape
        assert D == self.dim

        eps = 1e-4  # stability epsilon

        # Concatenate inputs for prior estimation
        x_tm1 = torch.cat([mean_tm1, var_tm1], dim=-1)  # (B, 2D)

        # Predict prior mean and prior covariance components
        mean_prior = self.prior_mean_net(x_tm1)  # (B, D)
        diag_prior = self.prior_var_net(x_tm1) + eps  # (B, D)
        offdiag_prior = self.prior_offdiag_net(x_tm1)  # (B, D*(D-1)//2)

        # Assemble full prior covariance matrix
        tril_idx = torch.tril_indices(D, D, offset=-1)
        P_prior = torch.diag_embed(diag_prior)
        P_prior[:, tril_idx[0], tril_idx[1]] = offdiag_prior
        P_prior = P_prior + P_prior.transpose(-2, -1)
        P_prior = P_prior - torch.diag_embed(torch.diagonal(P_prior, dim1=-2, dim2=-1)) + torch.diag_embed(diag_prior)

        # Stabilize covariance matrix with small diagonal bias if needed
        P_prior = P_prior + eps * torch.eye(D, device=P_prior.device).unsqueeze(0)

        # Measurement precision
        R_inv = torch.diag_embed(1.0 / (var_t + eps))
        P_prior_inv = torch.linalg.inv(P_prior)
        P_post_inv = P_prior_inv + R_inv
        P_post = torch.linalg.inv(P_post_inv)

        # Bayesian mean update
        b = P_prior_inv @ mean_prior.unsqueeze(-1) + R_inv @ mean_t.unsqueeze(-1)
        m_post = (P_post @ b).squeeze(-1)

        return m_post, P_post

def train_and_validate(model, train_loader, val_loader, optimizer, epoch: int, device="cuda"):
    model.train()
    train_loss = 0.0
    counter = 0

    for batch in train_loader:
        mean_tm1, var_tm1, mean_t, var_t, gt_t = [b.to(device) for b in batch]
        post_mean, _ = model(mean_tm1, var_tm1, mean_t, var_t)
        loss, _ = compute_mpjpe(post_mean, gt_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if counter % 100 == 0:
            print(f"Train step {counter}: Loss = {loss.item():.4f}")
        counter += 1

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            mean_tm1, var_tm1, mean_t, var_t, gt_t = [b.to(device) for b in batch]
            post_mean, _ = model(mean_tm1, var_tm1, mean_t, var_t)
            loss, _ = compute_mpjpe(post_mean, gt_t)
            val_loss += loss.item()
    val_loss /= len(val_loader)


    return train_loss, val_loss

def main():
    data_dir = "results_recalibrated_test"
    batch_size = 32
    epochs = 100
    dim = 78

    train_dataset = SequentialPairTrainDataset(data_dir)
    val_dataset = SequentialPairValidationDataset(data_dir, participants=["p2", "p12"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BayesianUpdateStep(dim=dim).to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)


    for epoch in range(epochs):
        
        train_loss, val_loss = train_and_validate(model, train_loader, val_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Train MPJPE: {train_loss:.4f} | Val MPJPE: {val_loss:.4f}")

if __name__ == "__main__":
    main()
    #create_data()

    
