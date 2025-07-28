import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import sys
from config import *
import trainLoop

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
    return aleatoric 


class RadarPoseTester:
    def __init__(self,root_path: str, seq_len: int = SEQLEN, device: str = TRAINCONFIG["device"]):
        self.root_path = root_path
        self.seq_len = seq_len
        self.device = device

    @staticmethod
    def compute_mpjpe(preds: torch.Tensor, targets: torch.Tensor, keypoints: bool = False):
        preds = preds.view(preds.size(0), 26, 3)
        targets = targets.view(targets.size(0), 26, 3)

        diff = preds - targets
        dist = torch.norm(diff, dim=-1)

        if keypoints:
            return dist.mean(dim=0), dist.std(dim=0)
        return dist.mean(), dist.std()
    
    def compute_p_mpjpe(self, preds: torch.Tensor, targets: torch.Tensor, keypoints: bool = False):
        preds = preds.view(preds.size(0), 26, 3)
        targets = targets.view(targets.size(0), 26, 3)

        aligned_preds = torch.stack([
            self.procrustes_torch(tgt, pred) for pred, tgt in zip(preds, targets)
        ])

        diff = aligned_preds - targets
        dist = torch.norm(diff, dim=-1)

        if keypoints:
            return dist.mean(dim=0), dist.std(dim=0)
        return dist.mean(), dist.std()
    
    def procrustes_torch(self, 
                         X: torch.Tensor, 
                         Y: torch.Tensor, 
                         scaling: bool = True, 
                         reflection: str = 'best'):
        """
        PyTorch implementation of Procrustes analysis. Aligns Y to X.
        Inputs:
            X: [N, 3] - target coordinates
            Y: [N, 3] - input coordinates to align
            scaling: bool - whether to apply scaling
            reflection: 'best' | True | False - control reflection in the transform
        Returns:
            Z: [N, 3] - aligned Y
        """
        muX = X.mean(dim=0, keepdim=True)
        muY = Y.mean(dim=0, keepdim=True)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0 ** 2).sum()
        ssY = (Y0 ** 2).sum()

        normX = torch.sqrt(ssX)
        normY = torch.sqrt(ssY)

        X0 /= normX
        Y0 /= normY

        A = X0.T @ Y0
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = V @ U.T

        if reflection != 'best':
            have_reflection = torch.det(T) < 0
            if reflection != have_reflection:
                V[:, -1] *= -1
                S[-1] *= -1
                T = V @ U.T

        traceTA = S.sum()

        if scaling:
            b = traceTA * normX / normY
            Z = normX * traceTA * (Y0 @ T) + muX
        else:
            b = 1.0
            Z = normY * (Y0 @ T) + muX

        return Z

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

    def predict_sequence(self, combination: list[str], keypoints: bool = False, return_preds: bool = False):
        radar, target = self.load_sequence(combination)
        if radar is None or target is None:
            return (None,) * 6

        batch_size = 32
        preds, vars, cdfs = [], [], []

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, radar.size(0) - self.seq_len, batch_size), desc=f"Predicting {'_'.join(combination)}"):
                end = min(i + batch_size, radar.size(0) - self.seq_len)
                batch = torch.stack([radar[j:j+self.seq_len] for j in range(i, end)])
                _, _, _, mu, var, cdf = self.model(batch)
                preds.append(mu)
                vars.append(var)
                cdfs.append(cdf)


        preds = torch.cat(preds, dim=0)
        vars = torch.cat(vars, dim=0)
        gts = target[self.seq_len:].reshape(-1, 26 * 3).to(self.device)
        cdfs = torch.cat(cdfs, dim= 0)

        print(cdfs.size())

        error, std = self.compute_p_mpjpe(preds, gts, keypoints)
        print(error)
        return error, std, preds, vars, gts, cdfs

    def evaluate(self, test_set: list[list[str]], keypoints: bool = False):
        losses, stds, preds, vars, gts, cdfs = [], [], [], [], [], []

        for comb in test_set:
            mean, std, pred, var, gt, cdf = self.predict_sequence(comb, keypoints)
            if mean is not None:
                losses.append(mean)
                stds.append(std)
                preds.append(pred)
                vars.append(var)
                gts.append(gt)
                cdfs.append(cdf)

        error = torch.mean(torch.stack(losses), dim=0)
        std = torch.mean(torch.stack(stds), dim=0)
        preds = torch.cat(preds, dim=0)
        vars = torch.cat(vars, dim=0)
        gts = torch.cat(gts, dim=0)
        cdfs = torch.cat(cdfs, dim=0)

        return error, std, preds, vars, gts, cdfs

    def evaluate_single(
        self,
        test_set: list[list[str]],
        keypoints: bool = False,
        batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each combination in test_set, load its radar frames and targets,
        predict point estimates in batches of size batch_size (μ only),
        fill variances with zeros, compute p-MPJPE & std per combination,
        then aggregate across all combinations.

        Returns:
            error (torch.Tensor): mean p-MPJPE over all combos
            std   (torch.Tensor): mean std dev over all combos
            preds (torch.Tensor): all μ preds stacked [N_total, 78]
            vars  (torch.Tensor): zero variances [N_total, 78]
            gts   (torch.Tensor): all ground truths [N_total, 78]
        """
        losses, stds = [], []
        all_preds, all_vars, all_gts = [], [], []

        for comb in test_set:
            radar, target = self.load_sequence(comb)
            if radar is None or target is None:
                continue

            preds_batches, gts_batches = [], []

            self.model.eval()
            with torch.no_grad():
                total = radar.size(0)
                for i in tqdm(range(0, total, batch_size), desc=f"Predicting {'_'.join(comb)}"):
                    end = min(i + batch_size, total)
                    batch_frames = radar[i:end].to(self.device)

                    # If your model still requires a time dimension of length 1, uncomment:
                    # batch_in = batch_frames.unsqueeze(1)  # [B,1,C,H,W]
                    # Otherwise feed it directly:
                    batch_in = batch_frames                     # [B,C,H,W]

                    # forward: returns (_,_,_, μ, _)
                    root, offset = self.model(batch_in)
                    preds = root.unsqueeze(dim = 1) + offset
                    preds = torch.cat([root.unsqueeze(dim = 1), preds], dim = 1)
                    preds_batches.append(preds)

                    gts_batch = target[i:end].reshape(-1, 26 * 3).to(self.device)
                    gts_batches.append(gts_batch)

            preds = torch.cat(preds_batches, dim=0)     # [N_frames, 78]
            gts   = torch.cat(gts_batches,   dim=0)     # [N_frames, 78]
            vars_ = torch.zeros_like(preds)             # zeros for var

            # compute p-MPJPE & std for this comb
            error, std = self.compute_mpjpe(preds, gts, keypoints)

            losses.append(error)
            stds.append(std)
            all_preds.append(preds)
            all_vars.append(vars_)
            all_gts.append(gts)

        # aggregate over combinations
        error = torch.mean(torch.stack(losses), dim=0)
        std   = torch.mean(torch.stack(stds),   dim=0)
        preds = torch.cat(all_preds, dim=0)
        vars  = torch.cat(all_vars,  dim=0)
        gts   = torch.cat(all_gts,   dim=0)

        return error, std, preds, vars, gts



    def run_evaluation(self, parts, angles, actions, reps, model_path: str):
        sys.path.append(MODELPATH)
        from vae_lstm_ho import RadProPoserVAE
        #from normalizing_flow import RadProPoserVAE
        #from vae_lstm_ho import CNN_LSTM
        #from vae_lstm_ho import HRRadarPose
        #from evidential_pose_regression import RadProPoserEvidential
        #self.model = RadProPoserVAE().to(self.device)
        self.model = RadProPoserVAE().to(self.device)
        #self.model = CNN_LSTM().to(self.device)
        #self.model = RadProPoserEvidential().to(self.device)
        self.model = trainLoop.loadCheckpoint(self.model, None, model_path)
        self.model.eval()
        all_preds, all_gts, all_vars, all_cdfs = [], [], [], []
        results_by_participant = {}

        for part in parts:
            results_by_participant[part] = {}
            for angle in angles:
                combos = [[part, angle, act, rep] for act in actions for rep in reps]
                error, std, preds, vars, gts, cdfs = self.evaluate(combos)
                

                if isinstance(error, torch.Tensor):
                    error = error.item() if error.numel() == 1 else error.cpu().numpy()
                if isinstance(std, torch.Tensor):
                    std = std.item() if std.numel() == 1 else std.cpu().numpy()

                results_by_participant[part][angle] = {
                    "error": error,
                    "std": std
                }

                all_preds.append(preds.cpu().numpy())
                all_gts.append(gts.cpu().numpy())
                all_vars.append(vars.cpu().numpy())
                all_cdfs.append(cdfs.cpu().numpy())

        os.makedirs("calibration_analysis", exist_ok=True)
        np.save("calibration_analysis/mu_val_gauss_laplace.npy", np.concatenate(all_preds, axis=0))
        np.save("calibration_analysis/gt_val_gauss_laplace.npy", np.concatenate(all_gts, axis=0))
        np.save("calibration_analysis/var_val_gauss_laplace.npy", np.concatenate(all_vars, axis=0))
        np.save("calibration_analysis/cdf_val_gauss_laplace.npy", np.concatenate(all_cdfs, axis=0))

        return results_by_participant

if __name__ == "__main__":
    tester = RadarPoseTester(root_path=PATHRAW)

    res = tester.run_evaluation(
        parts= ["p1"], #["p2", "p12"],#["p1"], #, "p2", "p12"],
        angles=["an0", "an1", "an2"],
        actions=["ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9"],
        reps=["r0", "r1"],
        model_path=os.path.join(HPECKPT, 
                                "/home/jonas/code/RadProPoser/trainedModels/humanPoseEstimation/VAE_Gaussian_Laplace/correct")
    )

    print(res)