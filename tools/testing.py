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
            return (None,) * 5

        batch_size = 32
        preds, vars = [], []

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, radar.size(0) - self.seq_len, batch_size), desc=f"Predicting {'_'.join(combination)}"):
                end = min(i + batch_size, radar.size(0) - self.seq_len)
                batch = torch.stack([radar[j:j+self.seq_len] for j in range(i, end)])
                _, _, _, mu, var = self.model(batch)
                #out = self.model(batch)
                #mu = out[0]
                #var = evidential_uncertainty(out[1], out[2], out[3])
                preds.append(mu)
                vars.append(var)

        preds = torch.cat(preds, dim=0)
        vars = torch.cat(vars, dim=0)
        gts = target[self.seq_len:].reshape(-1, 26 * 3).to(self.device)

        error, std = self.compute_p_mpjpe(preds, gts, keypoints)
        print(error)
        return error, std, preds, vars, gts

    def evaluate(self, test_set: list[list[str]], keypoints: bool = False):
        losses, stds, preds, vars, gts = [], [], [], [], []

        for comb in test_set:
            mean, std, pred, var, gt = self.predict_sequence(comb, keypoints)
            if mean is not None:
                losses.append(mean)
                stds.append(std)
                preds.append(pred)
                vars.append(var)
                gts.append(gt)

        error = torch.mean(torch.stack(losses), dim=0)
        std = torch.mean(torch.stack(stds), dim=0)
        preds = torch.cat(preds, dim=0)
        vars = torch.cat(vars, dim=0)
        gts = torch.cat(gts, dim=0)

        return error, std, preds, vars, gts

    def run_evaluation(self, parts, angles, actions, reps, model_path: str):
        sys.path.append(MODELPATH)
        from vae_lstm_ho import RadProPoserVAE
        #from vae_lstm_ho import CNN_LSTM
        #from evidential_pose_regression import RadProPoserEvidential
        self.model = RadProPoserVAE().to(self.device)
        #self.model = CNN_LSTM().to(self.device)
        #self.model = RadProPoserEvidential().to(self.device)
        self.model = trainLoop.loadCheckpoint(self.model, None, model_path)
        self.model.eval()
        all_preds, all_gts, all_vars = [], [], []
        results_by_participant = {}

        for part in parts:
            results_by_participant[part] = {}
            for angle in angles:
                combos = [[part, angle, act, rep] for act in actions for rep in reps]
                error, std, preds, vars, gts = self.evaluate(combos)

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

        os.makedirs("prediction_data", exist_ok=True)
        #np.save("prediction_data/all_predictions_validation_gaussian_laplace.npy", np.concatenate(all_preds, axis=0))
        #np.save("prediction_data/all_ground_truths_validation_gaussian_laplace.npy", np.concatenate(all_gts, axis=0))
        #np.save("prediction_data/all_sigmas_validation_gaussian_laplace.npy", np.concatenate(all_vars, axis=0))

        return results_by_participant

if __name__ == "__main__":
    tester = RadarPoseTester(root_path=PATHRAW)

    res = tester.run_evaluation(
        parts= ["p1", "p2", "p12"],
        angles=["an0", "an1", "an2"],
        actions=["ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9"],
        reps=["r0", "r1"],
        model_path=os.path.join(HPECKPT, "/home/jonas/code/RadProPoser/trainedModels/humanPoseEstimation/RPP_Gauss_gauss_final9")
    )

    print(res)