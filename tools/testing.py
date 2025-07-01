import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import sys
from config import *
import trainLoop

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
                preds.append(mu)
                vars.append(var)

        preds = torch.cat(preds, dim=0)
        vars = torch.cat(vars, dim=0)
        gts = target[self.seq_len:].reshape(-1, 26 * 3).to(self.device)

        error, std = self.compute_mpjpe(preds, gts, keypoints)
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
        self.model = RadProPoserVAE().to(self.device)
        self.model = trainLoop.loadCheckpoint(self.model, None, model_path)

        all_preds, all_gts, all_vars = [], [], []

        for part in parts:
            for angle in angles:
                combos = [[part, angle, act, rep] for act in actions for rep in reps]
                error, std, preds, vars, gts = self.evaluate(combos)

                all_preds.append(preds.cpu().numpy())
                all_gts.append(gts.cpu().numpy())
                all_vars.append(vars.cpu().numpy())

        os.makedirs("prediction_data", exist_ok=True)
        np.save("prediction_data/all_predictions_validation.npy", np.concatenate(all_preds, axis=0))
        np.save("prediction_data/all_ground_truths_validation.npy", np.concatenate(all_gts, axis=0))
        np.save("prediction_data/all_sigmas_validation.npy", np.concatenate(all_vars, axis=0))


if __name__ == "__main__":
    tester = RadarPoseTester(root_path=PATHRAW)

    tester.run_evaluation(
        parts=["p1"],
        angles=["an0", "an1", "an2"],
        actions=["ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9"],
        reps=["r0", "r1"],
        model_path=os.path.join(HPECKPT, "RadProPoserVAE3")
    )