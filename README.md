
# RadProPoser: Uncertainty-Aware Human Pose Estimation and Activity Classification from Raw Radar Data
 
<div align="center">
  <img src="assets/overview.png" alt="Project Overview" width="1000">
</div>

## Folder Structure
The folder structure of this repository is organized as follows:

```
RadProPoser/
├── data/                  
│   ├── raw_radar_data/     
│   │   ├── radar 
│   │   ├── skeletons
│   │   ├── README.md  
├── models                   
├── tools 
├── trainedModels 
├── requirements.txt             

```

---

## Installation
A working installation of CUDA should be provided if GPU should be used. We used python 3.12 for the analysis.
To get started with the project, follow the steps below:

1. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

2. **Install torch with cuda_12.4**:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download data**:
   Our data can be downloaded anonymously here: https://doi.org/10.5281/zenodo.14035103. Place our dataset in the `/data` folder. Refer to `radar_raw_data/README.md` for dataset details.

---

## Radar-based Human Pose Estimation

### 1. Configuration
Before training or testing, configure the model in `tools/config.py`:
- Set `MODELNAME` to the model you want to train. default is the best performing model as described in the paper
- Set `PATHORIGIN` to your data root path
- Set `PATHRAW` to your raw radar data path

The dataloader currently includes all training participants, including the original validation participant, because the hyperparameters are already optimized and we aim for maximum performance on the held-out test set. If you want to tune your own hyperparameters, remove the validation participant from the training set to restore a proper validation split and track generalization each epoch.

### 2. Training
The training data (unfolded sequences) is automatically generated based on specified parameters and directories in `config.py`. Around 1.7 to 2 TB of free disk space should be available. The model is trained using Weights and Biases (wandb). You need to create a wandb account.

Create data and train the model:
```bash
python tools/trainScript.py
```

### 3. Testing
Run evaluation on the test set. Results are saved to `tools/calibration_analysis/`:
```bash
python tools/testing.py
```

### 4. Calibration Analysis and Recalibration
After testing, run recalibration analysis. This fits recalibration models on validation data (p1) and evaluates on test data (p2, p12). Results and plots are saved to `tools/calibration_plots/` and `tools/calibrated_models/`:
```bash
python tools/calibration_test.py
```




