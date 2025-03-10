# PISC_InvONet

## Introduction
PISC_InvONet (Physics-informed Self-adaptive Convolutional Inversion Operator Network) is a seismic velocity inversion method that integrates physics-based forward operators with self-adaptive deep learning techniques. This method reduces reliance on large datasets and initial models by leveraging transfer learning and iterative updates.

This repository contains the complete implementation of PISC_InvONet, including data preprocessing, model training, evaluation, and visualization.

---

## Repository Structure

### **PISC_InvONet**
This folder contains the core implementation.

- **Config/** – Configuration files for parameters and paths.
- **data/** – Seismic velocity profiles.
- **Functions/** – Utility functions for data loading, visualization, and mathematical operations (fully commented).
- **Models/** – Includes neural network models, physics-based forward operators, and essential network components:
  - `PhySimulator.py`: Pre-implemented forward operator for seismic simulations.
  - `network.py`: Customizable neural network architecture.
- **results/** – Stores generated inversion results.
- **Trains/** – Main training scripts:
  - `invnet_models/`: Stores results for physics-informed models.
  - `data_generate.py`: Generates training data.
  - `GAN_train.py`: Trains the GAN-based inversion model.
  - `network.py`: Defines GAN architecture.
  - `traditional_FWI.py`: Implements traditional Full-Waveform Inversion (FWI).
  - `transforms/`: Mathematical transformations.
  - `utils/`: Utility functions.

---

### **OpenFWI**
This folder contains additional resources and datasets for seismic inversion.

- **FWI/** – Stores physics-driven inversion results.
- **invert_models/** – Stores data-driven inversion results.
- **phy_data/** – Contains data for physics-driven experiments.
- **split_suzy/** – Stores dataset paths in `.txt` format for model training.
- **conv2former/** – Implementation of convolution-modulated models.
- **dataset/** – Data loading classes.
- **dataset_config.json** – Configuration for different dataset types.
- **DeformConv/** – Implementation of deformable convolutions.
- **FWI/** – Physics-guided inversion models.
- **HWD/** – Wavelet downsampling functions.
- **myselect/** – Grid search and hyperparameter tuning scripts.
- **network/** – All neural network models. Modify this to implement custom architectures.
- **pytorch_ssim/** – Structural similarity index calculation library.
- **Scheduler/** – Learning rate scheduler.
- **shannon/** – Shannon entropy and SI/GSI metric computation.
- **test/** – Network performance evaluation scripts.
- **train/** – Main training scripts.
- **transforms/** – Data transformations (cropping, padding, normalization).
- **utils/** – Various helper functions.
- **vis/** – Visualization functions.

---

## How to Run

### **1. Configure Paths and Parameters**
Modify the parameter and path settings in the `Config` folder (`param` and `path`).

### **2. Generate Training Data**
```bash
python data_generate.py
