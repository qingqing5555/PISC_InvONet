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

### **3.Run Traditional Full-Waveform Inversion (FWI)**
```bash
python traditional_FWI.py

### **4.Train GAN-based Physics-Guided Inversion Model**Note: Adjust network hyperparameters based on different velocity profile sizes (check comments in the code).
```bash
python GAN_train.py

## **Additional Features**

### **For OpenFWI:**
- **Download Data:**  The seismic datasets are accessible as follows: CurveVel-B dataset at OpenFWI (Doi: 10.48550/arXiv.2111.02926), Simulate-SEG-Salt dataset at SEG Data Repository (Doi: 10.1190/geo2018-0249.1), and the Marmousi model at the WDC for Geophysics, Beijing (Doi: 10.1190/1.1437051).
- **Generate Custom Data:** Use `FWI.py` to generate seismic data or `data_generate.py` from PI-InversionNet.
- **Modify Network Architecture:** Implement custom models in `network.py`.

### **Train a New Model:**
```bash
python train.py
```
Modify dataset and model parameters in `README.md` and update the loss function to combine physics and data loss.

### **Use the Forward Operator:**
Refer to `FWI.py` (`generate_data`) or `train.py` (`train_one_epoch`) for integration.

### **Test a Trained Model:**
```bash
python test.py
```
Adjust parameters accordingly.

---

## **Requirements**

Ensure you have the following dependencies installed:
```bash
pip install numpy scipy torch torchvision matplotlib tqdm h5py
```
Other dependencies may include:
```bash
pip install pytorch_ssim scikit-learn openfwi
```
Alternatively, you can install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## **Results**
The trained model and experimental results can be found in the `results/` directory.  
The model outputs **seismic velocity inversion maps** and **intermediate processing results**.
