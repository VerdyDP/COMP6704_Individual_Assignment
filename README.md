# COMP6704_Individual_Assignment
## Overview
This project implements a **Federated Learning (FL)** experiment on the **MNIST** dataset using **PyTorch**, comparing **Non-DP FedAvg** with **Differentially Private (DP) FedAvg** using DP-SGD. The experiments study the effect of different **gradient clipping bounds (C)** and **Gaussian noise scales (σ)** on model performance and privacy.

The main features include:
- A simple **CNN model** for MNIST classification.
- **Splitting the MNIST dataset** across multiple clients to simulate a federated learning environment.
- **Local DP-SGD training** with gradient clipping and Gaussian noise addition.
- **FedAvg aggregation** to combine models from multiple clients.
- **RDP-based privacy accounting** to estimate the privacy loss (ε) for each round.
- Comparison of DP and non-DP accuracy over multiple rounds.
- Plotting **accuracy vs federated rounds**.

---

## File
- `comp6704_individualAssign.py`: The main Python script containing the entire experiment.

---

## Usage
Run the experiment with:
```bash
python comp6704_individualAssign.py
```
The script will:
- Load the MNIST dataset and split it across 20 clients.
- Perform Non-DP FedAvg as a baseline.
- Run DP-FL experiments for multiple clipping bounds (C=1.0, 0.5) and noise scales (σ=1.0, 5.0, 7.0).
- Compute ε (epsilon) for each round using RDP accounting.
- Plot accuracy vs federated rounds and save the figure as accuracy_vs_rounds.png.

---

## Experiment Settings
Number of clients: 20
Federated rounds: 20
Batch size: 64 for clients, 256 for testing
Learning rate: 0.01
Clipping bounds: C = 1.0, 0.5
Noise scales: σ = 0.5, 1.0, 2.0
RDP orders: 2 to 127
Delta (δ): 1e-5

---

## Outputs
Console prints:
- Accuracy of Non-DP FedAvg per round.
- Accuracy and ε of DP-FL per round for each (C, σ) combination.
Plot:
- accuracy_vs_rounds.png: Test accuracy curves comparing Non-DP and DP-FL.

---

## Notes
- The DP-FL experiments demonstrate the trade-off between model accuracy and privacy.
- Users can modify C_list and sigma_list to explore other DP settings.
- Ensure a GPU is available for faster training, although the code can run on CPU.

---

## Requirements
- Python 3.8+
- PyTorch 2.x
- torchvision
- numpy
- matplotlib

You can install the required packages via pip:

```bash
pip install torch torchvision numpy matplotlib
```
