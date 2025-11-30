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

## Requirements
- Python 3.8+
- PyTorch 2.x
- torchvision
- numpy
- matplotlib

You can install the required packages via pip:

```bash
pip install torch torchvision numpy matplotlib
