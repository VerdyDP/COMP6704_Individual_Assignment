import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt


# ==========================================
# 1. CNN Model
# ==========================================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ==========================================
# 2. Split MNIST dataset into clients
# ==========================================
def split_dataset(dataset, num_clients=20):
    data_per_client = len(dataset) // num_clients
    clients = []
    for i in range(num_clients):
        idx = list(range(i * data_per_client, (i + 1) * data_per_client))
        clients.append(Subset(dataset, idx))
    return clients


# ==========================================
# 3. DP-SGD utilities
# ==========================================
def clip_gradients(model, C):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += (p.grad.data.norm(2).item() ** 2)
    total_norm = total_norm ** 0.5

    coef = min(1.0, C / (total_norm + 1e-6))
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(coef)


def add_gaussian_noise(model, C, sigma, batch_size):
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.normal(
                mean=0.0,
                std=sigma * C,
                size=p.grad.shape,
                device=p.grad.device
            )
            p.grad.data.add_(noise / batch_size)


# ==========================================
# 4. RDP accountant
# ==========================================
def compute_rdp(q, sigma, steps, orders):
    return np.array([steps * q * q * order / (2 * sigma * sigma) for order in orders])


def get_epsilon(orders, rdp, delta):
    eps = rdp - np.log(delta) / (orders - 1)
    return eps.min()


# ==========================================
# 5. Local DP-SGD training
# ==========================================
def local_train(model, loader, lr, C, sigma, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()

        clip_gradients(model, C)
        add_gaussian_noise(model, C, sigma, batch_size=len(x))
        opt.step()

    return model.state_dict()


# ==========================================
# 6. FedAvg aggregation
# ==========================================
def fedavg(models):
    avg = {}
    for key in models[0].keys():
        avg[key] = sum(m[key] for m in models) / len(models)
    return avg


# ==========================================
# 7. Evaluation
# ==========================================
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


# ==========================================
# 8. Main experiment
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    clients = split_dataset(train_set, num_clients=20)
    client_loaders = [DataLoader(c, batch_size=64, shuffle=True) for c in clients]
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    rounds = 20
    C_list = [1.0, 0.5]   # clipping bounds
    sigma_list = [0.5, 1.0, 2.0]
    delta = 1e-5
    orders = np.arange(2, 128)

    acc_results = {}   # (C, sigma) -> list of accuracy

    # -----------------------------
    # Non-DP baseline
    # -----------------------------
    print("\n===== Non-DP FedAvg Baseline =====\n")
    baseline_model = CNN().to(device)
    acc_record_nodp = []

    for r in range(rounds):
        local_models = []
        for loader in client_loaders:
            local_model = CNN().to(device)
            local_model.load_state_dict(baseline_model.state_dict())

            # Non-DP training
            local_model.train()
            opt = torch.optim.SGD(local_model.parameters(), lr=0.01)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(local_model(x), y)
                loss.backward()
                opt.step()

            local_models.append(local_model.state_dict())

        w = fedavg(local_models)
        baseline_model.load_state_dict(w)
        acc = test(baseline_model, test_loader, device)
        acc_record_nodp.append(acc)
        print(f"Non-DP Round {r+1}: accuracy={acc:.4f}")

    # -----------------------------
    # DP-FL experiments
    # -----------------------------
    for C in C_list:
        print(f"\n============================")
        print(f"   Clipping Bound C = {C}")
        print(f"============================")

        for sigma in sigma_list:
            print(f"\n---- sigma = {sigma} ----\n")
            global_model = CNN().to(device)
            rdp_total = np.zeros_like(orders, dtype=float)
            acc_results[(C, sigma)] = []

            for r in range(rounds):
                local_models = []
                for loader in client_loaders:
                    local_model = CNN().to(device)
                    local_model.load_state_dict(global_model.state_dict())
                    w = local_train(local_model, loader, lr=0.01, C=C, sigma=sigma, device=device)
                    local_models.append(w)

                global_weights = fedavg(local_models)
                global_model.load_state_dict(global_weights)

                # Privacy accounting
                q = 64 / 3000
                rdp_total += compute_rdp(q, sigma, steps=1, orders=orders)
                eps = get_epsilon(orders, rdp_total, delta)

                acc = test(global_model, test_loader, device)
                acc_results[(C, sigma)].append(acc)
                print(f"Round {r+1:02d}: accuracy={acc:.4f}, epsilon={eps:.3f}")

    # -----------------------------
    # Plot accuracy vs rounds
    # -----------------------------
    plt.figure(figsize=(8, 5))

    # Non-DP baseline
    plt.plot(range(1, rounds+1), acc_record_nodp, label="Non-DP FedAvg", linewidth=2)

    # DP curves
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'gray']
    line_id = 0
    for C in C_list:
        for sigma in sigma_list:
            label = f"DP (C={C}, σ={sigma})"
            plt.plot(range(1, rounds+1),
                     acc_results[(C, sigma)],
                     label=label,
                     color=colors[line_id % len(colors)],
                     linestyle='--')
            line_id += 1

    plt.xlabel("Federated Rounds")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Rounds (DP-FL vs Non-DP)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_rounds.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
