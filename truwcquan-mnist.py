import json
import matplotlib.pyplot as plt

import os

# Load file
filepath = os.path.join("results", "mnist_convex_results.json")
with open(filepath, "r") as f:
    data = json.load(f)

# Lấy danh sách các method
methods = [k for k in data.keys() if k not in ["dataset", "P_star", "config"]]

# =========================
# 1. TRAINING LOSS
# =========================
plt.figure()

for m in methods:
    passes = data[m]["passes"]
    loss = data[m]["training_loss"]
    plt.plot(passes, loss, label=m)

plt.xlabel("Gradient / n (passes)")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Passes")
plt.yscale("log")
plt.legend()
plt.grid()

plt.show()


# =========================
# 2. LOSS - OPTIMUM
# =========================
plt.figure()

for m in methods:
    passes = data[m]["passes"]
    residual = data[m]["loss_residual"]
    plt.plot(passes, residual, label=m)

plt.xlabel("Gradient / n (passes)")
plt.ylabel("Loss - Optimum")
plt.title("Loss Residual vs Passes")
plt.yscale("log")
plt.legend()
plt.grid()

plt.show()


# =========================
# 3. GRADIENT VARIANCE
# =========================
plt.figure()

for m in methods:
    passes = data[m]["passes"]
    var = data[m]["grad_variance"]
    plt.plot(passes, var, label=m)

plt.xlabel("Gradient / n (passes)")
plt.ylabel("Gradient Variance")
plt.title("Gradient Variance vs Passes")
plt.yscale("log")
plt.legend()
plt.grid()

plt.show()