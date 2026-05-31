"""
debug_diagnostic_fast.py — Quick diagnostic to validate hypotheses
"""
import os, sys, json, numpy as np
np.random.seed(42)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source'))

from utils.data_loader import load_dataset
from models.logistic import loss_multiclass, full_grad_multiclass, stoch_grad_multiclass
from optimizers.sgd import warm_start

DATASET = 'mnist'
LAM = 1e-4
MULTICLASS = True

print("=" * 70)
print("FAST DIAGNOSTIC: MNIST Convex Results")
print("=" * 70)

X_train, y_train, X_test, y_test = load_dataset(DATASET)
n, d = X_train.shape
K = len(np.unique(y_train))
print(f"Dataset: n={n}, d={d}, K={K}")

with open('results/optimal_loss.json', 'r') as f:
    optimal = json.load(f)
P_star_old = float(optimal[DATASET]['P_star'])
print(f"Current P_star: {P_star_old:.15f}")

# ============================================================================
# TEST 1: P_star verification
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: P_star verification")
print("=" * 70)

w_init = warm_start(X_train, y_train, LAM, MULTICLASS, n_epochs=1, lr=0.01)
loss_init = loss_multiclass(w_init, X_train, y_train, LAM)
print(f"Loss after warm-start (1 epoch SGD): {loss_init:.10f}")
print(f"Residual (w_init - P_star): {loss_init - P_star_old:.6e}")
print(f"--> NEGATIVE = P_star is NOT the true minimum!")

# Quick GD with better settings
print(f"\nRunning quick GD with better params...")
w = np.zeros((d, K))
lr = 0.1
best_loss = float('inf')
for t in range(1000):
    g = full_grad_multiclass(w, X_train, y_train, LAM)
    current_loss = loss_multiclass(w, X_train, y_train, LAM)
    
    # Less aggressive Armijo (c=1e-4 instead of 0.5)
    step = lr
    for _ in range(20):
        w_new = w - step * g
        new_loss = loss_multiclass(w_new, X_train, y_train, LAM)
        if new_loss <= current_loss - 1e-4 * step * np.sum(g * g) or step < 1e-12:
            break
        step *= 0.5
    w = w_new
    if new_loss < best_loss:
        best_loss = new_loss
    if t % 200 == 0:
        print(f"  iter {t:4d}: loss = {new_loss:.10f} (step = {step:.2e})")

P_star_new = loss_multiclass(w, X_train, y_train, LAM)
print(f"\nNew P_star (quick GD): {P_star_new:.10f}")
print(f"Old P_star:            {P_star_old:.10f}")
print(f"Difference:            {P_star_new - P_star_old:.6e}")

# Check if warm-start loss is still below new P_star
loss_init2 = loss_multiclass(w_init, X_train, y_train, LAM)
print(f"\nWarm-start loss vs New P_star: {loss_init2 - P_star_new:.6e}")

# ============================================================================
# TEST 2: SDCA — check with small subset
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: SDCA quick check")
print("=" * 70)

from optimizers.sdca import sdca_epoch_multiclass

n_sub = 500
X_sub = X_train[:n_sub]
y_sub = y_train[:n_sub]
K_sub = len(np.unique(y_sub))

alpha = np.zeros((n_sub, d, K_sub))
W = np.zeros((d, K_sub))
loss0 = loss_multiclass(W, X_sub, y_sub, LAM)
print(f"Loss before SDCA: {loss0:.6f}")

lr_def = 1.0 / (LAM * n_sub)
print(f"Default lr = 1/(lam*n) = {lr_def:.4f}")
alpha, W = sdca_epoch_multiclass(alpha, W, X_sub, y_sub, LAM, n_sub, K_sub, lr_def)
loss1 = loss_multiclass(W, X_sub, y_sub, LAM)
print(f"Loss after 1 epoch (default lr): {loss1:.6f} (change: {loss1-loss0:.4e})")

# Try with much smaller lr
alpha2 = np.zeros((n_sub, d, K_sub))
W2 = np.zeros((d, K_sub))
alpha2, W2 = sdca_epoch_multiclass(alpha2, W2, X_sub, y_sub, LAM, n_sub, K_sub, 0.01)
loss2 = loss_multiclass(W2, X_sub, y_sub, LAM)
print(f"Loss after 1 epoch (lr=0.01):    {loss2:.6f} (change: {loss2-loss0:.4e})")

# ============================================================================
# TEST 3: SVRG variance
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: SVRG variance tracking")
print("=" * 70)

from optimizers.svrg import svrg_outer_loop

m = 2 * n
w_svrg = w_init.copy()
w_result, var_tracked = svrg_outer_loop(
    w_svrg, X_train, y_train,
    lr=0.025, lam=LAM, m=m,
    multiclass=True, option='I', track_variance=True,
)

# True gradient variance
mu = full_grad_multiclass(w_result, X_train, y_train, LAM)
n_sample = 500
indices = np.random.choice(n, size=min(n_sample, n), replace=False)
var_sum = 0.0
for i in indices:
    g_i = stoch_grad_multiclass(w_result, X_train[i], y_train[i], LAM)
    diff = g_i - mu
    var_sum += np.sum(diff * diff)
var_true = var_sum / len(indices)

print(f"SVRG tracked variance: {var_tracked:.6e}")
print(f"True gradient variance: {var_true:.6e}")
print(f"Ratio: {var_tracked/var_true:.4f}")

# ============================================================================
# TEST 4: Check if SVRG update is correct
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: SVRG update correctness")
print("=" * 70)

# Check: does SVRG decrease the loss?
loss_before = loss_multiclass(w_svrg, X_train, y_train, LAM)
loss_after = loss_multiclass(w_result, X_train, y_train, LAM)
print(f"Loss before SVRG outer iter: {loss_before:.10f}")
print(f"Loss after SVRG outer iter:  {loss_after:.10f}")
print(f"Change: {loss_after - loss_before:.6e}")

print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)
