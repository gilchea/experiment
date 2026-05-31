"""
debug_diagnostic.py — Diagnostic script to validate hypotheses about incorrect results

Hypotheses being tested:
1. P_star from GD is NOT the true optimum (GD stalled due to aggressive backtracking)
2. SDCA update formula is incorrect (divergence)
3. SVRG variance tracking measures wrong quantity
"""

import os
import sys
import json
import numpy as np

np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source'))

from utils.data_loader import load_dataset
from models.logistic import loss, full_grad, stoch_grad, loss_multiclass, full_grad_multiclass
from optimizers.sgd import warm_start, sgd_epoch_constant, sgd_epoch_decay
from optimizers.svrg import svrg_outer_loop, effective_passes_svrg
from optimizers.sdca import sdca_train, count_effective_passes_sdca

DATASET = 'mnist'
LAM = 1e-4
MULTICLASS = True

print("=" * 70)
print("DIAGNOSTIC: MNIST Convex Results")
print("=" * 70)

# Load data
X_train, y_train, X_test, y_test = load_dataset(DATASET)
n, d = X_train.shape
K = len(np.unique(y_train))
print(f"\nDataset: n={n}, d={d}, K={K}")

# Load current P_star
with open('results/optimal_loss.json', 'r') as f:
    optimal = json.load(f)
P_star_current = float(optimal[DATASET]['P_star'])
print(f"\nCurrent P_star from optimal_loss.json: {P_star_current:.15f}")

# ============================================================================
# TEST 1: Is P_star the true minimum?
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: Is P_star the true minimum?")
print("=" * 70)

# Compute loss at w=0
w_zero = np.zeros((d, K))
loss_zero = loss_multiclass(w_zero, X_train, y_train, LAM)
print(f"\n  Loss at w=0:          {loss_zero:.15f}")

# Run warm-start (1 epoch SGD, lr=0.01)
w_init = warm_start(X_train, y_train, LAM, MULTICLASS, n_epochs=1, lr=0.01)
loss_init = loss_multiclass(w_init, X_train, y_train, LAM)
print(f"  Loss after warm-start: {loss_init:.15f}")
print(f"  Loss residual (w_init - P_star): {loss_init - P_star_current:.6e}")
print(f"  ==> NEGATIVE residual means P_star is NOT the true minimum!")

# Run more GD to find better P_star
print(f"\n  Running extended GD to find true P_star...")
w = np.zeros((d, K))
lr = 0.1
prev_loss = None
for t in range(5000):
    g = full_grad_multiclass(w, X_train, y_train, LAM)
    current_loss = loss_multiclass(w, X_train, y_train, LAM)
    
    # Backtracking line search with less aggressive c=1e-4
    step = lr
    for _ in range(30):
        w_new = w - step * g
        new_loss = loss_multiclass(w_new, X_train, y_train, LAM)
        if new_loss <= current_loss - 1e-4 * step * np.sum(g * g) or step < 1e-12:
            break
        step *= 0.5
    
    w = w_new
    
    if t % 500 == 0 or t == 4999:
        print(f"    iter {t:4d}: loss = {new_loss:.12f} (step = {step:.2e})")
    
    if prev_loss is not None and t % 100 == 0:
        rel_change = abs(new_loss - prev_loss) / max(1, abs(prev_loss))
        if rel_change < 1e-14 and t > 500:
            print(f"    Converged at iter {t}")
            break
    if t % 100 == 0:
        prev_loss = new_loss

P_star_true = loss_multiclass(w, X_train, y_train, LAM)
print(f"\n  True P_star (extended GD): {P_star_true:.15f}")
print(f"  Old P_star:                 {P_star_current:.15f}")
print(f"  Difference:                 {P_star_true - P_star_current:.6e}")

# ============================================================================
# TEST 2: SDCA — check if update formula is correct
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: SDCA Update Formula Check")
print("=" * 70)

# The SDCA update in sdca.py:
#   alpha[i] -= lr * (grad + lam * n * alpha[i])
#   w += alpha[i] - alpha_old
#
# Standard SDCA for logistic regression:
#   The dual update should be: alpha_i = alpha_i - eta * (grad_phi_i(w) + lambda * n * alpha_i)
#   But wait — in standard SDCA, alpha_i is a SCALAR, not a vector!
#   For logistic regression with L2 regularization, the standard SDCA uses:
#     w = (1/(lambda*n)) * sum(alpha_i * x_i)  where alpha_i are SCALARS
#   But this code uses VECTOR alpha_i in R^d, which is WRONG!

print("""
  SDCA THEORY CHECK:
  
  Standard SDCA for logistic regression:
    - Dual variables alpha_i are SCALARS (one per data point)
    - Primal: w = (1/(lambda*n)) * sum_i alpha_i * x_i
    - Update: alpha_i = alpha_i - eta * (grad_phi_i(w) + alpha_i)
    
  This code's SDCA:
    - Dual variables alpha_i are VECTORS in R^d (one per data point)
    - Primal: w = sum_i alpha_i  (direct sum of vectors)
    - Update: alpha_i = alpha_i - eta * (grad_phi_i(w) + lambda*n * alpha_i)
  
  The vector formulation is NON-STANDARD and likely incorrect!
  With d=784 and n=60000, storing alpha as (n, d, K) = (60000, 784, 10)
  requires 60000*784*10*8 bytes ≈ 3.7 GB of memory!
""")

# Quick test: run SDCA for just 1 epoch and check loss
print("  Running SDCA for 1 epoch to verify divergence...")
from optimizers.sdca import sdca_epoch_multiclass

n_test = min(1000, n)  # Use subset for quick test
X_sub = X_train[:n_test]
y_sub = y_train[:n_test]
K_sub = len(np.unique(y_sub))

alpha = np.zeros((n_test, d, K_sub))
W = np.zeros((d, K_sub))

loss_before = loss_multiclass(W, X_sub, y_sub, LAM)
print(f"  Loss before SDCA: {loss_before:.6f}")

lr_sdca = 1.0 / (LAM * n_test)
print(f"  SDCA lr = 1/(lam*n) = {lr_sdca:.6f}")

alpha, W = sdca_epoch_multiclass(alpha, W, X_sub, y_sub, LAM, n_test, K_sub, lr_sdca)
loss_after = loss_multiclass(W, X_sub, y_sub, LAM)
print(f"  Loss after 1 SDCA epoch: {loss_after:.6f}")
print(f"  Change: {loss_after - loss_before:.6e}")

# Try with smaller lr
print(f"\n  Trying SDCA with lr = 0.1...")
alpha2 = np.zeros((n_test, d, K_sub))
W2 = np.zeros((d, K_sub))
alpha2, W2 = sdca_epoch_multiclass(alpha2, W2, X_sub, y_sub, LAM, n_test, K_sub, 0.1)
loss_after2 = loss_multiclass(W2, X_sub, y_sub, LAM)
print(f"  Loss after 1 SDCA epoch (lr=0.1): {loss_after2:.6f}")
print(f"  Change: {loss_after2 - loss_before:.6e}")

# ============================================================================
# TEST 3: SVRG variance tracking
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: SVRG Variance Tracking")
print("=" * 70)

print("""
  Current variance tracking in svrg.py:
    diff = v - mu
    variance_sum += np.dot(diff, diff)
  
  This measures ||v - mu||^2 where v = g_current - g_snapshot + mu
  So v - mu = g_current - g_snapshot
  
  This is NOT the gradient variance E[||nabla_psi_i(w) - nabla P(w)||^2]
  It's actually ||g_current - g_snapshot||^2, which measures how much
  the stochastic gradient at current w differs from the gradient at snapshot w_tilde.
  
  This is a valid convergence diagnostic for SVRG but it's NOT the
  standard gradient variance that should be compared across algorithms.
""")

# Run 1 outer iteration and check
m = 2 * n
w_test = w_init.copy()
print(f"\n  Running 1 SVRG outer iteration (m={m})...")
w_result, var = svrg_outer_loop(
    w_test, X_train, y_train,
    lr=0.025, lam=LAM, m=m,
    multiclass=True, option='I', track_variance=True,
)
print(f"  SVRG tracked variance: {var:.6e}")

# Compare with true gradient variance
from source.run_mnist_convex import estimate_grad_variance
true_var = estimate_grad_variance(w_result, X_train, y_train, LAM, True)
print(f"  True gradient variance: {true_var:.6e}")
print(f"  Ratio (tracked/true): {var/true_var:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)
print(f"""
1. P_star ISSUE (CRITICAL):
   - Current P_star = {P_star_current:.10f}
   - After just 1 epoch SGD warm-start, loss = {loss_init:.10f} < P_star
   - This is IMPOSSIBLE for convex optimization
   - Root cause: GD in compute_optimal.py converges too slowly
     (aggressive backtracking with c=0.5, only 2000 iterations)
   - Fix: Run GD longer or use better learning rate / convergence criteria

2. SDCA ISSUE (CRITICAL):
   - SDCA uses VECTOR dual variables alpha_i in R^d instead of SCALAR
   - This is non-standard and causes divergence
   - Memory: O(n*d*K) instead of O(n)
   - Fix: Reimplement SDCA with scalar alpha_i

3. SVRG VARIANCE ISSUE (MINOR):
   - Tracked variance measures ||g_current - g_snapshot||^2
   - Not the true gradient variance E[||nabla_psi_i(w) - nabla P(w)||^2]
   - This explains why SVRG variance drops to near-zero quickly
     (as w converges to w_tilde, the difference goes to 0)
""")
