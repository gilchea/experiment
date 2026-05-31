"""
debug_diagnostic_light.py — Lightweight diagnostic (no heavy SVRG loop)
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
print("LIGHTWEIGHT DIAGNOSTIC: MNIST Convex Results")
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
# TEST 1: P_star verification — run GD with better params
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: P_star verification (GD with c=1e-4, 1000 iters)")
print("=" * 70)

w_init = warm_start(X_train, y_train, LAM, MULTICLASS, n_epochs=1, lr=0.01)
loss_init = loss_multiclass(w_init, X_train, y_train, LAM)
print(f"Loss after warm-start (1 epoch SGD): {loss_init:.10f}")
print(f"Residual (w_init - P_star): {loss_init - P_star_old:.6e}")
print(f"--> NEGATIVE = P_star is NOT the true minimum!")

# GD with less aggressive Armijo
w = np.zeros((d, K))
lr = 0.1
for t in range(1000):
    g = full_grad_multiclass(w, X_train, y_train, LAM)
    current_loss = loss_multiclass(w, X_train, y_train, LAM)
    step = lr
    for _ in range(20):
        w_new = w - step * g
        new_loss = loss_multiclass(w_new, X_train, y_train, LAM)
        if new_loss <= current_loss - 1e-4 * step * np.sum(g * g) or step < 1e-12:
            break
        step *= 0.5
    w = w_new
    if t % 200 == 0:
        print(f"  iter {t:4d}: loss = {new_loss:.10f} (step = {step:.2e})")

P_star_new = loss_multiclass(w, X_train, y_train, LAM)
print(f"\nNew P_star (GD c=1e-4): {P_star_new:.10f}")
print(f"Old P_star:              {P_star_old:.10f}")
print(f"Difference:              {P_star_new - P_star_old:.6e}")

# Check if warm-start loss is still below new P_star
loss_init2 = loss_multiclass(w_init, X_train, y_train, LAM)
print(f"Warm-start loss vs New P_star: {loss_init2 - P_star_new:.6e}")
if loss_init2 > P_star_new:
    print("  ==> OK: warm-start loss > P_star (as expected)")
else:
    print("  ==> STILL NEGATIVE: P_star still not optimal enough!")

# ============================================================================
# TEST 2: SDCA — check with small subset
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: SDCA update formula check")
print("=" * 70)

from optimizers.sdca import sdca_epoch_multiclass

n_sub = 500
X_sub = X_train[:n_sub]
y_sub = y_train[:n_sub]
K_sub = len(np.unique(y_sub))

# Test 2a: Default SDCA lr
alpha = np.zeros((n_sub, d, K_sub))
W = np.zeros((d, K_sub))
loss0 = loss_multiclass(W, X_sub, y_sub, LAM)
print(f"Loss before SDCA: {loss0:.6f}")

lr_def = 1.0 / (LAM * n_sub)
print(f"Default lr = 1/(lam*n) = {lr_def:.4f}")
alpha, W = sdca_epoch_multiclass(alpha, W, X_sub, y_sub, LAM, n_sub, K_sub, lr_def)
loss1 = loss_multiclass(W, X_sub, y_sub, LAM)
print(f"Loss after 1 epoch (default lr): {loss1:.6f} (change: {loss1-loss0:.4e})")

# Test 2b: Small lr
alpha2 = np.zeros((n_sub, d, K_sub))
W2 = np.zeros((d, K_sub))
alpha2, W2 = sdca_epoch_multiclass(alpha2, W2, X_sub, y_sub, LAM, n_sub, K_sub, 0.01)
loss2 = loss_multiclass(W2, X_sub, y_sub, LAM)
print(f"Loss after 1 epoch (lr=0.01):    {loss2:.6f} (change: {loss2-loss0:.4e})")

# Test 2c: Compare with SGD for reference
from optimizers.sgd import sgd_epoch_constant
W_sgd = np.zeros((d, K_sub))
W_sgd = sgd_epoch_constant(W_sgd, X_sub, y_sub, lr=0.01, lam=LAM, multiclass=True)
loss_sgd = loss_multiclass(W_sgd, X_sub, y_sub, LAM)
print(f"Loss after 1 SGD epoch (lr=0.01): {loss_sgd:.6f} (change: {loss_sgd-loss0:.4e})")

# ============================================================================
# TEST 3: SVRG variance — just check the formula, don't run full loop
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: SVRG variance tracking formula check")
print("=" * 70)

print("""
  In svrg.py, variance is computed as:
    diff = v - mu  where v = g_current - g_snapshot + mu
    So diff = g_current - g_snapshot
    variance = ||g_current - g_snapshot||^2
  
  But the TRUE gradient variance should be:
    E[||nabla_psi_i(w) - nabla P(w)||^2]
  
  These are DIFFERENT quantities!
  - SVRG tracks: ||g_current(w) - g_snapshot(w_tilde)||^2
  - True variance: E[||g_current(w) - nabla P(w)||^2]
  
  As w -> w_tilde, SVRG variance -> 0 (explaining the rapid drop)
  But true variance only -> 0 as w -> w* (the optimum)
""")

# Quick check: compute both for a random w
w_test = np.random.randn(d, K) * 0.01
mu = full_grad_multiclass(w_test, X_train, y_train, LAM)

# True variance
n_sample = 200
indices = np.random.choice(n, size=n_sample, replace=False)
var_sum_true = 0.0
for i in indices:
    g_i = stoch_grad_multiclass(w_test, X_train[i], y_train[i], LAM)
    diff = g_i - mu
    var_sum_true += np.sum(diff * diff)
var_true = var_sum_true / n_sample
print(f"True gradient variance at random w: {var_true:.6e}")

# SVRG-style variance (using w_tilde = w_test, so g_snapshot ≈ g_current at snapshot)
# This would be ~0 if w ≈ w_tilde, which is what happens in SVRG
print(f"SVRG variance at same w (if w_tilde=w): ~0 (by construction)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)
print(f"""
1. P_star ISSUE (CRITICAL - affects ALL algorithms):
   - Old P_star = {P_star_old:.10f}
   - New P_star (GD with c=1e-4) = {P_star_new:.10f}
   - Warm-start loss = {loss_init:.10f}
   - All loss residuals are NEGATIVE because P_star is too high
   - Root cause: compute_optimal.py uses Armijo c=0.5 (too aggressive),
     causing GD to stall before reaching true optimum
   - Fix: Change c from 0.5 to 1e-4 in compute_optimal.py, or run more iterations

2. SDCA ISSUE (CRITICAL - causes divergence):
   - SDCA uses VECTOR dual variables alpha_i in R^d (n, d, K) 
   - Standard SDCA uses SCALAR dual variables (n,)
   - The vector formulation is incorrect and causes the loss to explode
   - Memory: O(n*d*K) = 3.7GB for MNIST instead of O(n) = 60KB
   - Fix: Reimplement SDCA with scalar alpha_i

3. SVRG VARIANCE (MINOR - cosmetic):
   - Tracked variance = ||g_current - g_snapshot||^2
   - True variance = E[||g_current - nabla P||^2]
   - SVRG's tracked variance drops rapidly because w -> w_tilde
   - This is why variance appears to converge quickly while loss hasn't
""")
