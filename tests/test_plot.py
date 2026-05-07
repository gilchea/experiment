"""Smoke test for plot_results.py"""
import sys
import numpy as np
from source.plot_results import compute_test_error, _safe_residuals

# Test _safe_residuals
res = [1.0, 0.0, -1.0, 1e-20]
safe = _safe_residuals(res)
print(f'_safe_residuals: {safe}')
assert safe[0] == 1.0
assert safe[1] == 1e-16  # clipped
assert safe[2] == 1e-16  # clipped
assert safe[3] == 1e-16  # clipped (below 1e-16)
print('  OK')

# Test compute_test_error (binary)
np.random.seed(42)
n, d = 100, 10
X_test = np.random.randn(n, d)
w = np.random.randn(d)
y_test = np.sign(X_test @ w + 0.1 * np.random.randn(n))
err = compute_test_error(w, X_test, y_test, multiclass=False)
print(f'Binary test error: {err:.2f}%')
assert 0 <= err <= 100, f'Error should be percentage: {err}'
print('  OK')

# Test compute_test_error (multi-class)
n, d, K = 100, 10, 5
X_test = np.random.randn(n, d)
W = np.random.randn(d, K)
logits = X_test @ W
y_test = np.argmax(logits, axis=1)
err = compute_test_error(W, X_test, y_test, multiclass=True)
print(f'Multi-class test error: {err:.2f}%')
assert 0 <= err <= 100, f'Error should be percentage: {err}'
print('  OK')

# Test imports
from source.plot_results import load_results, load_all_results, plot_figure1, plot_individual_figures
print('All plot_results imports OK')

print()
print('All plot tests passed!')
sys.stdout.flush()
