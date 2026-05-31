# requirements.txt Update Plan

The current `requirements.txt` contains only:
```
numpy
scikit-learn
```

**Updated content** (needs Code mode to write):
```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

**Changes**:
1. Added version pins for stability
2. Added `scipy` (required for sparse matrix operations with RCV1/Covtype)
3. Added `matplotlib` (required for plot_results.py)
