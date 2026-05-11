import pickle

files = [
    r"d:\Khóa luận tốt nghiệp\experiment\data\scalers\cifar10_channel_stats.pkl",
    r"d:\Khóa luận tốt nghiệp\experiment\data\scalers\covtype_maxabs_scaler.pkl",
    r"d:\Khóa luận tốt nghiệp\experiment\data\scalers\mnist_standard_scaler.pkl",
    r"d:\Khóa luận tốt nghiệp\experiment\data\scalers\rcv1_maxabs_scaler.pkl"
]

import os

for f in files:
    print(f"--- {os.path.basename(f)} ---")
    try:
        with open(f, 'rb') as file:
            obj = pickle.load(file)
            print(f"Type: {type(obj)}")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    print(f"  {k}: type {type(v)}, val: {v}")
            else:
                if hasattr(obj, 'scale_'):
                    print(f"  scale_ shape: {obj.scale_.shape}")
                    print(f"  scale_ min/max: {obj.scale_.min()}, {obj.scale_.max()}")
                if hasattr(obj, 'mean_'):
                    print(f"  mean_ shape: {obj.mean_.shape}")
                if hasattr(obj, 'var_'):
                    print(f"  var_ shape: {obj.var_.shape}")
                if hasattr(obj, 'max_abs_'):
                    print(f"  max_abs_ shape: {obj.max_abs_.shape}")
                    print(f"  max_abs_ min/max: {obj.max_abs_.min()}, {obj.max_abs_.max()}")
    except Exception as e:
        print(f"Error: {e}")
