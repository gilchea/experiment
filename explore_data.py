import os
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file

def explore_libsvm(file_path):
    print(f"\n--- Exploring LIBSVM data: {os.path.basename(file_path)} ---")
    try:
        # Load sparse data
        X, y = load_svmlight_file(file_path)
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Labels: {np.unique(y)}")
        print(f"Feature matrix type: {type(X)}")
        # Show a snippet of the first sample
        print("First sample (non-zero entries):")
        nonzero_indices = X[0].nonzero()[1]
        for idx in nonzero_indices[:10]:
            print(f"  Feature {idx}: {X[0, idx]}")
    except Exception as e:
        print(f"Error loading LIBSVM: {e}")

def explore_cifar10(dir_path):
    print(f"\n--- Exploring CIFAR-10 data: {os.path.basename(dir_path)} ---")
    batch_file = os.path.join(dir_path, 'cifar-10-batches-py', 'data_batch_1')
    if not os.path.exists(batch_file):
        print(f"CIFAR-10 batch file not found at {batch_file}")
        return
    
    try:
        with open(batch_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        
        X = data_dict[b'data']
        y = data_dict[b'labels']
        print(f"Shape of data batch: {X.shape}")
        print(f"Number of labels: {len(y)}")
        print(f"Image resolution: 32x32x3 (flattened to 3072)")
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")

def explore_mnist_binary(file_path):
    print(f"\n--- Exploring MNIST Binary data: {os.path.basename(file_path)} ---")
    try:
        with open(file_path, 'rb') as f:
            # Read magic number and metadata
            magic = int.from_bytes(f.read(4), 'big')
            num_items = int.from_bytes(f.read(4), 'big')
            print(f"Magic number: {magic}")
            print(f"Number of items: {num_items}")
            
            if magic == 2051: # Images
                rows = int.from_bytes(f.read(4), 'big')
                cols = int.from_bytes(f.read(4), 'big')
                print(f"Dimensions: {rows}x{cols}")
                # Read first image
                data = np.frombuffer(f.read(rows * cols), dtype=np.uint8)
                print(f"First image data (first 10 pixels): {data[:10]}")
    except Exception as e:
        print(f"Error loading MNIST binary: {e}")

if __name__ == "__main__":
    base_data_path = r"d:\Khóa luận tốt nghiệp\experiment\data"
    
    # 1. RCV1 (LIBSVM)
    rcv1_path = os.path.join(base_data_path, "rcv1", "rcv1_train.binary", "rcv1_train.binary")
    if os.path.exists(rcv1_path):
        explore_libsvm(rcv1_path)
        
    # 2. Covtype (LIBSVM)
    covtype_path = os.path.join(base_data_path, "covtype", "covtype.libsvm.binary", "covtype.libsvm.binary")
    if os.path.exists(covtype_path):
        explore_libsvm(covtype_path)
        
    # 3. CIFAR-10
    cifar_path = os.path.join(base_data_path, "cifar-10-python")
    if os.path.exists(cifar_path):
        explore_cifar10(cifar_path)
        
    # 4. MNIST Binary
    mnist_path = os.path.join(base_data_path, "mnist", "train-images-idx3-ubyte", "train-images.idx3-ubyte")
    if os.path.exists(mnist_path):
        explore_mnist_binary(mnist_path)
