#!/usr/bin/env python3
"""
Quick check for remote server data files
Run this on the remote server to verify data integrity
"""

import os
import sys
import numpy as np

print("="*70)
print("DATA FILE VERIFICATION")
print("="*70)

# Check if file exists
data_file = "training_vectors_01pct_Fasttext.npy"

if not os.path.exists(data_file):
    print(f"\n[ERROR] File not found: {data_file}")
    print("\nPlease upload the file from local machine:")
    print("  scp training_vectors_01pct_Fasttext.npy jiuzhou@er074.utah.cloudlab.us:~/Approximate/")
    sys.exit(1)

# Check file size
file_size = os.path.getsize(data_file)
print(f"\n[INFO] File found: {data_file}")
print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

# Load and check shape
try:
    X = np.load(data_file)
    print(f"  Data shape: {X.shape}")
    print(f"  Data type: {X.dtype}")
    print(f"  Memory usage: {X.nbytes/1024/1024:.2f} MB")
    
    # Check if data looks reasonable
    N, n = X.shape
    
    if N < 100:
        print(f"\n[WARNING] Only {N} samples - this seems too small!")
        print("  Expected: ~7000-8000 samples for 1% Fasttext")
        print("  Please verify the data file on local machine first:")
        print("    python -c \"import numpy as np; print(np.load('training_vectors_01pct_Fasttext.npy').shape)\"")
    elif N < 1000:
        print(f"\n[WARNING] Only {N} samples - might be insufficient")
        print("  Expected: ~7000-8000 samples for 1% Fasttext")
    else:
        print(f"\n[OK] Data looks good!")
        print(f"  {N} samples × {n} features")
        print(f"  Mean: {X.mean():.6f}, Std: {X.std():.6f}")
        print(f"  Min: {X.min():.6f}, Max: {X.max():.6f}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(X)):
            print("\n[WARNING] Data contains NaN values")
        if np.any(np.isinf(X)):
            print("\n[WARNING] Data contains Inf values")
        
        print("\n[READY] You can now run: python3 test_optimized_mpad.py")
        
except Exception as e:
    print(f"\n[ERROR] Failed to load data: {e}")
    sys.exit(1)

print("="*70)

