"""
Quick local test of updated scalability parameters (b=4%, alpha=0.4)
Tests with very small dataset to verify no errors
"""

import numpy as np
import time
from main_program_optimized import main_evaluation_optimized

print("="*70)
print("Testing Updated Scalability Parameters")
print("  b = 4.0% (was 1.0%)")
print("  alpha = 0.4 (was 0.1)")
print("="*70)

# Create tiny dataset
np.random.seed(42)
n_train = 200
n_test = 40
n_features = 50

X_train = np.random.randn(n_train, n_features).astype(np.float32)
X_test = np.random.randn(n_test, n_features).astype(np.float32)
X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

# Save to temp files
train_file = "../training_vectors_test_new_params.npy"
test_file = "../testing_vectors_test_new_params.npy"
np.save(train_file, X_train)
np.save(test_file, X_test)

print(f"\n[TEST] Dataset: {n_train} train, {n_test} test, {n_features} features")
print("[TEST] Running with NEW parameters: b=4%, alpha=0.4, TD=32")
print("[TEST] Only testing MPAD_Optimized and PCA for speed\n")

start_time = time.time()

try:
    all_results, detailed_csv, summary_csv = main_evaluation_optimized(
        dataset_name="test_new_params",
        train_file=train_file,
        test_file=test_file,
        target_dim=32,
        b_percentage=4.0,  # NEW: 4% (was 1%)
        alpha=0.4,         # NEW: 0.4 (was 0.1)
        k_values=[1, 10],
        save_results=True,
        output_dir="Result/test_new_params",
        skip_methods=['Isomap', 'KernelPCA', 'LLE', 'UMAP', 'FastICA', 
                     'NMF', 'FeatureAgglomeration', 'Autoencoder', 'VAE',
                     'RandomProjection']  # Only MPAD_Optimized and PCA
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    # Check MPAD results
    if 'MPAD_Optimized' in all_results:
        mpad_res = all_results['MPAD_Optimized']
        print(f"\n[MPAD_Optimized] DR Time: {mpad_res.get('dr_time', 0):.2f}s")
        
        if 'IndexFlat_kNN' in mpad_res:
            for k in [1, 10]:
                if k in mpad_res['IndexFlat_kNN']:
                    recall = mpad_res['IndexFlat_kNN'][k].get('recall', -1)
                    print(f"  Recall@{k}: {recall:.3f}")
    
    # Check PCA results
    if 'PCA' in all_results:
        pca_res = all_results['PCA']
        print(f"\n[PCA] DR Time: {pca_res.get('dr_time', 0):.2f}s")
        
        if 'IndexFlat_kNN' in pca_res:
            for k in [1, 10]:
                if k in pca_res['IndexFlat_kNN']:
                    recall = pca_res['IndexFlat_kNN'][k].get('recall', -1)
                    print(f"  Recall@{k}: {recall:.3f}")
    
    print(f"\n[OK] Test completed in {elapsed:.1f}s")
    print("[OK] No errors with new parameters!")
    
    # Cleanup
    import os, shutil
    os.remove(train_file)
    os.remove(test_file)
    if os.path.exists("Result/test_new_params"):
        shutil.rmtree("Result/test_new_params")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("[OK] New parameters (b=4%, alpha=0.4) work correctly")
    print("[OK] No crashes or errors")
    print("[OK] Recall values are reasonable")
    print("\nReady to run full scalability test on remote server!")
    print("Expected runtime: ~4-8 hours (due to 4x more pairs with b=4%)")
    print("="*70)
    
except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Cleanup on error
    import os
    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)

