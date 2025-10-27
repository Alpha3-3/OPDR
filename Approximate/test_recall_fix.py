"""
Quick test to verify recall data extraction fix
Tests that all recall columns are properly populated
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

def test_recall_extraction():
    """Test recall extraction with a tiny synthetic dataset"""
    print("=" * 70)
    print("Testing Recall Data Extraction Fix")
    print("=" * 70)
    print()
    
    # Import after adding to path
    from main_program_optimized import main_evaluation_optimized
    
    # Create tiny synthetic dataset and save to temp files
    np.random.seed(42)
    n_samples = 500
    n_features = 100
    
    print(f"[TEST] Creating synthetic dataset: {n_samples} samples, {n_features} features")
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    X_test = np.random.randn(100, n_features).astype(np.float32)
    
    # L2 normalize
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    
    # Save to temp files
    train_file = "../training_vectors_test_recall_fix.npy"
    test_file = "../testing_vectors_test_recall_fix.npy"
    np.save(train_file, X_train)
    np.save(test_file, X_test)
    print(f"[TEST] Saved to {train_file} and {test_file}")
    
    print("[TEST] Running evaluation with PCA and RandomProjection...")
    try:
        all_results, detailed_csv, summary_csv = main_evaluation_optimized(
            dataset_name="test_recall_fix",
            train_file=train_file,
            test_file=test_file,
            target_dim=32,
            b_percentage=1.0,
            alpha=0.1,
            k_values=[1, 10],
            save_results=True,
            output_dir="Result/test_recall_fix",
            skip_methods=['Isomap', 'KernelPCA', 'LLE', 'MPAD_Optimized', 
                         'UMAP', 'FastICA', 'NMF', 'FeatureAgglomeration',
                         'Autoencoder', 'VAE']  # Only test PCA and RandomProjection
        )
        
        print(f"\n[TEST] Results saved to: {detailed_csv}")
        
        # Read and check results
        df = pd.read_csv(detailed_csv)
        print("\n" + "=" * 70)
        print("VERIFICATION RESULTS")
        print("=" * 70)
        
        # Check recall columns
        recall_cols = [col for col in df.columns if 'recall' in col.lower()]
        print(f"\n[CHECK] Found {len(recall_cols)} recall columns:")
        
        all_ok = True
        for col in recall_cols:
            non_null = df[col].notna().sum()
            total = len(df)
            min_val = df[col].min() if non_null > 0 else np.nan
            max_val = df[col].max() if non_null > 0 else np.nan
            
            status = "[OK]" if non_null == total else "[FAIL]"
            if non_null != total:
                all_ok = False
            
            print(f"  {status} {col:30s} : {non_null:2d}/{total:2d} non-null", end="")
            if non_null > 0:
                print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
            else:
                print("  NO DATA")
        
        # Check for IndexFlat_kNN specifically
        indexflat_cols = [col for col in recall_cols if 'indexflat' in col.lower()]
        print(f"\n[CHECK] IndexFlat_kNN columns: {len(indexflat_cols)}")
        for col in indexflat_cols:
            non_null = df[col].notna().sum()
            if non_null > 0:
                mean_val = df[col].mean()
                # IndexFlat should NOT be 1.0 (since ground truth is from original space)
                if abs(mean_val - 1.0) < 0.01:
                    print(f"  [WARN] {col} mean = {mean_val:.3f} (should be < 1.0)")
                    all_ok = False
                else:
                    print(f"  [OK] {col} mean = {mean_val:.3f} (correct, < 1.0)")
            else:
                print(f"  [FAIL] {col} has no data")
                all_ok = False
        
        # Check for type errors (should not crash)
        print("\n[CHECK] Type error check:")
        print("  [OK] No 'TypeError: argument of type float is not iterable'")
        
        print("\n" + "=" * 70)
        if all_ok:
            print("[SUCCESS] All checks passed!")
            print("=" * 70)
            
            # Cleanup temp files
            print("\n[CLEANUP] Removing temporary files...")
            os.remove(train_file)
            os.remove(test_file)
            import shutil
            if os.path.exists("Result/test_recall_fix"):
                shutil.rmtree("Result/test_recall_fix")
            print("[OK] Cleanup complete")
            
            return 0
        else:
            print("[FAILURE] Some checks failed")
            print("=" * 70)
            return 1
            
    except TypeError as e:
        if "argument of type 'float' is not iterable" in str(e):
            print(f"\n[FAILURE] Type error still present: {e}")
            print("=" * 70)
            return 1
        else:
            raise
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = test_recall_extraction()
    sys.exit(exit_code)
