"""Quick test to check CSV structure"""
import numpy as np
import pandas as pd
from main_program_optimized import main_evaluation_optimized

# Create and save tiny dataset
np.random.seed(42)
X_train = np.random.randn(100, 50).astype(np.float32)
X_test = np.random.randn(20, 50).astype(np.float32)
X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

train_file = "../training_vectors_test_csv.npy"
test_file = "../testing_vectors_test_csv.npy"
np.save(train_file, X_train)
np.save(test_file, X_test)

print("Running quick evaluation (PCA only)...")
all_results, detailed_csv, summary_csv = main_evaluation_optimized(
    dataset_name="test_csv",
    train_file=train_file,
    test_file=test_file,
    target_dim=16,
    b_percentage=1.0,
    alpha=0.1,
    k_values=[1, 10],
    save_results=True,
    output_dir="Result/test_csv",
    skip_methods=['Isomap', 'KernelPCA', 'LLE', 'MPAD', 'MPAD_Optimized',
                 'UMAP', 'FastICA', 'NMF', 'FeatureAgglomeration',
                 'Autoencoder', 'VAE', 'RandomProjection']
)

print(f"\n{'='*70}")
print("CSV STRUCTURE CHECK")
print(f"{'='*70}")

# Read detailed CSV
df = pd.read_csv(detailed_csv)
print(f"\nDetailed CSV shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for recall columns
recall_cols = [col for col in df.columns if 'recall' in col.lower()]
print(f"\nRecall columns found: {len(recall_cols)}")
for col in recall_cols:
    print(f"  - {col}")

# Show sample data
print(f"\nSample data (first 10 rows):")
if len(recall_cols) > 0:
    display_cols = ['method', 'index_method', 'k'] + recall_cols[:3]
    print(df[display_cols].head(10).to_string())
else:
    print(df.head(10).to_string())

# Check for IndexFlat_kNN specifically
if 'index_method' in df.columns:
    indexflat_rows = df[df['index_method'] == 'IndexFlat_kNN']
    print(f"\nIndexFlat_kNN rows: {len(indexflat_rows)}")
    if len(indexflat_rows) > 0:
        print(indexflat_rows[['method', 'index_method', 'k', 'recall_at_k']].to_string())

# Cleanup
import os, shutil
os.remove(train_file)
os.remove(test_file)
if os.path.exists("Result/test_csv"):
    shutil.rmtree("Result/test_csv")
print(f"\n{'='*70}")
print("[OK] Test complete, files cleaned up")
print(f"{'='*70}")

