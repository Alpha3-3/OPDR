import numpy as np
import os
import scanpy as sc
from tqdm import tqdm
import random

# Absolute base directory for all datasets (Windows path)
BASE_DATA_DIR = r'D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD\Dataset processing and testing'

def l2_normalize(vectors):
    """L2 normalize vectors"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms

# -------------------- ANN dataset loaders (SIFT1M / Deep10M) --------------------

def _read_fvecs(file_path):
    """Read .fvecs file (int32 dim header + float32 vector)."""
    import numpy as np
    with open(file_path, 'rb') as f:
        data = f.read()
    # each vector: int32 (dim) + dim * float32
    # robust parse
    arr = np.frombuffer(data, dtype=np.int32)
    dim = arr[0]
    # reshape as (num*(1+dim)) int32 view, then skip headers
    num = len(arr) // (1 + dim)
    if num * (1 + dim) != len(arr):
        raise ValueError(f"Invalid fvecs file: {file_path}")
    arr = arr.reshape(num, 1 + dim)
    vecs = arr[:, 1:].view(np.float32)
    return np.ascontiguousarray(vecs)

def _read_bvecs(file_path):
    """Read .bvecs file (int32 dim header + uint8 vector), cast to float32."""
    import numpy as np
    with open(file_path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.int32)
    dim = arr[0]
    # reinterpret as bytes and parse records
    # Each record: int32 + dim*uint8
    rec_size = 4 + dim
    if len(data) % rec_size != 0:
        raise ValueError(f"Invalid bvecs file: {file_path}")
    num = len(data) // rec_size
    vecs = np.empty((num, dim), dtype=np.float32)
    offset = 0
    for i in range(num):
        # skip int32 header
        offset += 4
        vec = np.frombuffer(data, dtype=np.uint8, count=dim, offset=offset).astype(np.float32)
        vecs[i] = vec
        offset += dim
    return vecs

def _read_ivecs(file_path):
    """Read .ivecs file (int32 dim header + int32 vector), cast to float32."""
    import numpy as np
    with open(file_path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.int32)
    dim = arr[0]
    num = len(arr) // (1 + dim)
    if num * (1 + dim) != len(arr):
        raise ValueError(f"Invalid ivecs file: {file_path}")
    arr = arr.reshape(num, 1 + dim)
    vecs = arr[:, 1:].astype(np.float32)
    return np.ascontiguousarray(vecs)

def _read_fbin(file_path):
    """Read .fbin file (header: int32 dim, int32 nb, followed by nb*dim float32).
    Commonly used by Deep10M/Deep1B subsets.
    """
    import numpy as np
    with open(file_path, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"Invalid fbin header in {file_path}")
        h0, h1 = int(header[0]), int(header[1])
        # Many FBIN files are stored as (nb, dim); detect and normalize
        if h0 > 100000 and h1 <= 4096:  # likely nb, dim
            nb, dim = h0, h1
        else:
            dim, nb = h0, h1
        expected = dim * nb
        vecs = np.fromfile(f, dtype=np.float32, count=expected)
        if vecs.size != expected:
            raise ValueError(f"File {file_path} truncated: expected {expected} floats, got {vecs.size}")
        vecs = vecs.reshape(nb, dim)
        return np.ascontiguousarray(vecs)

def _load_ann_split(dataset_dir):
    """Load base/query from a dataset directory. Supports fvecs/bvecs/ivecs/npy."""
    import os
    names = os.listdir(dataset_dir)
    lower_map = {name.lower(): name for name in names}
    # Prefer known filenames first
    preferred = [
        ('base.10m.fbin', 'query.public.10k.fbin'),  # Deep10M
        ('sift_base.fvecs', 'sift_query.fvecs'),     # SIFT1M
    ]
    base_path = query_path = None
    for bname, qname in preferred:
        if bname in lower_map and qname in lower_map:
            base_path = os.path.join(dataset_dir, lower_map[bname])
            query_path = os.path.join(dataset_dir, lower_map[qname])
            break
    if base_path is None:
        # Fallback: pick first matching by prefix
        for name in names:
            lower = name.lower()
            if lower.startswith('base') or lower.startswith('learn'):
                base_path = os.path.join(dataset_dir, name)
                break
        for name in names:
            lower = name.lower()
            if lower.startswith('query') or lower.startswith('queries'):
                query_path = os.path.join(dataset_dir, name)
                break
    if base_path is None or query_path is None:
        raise FileNotFoundError(f"Could not find base/query in {dataset_dir}")

    def load_one(path):
        lower = path.lower()
        if lower.endswith('.fvecs'):
            return _read_fvecs(path)
        if lower.endswith('.bvecs'):
            return _read_bvecs(path)
        if lower.endswith('.ivecs'):
            return _read_ivecs(path)
        if lower.endswith('.fbin'):
            return _read_fbin(path)
        if lower.endswith('.npy'):
            return np.load(path)
        raise ValueError(f"Unsupported file type: {path}")

    X_train = load_one(base_path)
    X_test = load_one(query_path)
    return X_train, X_test

def process_sift1m(output_dir="."):
    """Process SIFT1M: base->train, query->test, L2-normalize, save .npy"""
    dataset_dir = os.path.join(BASE_DATA_DIR, 'SIFT1M')
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"SIFT1M directory not found: {dataset_dir}")
    X_train, X_test = _load_ann_split(dataset_dir)
    X_train = l2_normalize(X_train.astype(np.float32))
    X_test = l2_normalize(X_test.astype(np.float32))
    np.save(os.path.join(output_dir, 'training_vectors_SIFT1M.npy'), X_train)
    np.save(os.path.join(output_dir, 'testing_vectors_SIFT1M.npy'), X_test)
    print(f"Saved SIFT1M: train {X_train.shape}, test {X_test.shape}")

def process_deep10m(output_dir="."):
    """Process Deep10M: base->train, query->test, L2-normalize, save .npy"""
    dataset_dir = os.path.join(BASE_DATA_DIR, 'Deep10M')
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Deep10M directory not found: {dataset_dir}")
    X_train, X_test = _load_ann_split(dataset_dir)
    X_train = l2_normalize(X_train.astype(np.float32))
    X_test = l2_normalize(X_test.astype(np.float32))
    np.save(os.path.join(output_dir, 'training_vectors_Deep10M.npy'), X_train)
    np.save(os.path.join(output_dir, 'testing_vectors_Deep10M.npy'), X_test)
    print(f"Saved Deep10M: train {X_train.shape}, test {X_test.shape}")

def load_fasttext_data(vec_file_path, subsample_ratio=0.01):
    """Load Fasttext data with subsampling"""
    print(f"Loading Fasttext data from {vec_file_path} with subsample ratio {subsample_ratio}")
    
    all_vectors_list = []
    
    try:
        with open(vec_file_path, 'r', encoding='utf-8') as file:
            header = next(file)
            num_total_vectors_in_file, dim = map(int, header.split())
            print(f"File reports {num_total_vectors_in_file} vectors of dimension {dim}")
            
            # Calculate how many vectors to read
            num_to_read = int(num_total_vectors_in_file * subsample_ratio)
            print(f"Reading {num_to_read} vectors ({subsample_ratio*100:.1f}% of total)")
            
            for i, line in enumerate(tqdm(file, desc="Reading vectors", total=num_to_read)):
                if i >= num_to_read:
                    break
                    
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                    
                try:
                    vector_components = np.array(parts[1:], dtype=float)
                    all_vectors_list.append(vector_components)
                except ValueError:
                    continue
                    
    except Exception as e:
        print(f"Error reading Fasttext file: {e}")
        raise
    
    if not all_vectors_list:
        raise ValueError("No valid vectors were read from the file")
    
    all_vectors = np.array(all_vectors_list)
    print(f"Successfully read {all_vectors.shape[0]} vectors of dimension {all_vectors.shape[1]}")
    
    return all_vectors

def load_isolet_data(data_file_path):
    """Load Isolet data"""
    print(f"Loading Isolet data from {data_file_path}")
    
    all_vectors_list = []
    
    try:
        with open(data_file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc="Processing lines"):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(',')
                try:
                    all_vectors_list.append(np.array(parts, dtype=float))
                except ValueError:
                    continue
                    
    except Exception as e:
        print(f"Error reading Isolet file: {e}")
        raise
    
    if not all_vectors_list:
        raise ValueError("No valid vectors were read from the file")
    
    all_vectors = np.array(all_vectors_list)
    print(f"Successfully read {all_vectors.shape[0]} vectors of dimension {all_vectors.shape[1]}")
    
    return all_vectors

def load_isolet_data_multiple(data_file_paths):
    """Load Isolet data from multiple files"""
    print(f"Loading Isolet data from {len(data_file_paths)} files")
    
    all_vectors_list = []
    
    for data_file_path in data_file_paths:
        if not os.path.exists(data_file_path):
            print(f"File {data_file_path} not found, skipping...")
            continue
            
        print(f"Reading from: {data_file_path}")
        try:
            with open(data_file_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file, desc=f"Processing {os.path.basename(data_file_path)}"):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split(',')
                    try:
                        all_vectors_list.append(np.array(parts, dtype=float))
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"Error reading {data_file_path}: {e}")
            continue
    
    if not all_vectors_list:
        raise ValueError("No data loaded from Isolet files")
    
    all_vectors = np.array(all_vectors_list)
    print(f"Successfully loaded {all_vectors.shape[0]} vectors of dimension {all_vectors.shape[1]} from {len(data_file_paths)} files")
    
    return all_vectors

def load_arcene_data(file_paths):
    """Load Arcene data from multiple files"""
    print(f"Loading Arcene data from {len(file_paths)} files")
    
    all_vectors_list = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File {file_path} not found, skipping...")
            continue
            
        print(f"Reading from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file, desc=f"Reading {os.path.basename(file_path)}"):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    try:
                        all_vectors_list.append(np.array(parts, dtype=float))
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not all_vectors_list:
        raise ValueError("No data loaded from Arcene files")
    
    all_vectors = np.array(all_vectors_list)
    print(f"Successfully loaded {all_vectors.shape[0]} vectors of dimension {all_vectors.shape[1]}")
    
    return all_vectors

def load_pbmc3k_data():
    """Load PBMC3k data from local file"""
    print("Loading PBMC3k data from local file")

    data_file_path = os.path.join(BASE_DATA_DIR, 'PBMC3k', 'data', 'pbmc3k_processed.h5ad')

    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"PBMC3k file not found at {data_file_path}")

    adata = sc.read_h5ad(data_file_path)

    if not isinstance(adata.X, np.ndarray):
        X = adata.X.toarray()
    else:
        X = adata.X

    # Filter rows with NaN values
    valid_mask = np.all(~np.isnan(X), axis=1)
    X_valid = X[valid_mask]

    print(f"Successfully loaded {X_valid.shape[0]} vectors of dimension {X_valid.shape[1]} from local file")
    return X_valid

def split_data_80_20(vectors, seed=1):
    """Split data into 80% training and 20% testing"""
    np.random.seed(seed)
    
    n_samples = vectors.shape[0]
    n_train = int(0.8 * n_samples)
    
    # Random permutation
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_data = vectors[train_indices]
    test_data = vectors[test_indices]
    
    print(f"Split: {train_data.shape[0]} training, {test_data.shape[0]} testing")
    return train_data, test_data

def split_fasttext_subsample(vectors, subsample_ratio, seed=1):
    """Split Fasttext subsample into train/test"""
    np.random.seed(seed)
    
    n_samples = vectors.shape[0]
    n_train = int(0.8 * n_samples)
    
    # Random permutation
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_data = vectors[train_indices]
    test_data = vectors[test_indices]
    
    print(f"Fasttext subsample ({subsample_ratio*100:.1f}%): {train_data.shape[0]} training, {test_data.shape[0]} testing")
    return train_data, test_data

def process_dataset(dataset_name, output_dir="."):
    """Process a single dataset"""
    print(f"\n=== Processing {dataset_name} ===")
    
    # Set random seed for reproducibility
    random.seed(1)
    np.random.seed(1)
    
    if dataset_name == "Fasttext":
        # Define paths - adjust these to your actual file locations
        vec_file_path = os.path.join(BASE_DATA_DIR, 'Fasttext', 'data', 'wiki-news-300d-1M.vec')
        if not os.path.exists(vec_file_path):
            raise FileNotFoundError(f"Fasttext file not found at {vec_file_path}")
        
        # Process different subsample ratios
        for ratio in [0.01, 0.05, 0.10, 1.00]:
            print(f"\n--- Processing Fasttext subsample {ratio*100:.1f}% ---")
            
            vectors = load_fasttext_data(vec_file_path, ratio)
            
            # L2 normalize
            vectors = l2_normalize(vectors)
            
            # Split into train/test
            train_data, test_data = split_fasttext_subsample(vectors, ratio)
            
            # Save files
            if abs(ratio - 1.0) < 1e-9:
                ratio_str = "100pct"
            else:
                ratio_str = f"{int(ratio*100):02d}pct"
            train_file = os.path.join(output_dir, f"training_vectors_{ratio_str}_Fasttext.npy")
            test_file = os.path.join(output_dir, f"testing_vectors_{ratio_str}_Fasttext.npy")
            
            np.save(train_file, train_data)
            np.save(test_file, test_data)
            
            print(f"Saved: {train_file}, {test_file}")
    
    elif dataset_name == "Isolet":
        # Load all Isolet data files
        data_file_paths = [
            os.path.join(BASE_DATA_DIR, 'Isolet', 'data', 'isolet1+2+3+4.data'),
            os.path.join(BASE_DATA_DIR, 'Isolet', 'data', 'isolet5.data')
        ]
        if not any(os.path.exists(p) for p in data_file_paths):
            raise FileNotFoundError("Isolet data files not found under base data dir")
        vectors = load_isolet_data_multiple(data_file_paths)
        
        # L2 normalize
        vectors = l2_normalize(vectors)
        
        # Split into train/test
        train_data, test_data = split_data_80_20(vectors)
        
        # Save files
        train_file = os.path.join(output_dir, "training_vectors_Isolet.npy")
        test_file = os.path.join(output_dir, "testing_vectors_Isolet.npy")
        
        np.save(train_file, train_data)
        np.save(test_file, test_data)
        
        print(f"Saved: {train_file}, {test_file}")
    
    elif dataset_name == "PBMC3k":
        vectors = load_pbmc3k_data()
        
        # L2 normalize
        vectors = l2_normalize(vectors)
        
        # Split into train/test
        train_data, test_data = split_data_80_20(vectors)
        
        # Save files
        train_file = os.path.join(output_dir, "training_vectors_PBMC3k.npy")
        test_file = os.path.join(output_dir, "testing_vectors_PBMC3k.npy")
        
        np.save(train_file, train_data)
        np.save(test_file, test_data)
        
        print(f"Saved: {train_file}, {test_file}")
    
    elif dataset_name == "Arcene":
        file_paths = [
            os.path.join(BASE_DATA_DIR, 'Arcene', 'data', 'arcene_train.data'),
            os.path.join(BASE_DATA_DIR, 'Arcene', 'data', 'arcene_valid.data'),
            os.path.join(BASE_DATA_DIR, 'Arcene', 'data', 'arcene_test.data')
        ]
        if not any(os.path.exists(p) for p in file_paths):
            raise FileNotFoundError("Arcene data files not found under base data dir")
        vectors = load_arcene_data(file_paths)
        
        # L2 normalize
        vectors = l2_normalize(vectors)
        
        # Split into train/test
        train_data, test_data = split_data_80_20(vectors)
        
        # Save files
        train_file = os.path.join(output_dir, "training_vectors_Arcene.npy")
        test_file = os.path.join(output_dir, "testing_vectors_Arcene.npy")
        
        np.save(train_file, train_data)
        np.save(test_file, test_data)
        
        print(f"Saved: {train_file}, {test_file}")

if __name__ == "__main__":
    # Process selected datasets
    datasets = ["Fasttext", "Isolet", "PBMC3k", "Arcene"]
    for dataset in datasets:
        try:
            process_dataset(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue
    # ANN datasets
    try:
        process_sift1m()
    except Exception as e:
        print(f"Error processing SIFT1M: {e}")
    try:
        process_deep10m()
    except Exception as e:
        print(f"Error processing Deep10M: {e}")
    print("\n=== Data preprocessing complete ===")
