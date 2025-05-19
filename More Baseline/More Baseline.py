import numpy as np
import pandas as pd
import time
import itertools
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import FastICA, NMF
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import LinearRegression
import os

# For autoencoder and VAE
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# -------------------------
# k-NN accuracy calculation (same as in the base code)
# -------------------------
def calculate_accuracy(original_data, reduced_data, new_original_data, new_reduced_data, k):
    total_start = time.perf_counter()
    nbrs_original = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(reduced_data)
    total_matches = 0
    for i in range(len(new_original_data)):
        inds_orig = nbrs_original.kneighbors(new_original_data[i].reshape(1, -1),
                                             return_distance=False)[0]
        inds_reduced = nbrs_reduced.kneighbors(new_reduced_data[i].reshape(1, -1),
                                               return_distance=False)[0]
        total_matches += len(set(inds_orig) & set(inds_reduced))
    total_time = time.perf_counter() - total_start
    # For diagnostic printing:
    print(f"Accuracy calc time for k={k}: {total_time:.4f}s")
    return total_matches / (len(new_original_data) * k)

# -------------------------
# Baseline method implementations
# -------------------------
def run_random_projection(X_train, X_test, target_dim):
    from sklearn.random_projection import GaussianRandomProjection
    t0 = time.perf_counter()
    rp = GaussianRandomProjection(n_components=target_dim, random_state=1)
    X_train_rp = rp.fit_transform(X_train)
    X_test_rp = rp.transform(X_test)
    t_elapsed = time.perf_counter() - t0
    return X_train_rp, X_test_rp, t_elapsed

def run_fastica(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    ica = FastICA(n_components=target_dim, random_state=1)
    X_train_ica = ica.fit_transform(X_train)
    X_test_ica = ica.transform(X_test)
    t_elapsed = time.perf_counter() - t0
    return X_train_ica, X_test_ica, t_elapsed

def run_tsne(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    tsne_model = TSNE(n_components=target_dim, method='exact', random_state=1)
    X_train_tsne = tsne_model.fit_transform(X_train)
    # Out-of-sample extension via linear regression
    reg = LinearRegression().fit(X_train, X_train_tsne)
    X_test_tsne = reg.predict(X_test)
    t_elapsed = time.perf_counter() - t0
    return X_train_tsne, X_test_tsne, t_elapsed

def run_nmf(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    # NMF requires non-negative data; shift data accordingly.
    global_min = min(X_train.min(), X_test.min())
    # Shift both training and test data so that they are non-negative.
    X_train_nmf = X_train - global_min
    X_test_nmf = X_test - global_min
    nmf_model = NMF(n_components=target_dim, init='random', random_state=1, max_iter=1000)
    X_train_nmf_trans = nmf_model.fit_transform(X_train_nmf)
    X_test_nmf_trans = nmf_model.transform(X_test_nmf)
    t_elapsed = time.perf_counter() - t0
    return X_train_nmf_trans, X_test_nmf_trans, t_elapsed

def run_lle(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    lle_model = LocallyLinearEmbedding(n_components=target_dim, n_neighbors=10, random_state=1)
    X_train_lle = lle_model.fit_transform(X_train)
    X_test_lle = lle_model.transform(X_test)
    t_elapsed = time.perf_counter() - t0
    return X_train_lle, X_test_lle, t_elapsed

def run_feature_agglomeration(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    fa_model = FeatureAgglomeration(n_clusters=target_dim)
    X_train_fa = fa_model.fit_transform(X_train)
    X_test_fa = fa_model.transform(X_test)
    t_elapsed = time.perf_counter() - t0
    return X_train_fa, X_test_fa, t_elapsed

def run_autoencoder(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    input_dim = X_train.shape[1]
    inputs = Input(shape=(input_dim,))
    encoded = Dense(target_dim, activation='relu')(inputs)
    decoded = Dense(input_dim, activation='linear')(encoded)
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    # Train the autoencoder (epochs can be increased for better performance)
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0)
    X_train_ae = encoder.predict(X_train)
    X_test_ae = encoder.predict(X_test)
    t_elapsed = time.perf_counter() - t0
    return X_train_ae, X_test_ae, t_elapsed

def run_vae(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    input_dim = X_train.shape[1]
    latent_dim = target_dim
    inputs = Input(shape=(input_dim,))
    h = Dense(128, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    # Decoder
    decoder_h = Dense(128, activation='relu')
    decoder_out = Dense(input_dim, activation='linear')
    h_decoded = decoder_h(z)
    outputs = decoder_out(h_decoded)
    vae = Model(inputs, outputs)
    # Loss: Reconstruction + KL divergence
    reconstruction_loss = tf.keras.losses.mse(inputs, outputs) * input_dim
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.fit(X_train, None, epochs=20, batch_size=32, verbose=0)
    # Use the mean as the encoded representation
    encoder = Model(inputs, z_mean)
    X_train_vae = encoder.predict(X_train)
    X_test_vae = encoder.predict(X_test)
    t_elapsed = time.perf_counter() - t0
    return X_train_vae, X_test_vae, t_elapsed

# Dictionary to map method names to functions
methods_mapping = {
    'RandomProjection': run_random_projection,
    'FastICA': run_fastica,
    'tSNE': run_tsne,
    'NMF': run_nmf,
    'LLE': run_lle,
    'Autoencoder': run_autoencoder,
    'VAE': run_vae,
    'FeatureAgglomeration': run_feature_agglomeration
}

# -------------------------
# Main pipeline over multiple datasets
# -------------------------
# Define your dataset file paths.
# (Adjust the file names as needed for your environment.)
datasets = {
    'Arcene': ('training_vectors_600_Arcene.npy', 'testing_vectors_300_Arcene.npy')
}

# Base parameter for feature selection for most datasets is 200,
# but for Fasttext use 298.
default_dim = 200

target_ratios = [0.05, 0.6]
# Target dimensions will be computed based on the dim used for that dataset.
k_values = [1, 3, 6, 10, 15]

# Loop over each dataset
for dname, (train_path, test_path) in datasets.items():
    print(f"\nProcessing dataset: {dname}")
    # Adjust sampling dim for Fasttext
    if dname == 'Fasttext':
        cur_dim = 300
    else:
    cur_dim = default_dim

    # Load the dataset (assumes .npy files)
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Files for dataset {dname} not found. Skipping...")
        continue
    training_vectors = np.load(train_path)
    testing_vectors = np.load(test_path)

    # Randomly select "cur_dim" features (if the data dimensionality is higher)
    total_dims = training_vectors.shape[1]
    if total_dims > cur_dim:
        np.random.seed(1)
        selected_dims = np.random.choice(total_dims, size=cur_dim, replace=False)
        X_train = training_vectors[:, selected_dims]
        X_test = testing_vectors[:, selected_dims]
    else:
        X_train = training_vectors
        X_test = testing_vectors

    # Standardize the data (using training mean and std)
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0] = 1  # avoid division by zero
    X_train_std = (X_train - train_mean) / train_std
    X_test_std = (X_test - train_mean) / train_std

    # Compute target dimensions for this dataset based on cur_dim
    target_dims = [max(1, int(cur_dim * r)) for r in target_ratios]

    results_rows = []
    # Loop over each method and target dimension combination
    for method_name, method_func in methods_mapping.items():
        for target_dim in target_dims:
            # Run the method on the standardized data.
            print(f"Method: {method_name}, target_dim: {target_dim}")
            X_train_red, X_test_red, method_time = method_func(X_train_std, X_test_std, target_dim)
            # For each k value, compute k-NN accuracy.
            for k in k_values:
                acc = calculate_accuracy(X_train_std, X_train_red, X_test_std, X_test_red, k)
                row = {
                    'Dataset': dname,
                    'Method': method_name,
                    'OriginalDim': cur_dim,
                    'TargetDim': target_dim,
                    'k': k,
                    'Accuracy': acc,
                    'MethodTime(s)': method_time
                }
                results_rows.append(row)

    # Save the results for this dataset to a CSV file.
    df_results = pd.DataFrame(results_rows)
    csv_filename = f"additional_baselines_{dname}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"Results for dataset {dname} saved to {csv_filename}")

print("All dataset processing completed.")
