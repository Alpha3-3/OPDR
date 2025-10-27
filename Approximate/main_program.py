import numpy as np
import pandas as pd
import time
import os
import multiprocessing as mp
from scipy.optimize import minimize
from sklearn.decomposition import PCA, KernelPCA, NMF, FastICA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist
import umap
import faiss
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import psutil
import gc
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Set random seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

# Memory monitoring utilities
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_peak_memory_usage():
    """Get peak memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().peak_wset / 1024 / 1024

def monitor_memory(func, *args, **kwargs):
    """Monitor memory usage during function execution"""
    gc.collect()  # Clean up before measurement
    start_memory = get_memory_usage()
    start_peak = get_peak_memory_usage()
    
    result = func(*args, **kwargs)
    
    end_memory = get_memory_usage()
    end_peak = get_peak_memory_usage()
    
    memory_used = end_memory - start_memory
    peak_memory = max(end_peak - start_peak, end_memory - start_memory)
    
    return result, memory_used, peak_memory

def check_gpu_availability():
    """Check GPU availability and print status"""
    try:
        num_gpus = faiss.get_num_gpus()
        if num_gpus > 0:
            print(f"[INFO] Found {num_gpus} GPU(s) available for Faiss acceleration")
            return True
        else:
            print(f"[INFO] No GPU available, using CPU for Faiss operations")
            return False
    except Exception as e:
        print(f"[WARNING] Error checking GPU availability: {e}")
        return False

def get_gpu_resource():
    """Get GPU resource if available, otherwise return None"""
    try:
        if faiss.get_num_gpus() > 0:
            return faiss.StandardGpuResources()
        else:
            return None
    except Exception as e:
        print(f"[WARNING] Error creating GPU resource: {e}")
        return None

class MPAD:
    """Multi-dimensional Projection for Approximate Distance (fast top-b via sorting & prefix sums)"""

    def __init__(self, b_percentage=1.0, alpha=0.1, target_dim=128, tol=1e-9, max_bs_iter=40):
        self.b_percentage = b_percentage
        self.alpha = alpha
        self.target_dim = target_dim
        self.projection_axes = None
        self.X_mean_ = None
        self.tol = tol                    # 数值稳定用
        self.max_bs_iter = max_bs_iter    # 二分的最大迭代

    # ---------- 核心：一次评估（给定 w 与 X_centered） ----------
    def _objective_and_grad(self, w, X, prev_ws):
        """
        返回: (f, g) 其中 f 是目标函数值（要最小化），g 是对 w 的梯度。
        实现要点：
          - v = w / ||w|| 保持单位范数；对 w 的梯度通过链式法则投影回切空间。
          - 主目标：选取投影差值 |p_i - p_j| 的 top-b% 的均值（取负号）。
          - 复杂度：O(N log N + N n)
        """
        N, n = X.shape
        if N <= 1:
            # 边界：没有成对距离
            v = w / (np.linalg.norm(w) + self.tol)
            g_pen, pen = self._ortho_grad_and_penalty(v, prev_ws)
            # d f / d w = 投影 (I - vv^T) * g_pen / ||w||
            g_w = self._project_grad_to_w(g_pen, v, w)
            return self.alpha * pen, self.alpha * g_w

        # 单位化方向
        v = w / (np.linalg.norm(w) + self.tol)

        # 1) 计算投影并排序
        p = X @ v                      # O(N n)
        order = np.argsort(p)
        s = p[order]                   # 升序
        # 前缀和（用于快速段求和）
        P = np.zeros(N + 1, dtype=s.dtype)
        P[1:] = np.cumsum(s)

        # 2) 需要的对数 B（top-b%）
        total_pairs = N * (N - 1) // 2
        B = max(1, min(total_pairs, int(round(self.b_percentage / 100.0 * total_pairs))))

        # 3) 二分阈值 Δ*，使得 count(>|Δ*|) <= B <= count(>=|Δ*|)
        # 使用双指针计数：对每个 i，找到 j_gt(i) 是第一个满足 s[j] - s[i] > Δ 的 j
        #                                j_ge(i) 是第一个满足 s[j] - s[i] >= Δ 的 j
        min_gap = float(np.inf)
        for i in range(N - 1):
            gap = s[i + 1] - s[i]
            if gap < min_gap:
                min_gap = gap
        if not np.isfinite(min_gap) or min_gap < 0:
            min_gap = 0.0

        lo = 0.0
        hi = (s[-1] - s[0]) + self.tol
        # 预分配工作数组以减少开销
        j_ge = np.empty(N, dtype=np.int64)
        j_gt = np.empty(N, dtype=np.int64)

        def count_pairs(delta, strict=False):
            # 返回 sum_i (N - j_* + 1) 的计数
            j = 0
            cnt = 0
            if strict:
                for i in range(N):
                    if j < i + 1:
                        j = i + 1
                    while j < N and s[j] - s[i] <= delta - self.tol:
                        j += 1
                    j_gt[i] = j if j < N else N
                    if j_gt[i] < N:
                        cnt += (N - j_gt[i])
                return cnt
            else:
                for i in range(N):
                    if j < i + 1:
                        j = i + 1
                    while j < N and s[j] - s[i] < delta - self.tol:
                        j += 1
                    j_ge[i] = j if j < N else N
                    if j_ge[i] < N:
                        cnt += (N - j_ge[i])
                return cnt

        # 二分 Δ*
        for _ in range(self.max_bs_iter):
            mid = 0.5 * (lo + hi)
            c_ge = count_pairs(mid, strict=False)
            c_gt = count_pairs(mid, strict=True)
            if c_ge < B:
                # 需要更小阈值以获得更多对
                hi = mid
            elif c_gt > B:
                # 需要更大阈值以减少严格大于的对数
                lo = mid
            else:
                lo = hi = mid
                break

        Delta = 0.5 * (lo + hi)

        # 再计算一次精确的三组量：count_gt / sum_gt / count_ge (>=) / equal-layer 的分解
        # 4) 以 j_gt / j_ge 为界，计算
        #    count_gt = sum_i (N - j_gt[i])
        #    sum_gt   = sum_i (sum_{j=j_gt[i]}^{N-1} (s[j] - s[i]))
        #    equal 区间：j in [j_ge[i], j_gt[i]-1]
        # 计算 sum_gt 用前缀和 O(N)
        # 以及"差分数组"做区间 +1，用于后续梯度的 c（右端计数）
        count_gt = 0
        sum_gt = 0.0

        # 右端计数（对每个 j 被作为"右端点"的次数）
        right_cnt_diff = np.zeros(N + 1, dtype=np.int64)
        # 左端计数（对每个 i 作为"左端点"参与的对数）
        left_gt_cnt = np.zeros(N, dtype=np.int64)

        # 先确保 j_gt 是一致的
        _ = count_pairs(Delta, strict=True)  # 填充 j_gt
        for i in range(N):
            j = j_gt[i]
            if j < N:
                cnt_i = N - j
                left_gt_cnt[i] = cnt_i
                count_gt += cnt_i
                # sum_{j=j}^{N-1} s[j] = P[N] - P[j]
                sum_gt += (P[N] - P[j]) - cnt_i * s[i]
                # 右端点区间加 1
                right_cnt_diff[j] += 1
                right_cnt_diff[N] -= 1

        right_gt_cnt = np.cumsum(right_cnt_diff[:-1])  # 长度 N

        # 5) 还需要从"等于 Δ 的层"中取 R = B - count_gt 个
        R = B - count_gt
        if R < 0:
            R = 0

        # 计算等于层的区间，并逐 i 取用 eq_take_i，使得总和为 R
        # 同时统计等于层对梯度的贡献（每取一个 (i,j) 就对 c[j] += 1, c[i] -= 1）
        eq_take_per_i = np.zeros(N, dtype=np.int64)
        if R > 0:
            # 先确保 j_ge 已正确
            _ = count_pairs(Delta, strict=False)  # 填充 j_ge
            # 用差分数组快速对 j 区间累加 eq_take_i
            eq_right_diff = np.zeros(N + 1, dtype=np.int64)
            for i in range(N):
                L = j_ge[i]
                Rg = j_gt[i] - 1  # inclusive
                if L < N and Rg >= L:
                    can = Rg - L + 1
                    take = can if R >= can else R
                    if take > 0:
                        eq_take_per_i[i] = take
                        eq_right_diff[L] += take
                        if L + take <= N - 1:
                            eq_right_diff[L + take] -= take
                        R -= take
                        if R == 0:
                            break
            eq_right_cnt = np.cumsum(eq_right_diff[:-1])  # 每个 j 作为"等于层右端点"的次数
        else:
            eq_right_cnt = np.zeros(N, dtype=np.int64)

        # 6) 目标函数的"被选 top-b 对"的总和 = sum_gt + (sum of R * Delta)
        sum_topB = sum_gt + (B - count_gt) * Delta
        mean_topB = sum_topB / float(B)
        # 主目标是取负号（越大越好 -> 最小化取负）
        main_obj = -mean_topB

        # 7) 构造对 p 的梯度系数 c：对每一对 (i,j)，|p_i - p_j| 的梯度是：
        #    对 j: +1, 对 i: -1 （因为 s[j] >= s[i] 保证符号为正）
        #    >Δ 的对：i 端 -left_gt_cnt[i]，j 端 +right_gt_cnt[j]
        #    ==Δ 的被取 R 部分：i 端 -eq_take_per_i[i]，j 端 +eq_right_cnt[j]
        c_sorted = right_gt_cnt.astype(np.float64) - 0.0  # 先放右端 +1 计数
        c_sorted -= left_gt_cnt.astype(np.float64)        # 左端 -count
        c_sorted += eq_right_cnt.astype(np.float64)       # 等于层右端 +1
        c_sorted -= eq_take_per_i.astype(np.float64)      # 等于层左端 -1

        # 把 c 从排序顺序还原到原顺序
        c = np.zeros(N, dtype=np.float64)
        c[order] = c_sorted

        # 8) 把对 p 的梯度回传到 w：g_v = X^T c
        g_main_v = X.T @ c  # O(N n)

        # 9) 加上正交惩罚（对 v），并合成到对 w 的梯度
        g_pen_v, pen = self._ortho_grad_and_penalty(v, prev_ws)

        # 10) 总的对 v 的梯度
        g_v = - (g_main_v / float(B)) + self.alpha * g_pen_v

        # 11) 投影到对 w 的梯度：若 v = w / ||w||，则
        #     d f / d w = (I - v v^T) g_v / ||w||
        g_w = self._project_grad_to_w(g_v, v, w)

        # 总目标：主目标 + 正交项
        f = main_obj + self.alpha * pen
        return f, g_w

    def _project_grad_to_w(self, g_v, v, w):
        """把对单位向量 v 的梯度投影为对 w 的梯度"""
        wn = np.linalg.norm(w) + self.tol
        # 投影到切空间 (I - vv^T) g_v
        gv_tangent = g_v - v * (v @ g_v)
        return gv_tangent / wn

    def _ortho_grad_and_penalty(self, v, prev_ws):
        """正交惩罚 ∑ (v·u)^2 及其对 v 的梯度 2∑(v·u)u"""
        if not prev_ws:
            return np.zeros_like(v), 0.0
        U = np.column_stack(prev_ws)   # n x k
        Vu = U.T @ v                   # k
        pen = float(np.sum(Vu * Vu))
        g = 2.0 * (U @ Vu)             # n
        return g, pen

    # ---------- 公开 API ----------
    def fit_transform(self, X):
        """Fit MPAD and transform data."""
        X = np.asarray(X, dtype=float)
        self.X_mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.X_mean_

        n = Xc.shape[1]
        prev_ws = []
        optimal_ws = []

        for axis in range(self.target_dim):
            # 目标 + 解析梯度
            def fun(w):
                f, g = self._objective_and_grad(w, Xc, prev_ws)
                return f, g

            # 随机初始化并单位化
            w0 = np.random.randn(n)
            w0 /= (np.linalg.norm(w0) + self.tol)

            res = minimize(fun, w0, method='L-BFGS-B', jac=True)
            v_opt = res.x / (np.linalg.norm(res.x) + self.tol)
            prev_ws.append(v_opt)
            optimal_ws.append(v_opt)

        self.projection_axes = np.column_stack(optimal_ws)  # n x m
        return Xc @ self.projection_axes                    # N x m

    def transform(self, X):
        """Transform new data using fitted projection axes (uses training mean)."""
        if self.projection_axes is None or self.X_mean_ is None:
            raise ValueError("Must fit the model before transforming")
        X = np.asarray(X, dtype=float)
        Xc = X - self.X_mean_
        return Xc @ self.projection_axes

class BaselineMethods:
    """Collection of baseline dimensionality reduction methods"""
    
    @staticmethod
    def run_pca(X_train, X_test, target_dim):
        """Principal Component Analysis"""
        pca = PCA(n_components=target_dim, random_state=1)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_test_pca
    
    @staticmethod
    def run_umap(X_train, X_test, target_dim):
        """UMAP"""
        umap_model = umap.UMAP(n_components=target_dim, random_state=1, n_jobs=1)
        X_train_umap = umap_model.fit_transform(X_train)
        X_test_umap = umap_model.transform(X_test)
        return X_train_umap, X_test_umap
    
    @staticmethod
    def run_isomap(X_train, X_test, target_dim):
        """Isomap"""
        isomap_model = Isomap(n_components=target_dim, n_neighbors=min(10, X_train.shape[0]-1))
        X_train_isomap = isomap_model.fit_transform(X_train)
        X_test_isomap = isomap_model.transform(X_test)
        return X_train_isomap, X_test_isomap
    
    @staticmethod
    def run_kernel_pca(X_train, X_test, target_dim):
        """Kernel PCA"""
        kpca = KernelPCA(n_components=target_dim, kernel='rbf', random_state=1)
        X_train_kpca = kpca.fit_transform(X_train)
        X_test_kpca = kpca.transform(X_test)
        return X_train_kpca, X_test_kpca
    
    @staticmethod
    def run_random_projection(X_train, X_test, target_dim):
        """Random Projection"""
        rp = GaussianRandomProjection(n_components=target_dim, random_state=1)
        X_train_rp = rp.fit_transform(X_train)
        X_test_rp = rp.transform(X_test)
        return X_train_rp, X_test_rp
    
    @staticmethod
    def run_tsne(X_train, X_test, target_dim):
        """t-SNE with out-of-sample extension (removed - incompatible with high dimensions)"""
        # t-SNE removed due to barnes_hut limitation (n_components < 4)
        # This method is disabled
        raise NotImplementedError("t-SNE has been removed due to dimension limitations")
    
    @staticmethod
    def run_nmf(X_train, X_test, target_dim):
        """Non-negative Matrix Factorization"""
        # Ensure non-negative data
        min_val = min(X_train.min(), X_test.min())
        X_train_nmf = X_train - min_val
        X_test_nmf = X_test - min_val
        
        nmf = NMF(n_components=target_dim, random_state=1, max_iter=1000)
        X_train_nmf_trans = nmf.fit_transform(X_train_nmf)
        X_test_nmf_trans = nmf.transform(X_test_nmf)
        return X_train_nmf_trans, X_test_nmf_trans
    
    @staticmethod
    def run_lle(X_train, X_test, target_dim):
        """Locally Linear Embedding"""
        lle = LocallyLinearEmbedding(n_components=target_dim, n_neighbors=min(10, X_train.shape[0]-1), random_state=1)
        X_train_lle = lle.fit_transform(X_train)
        X_test_lle = lle.transform(X_test)
        return X_train_lle, X_test_lle
    
    @staticmethod
    def run_feature_agglomeration(X_train, X_test, target_dim):
        """Feature Agglomeration"""
        fa = FeatureAgglomeration(n_clusters=target_dim)
        X_train_fa = fa.fit_transform(X_train)
        X_test_fa = fa.transform(X_test)
        return X_train_fa, X_test_fa
    
    @staticmethod
    def run_autoencoder(X_train, X_test, target_dim):
        """Autoencoder"""
        input_dim = X_train.shape[1]
        
        # Encoder
        inputs = Input(shape=(input_dim,))
        encoded = Dense(target_dim, activation='relu')(inputs)
        
        # Decoder
        decoded = Dense(input_dim, activation='linear')(encoded)
        
        # Autoencoder model
        autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0)
        
        X_train_ae = encoder.predict(X_train, batch_size=256)
        X_test_ae = encoder.predict(X_test, batch_size=256)
        
        K.clear_session()
        return X_train_ae, X_test_ae
    
    @staticmethod
    def run_vae(X_train, X_test, target_dim):
        """Variational Autoencoder"""
        input_dim = X_train.shape[1]
        latent_dim = target_dim
        
        # Encoder
        inputs = Input(shape=(input_dim,))
        h = Dense(128, activation='relu')(inputs)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        z = Lambda(sampling)([z_mean, z_log_var])
        
        # Decoder
        decoder_h = Dense(128, activation='relu')
        decoder_out = Dense(input_dim, activation='sigmoid')
        
        h_decoded = decoder_h(z)
        outputs = decoder_out(h_decoded)
        
        # VAE model
        vae = Model(inputs, outputs)
        
        # Loss function
        reconstruction_loss = tf.keras.losses.mse(inputs, outputs) * input_dim
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        
        vae.compile(optimizer='adam')
        vae.fit(X_train, None, epochs=20, batch_size=32, verbose=0)
        
        # Encoder for inference
        encoder = Model(inputs, z_mean)
        X_train_vae = encoder.predict(X_train, batch_size=256)
        X_test_vae = encoder.predict(X_test, batch_size=256)
        
        K.clear_session()
        return X_train_vae, X_test_vae

class IndexMethods:
    """Collection of indexing methods for approximate nearest neighbor search"""
    
    @staticmethod
    def exact_knn(data_train, data_test, k):
        """Exact k-NN using Faiss IndexFlatL2 (ground truth)"""
        dim = data_train.shape[1]
        
        # Convert to float32 for Faiss
        data_train_f32 = data_train.astype(np.float32)
        data_test_f32 = data_test.astype(np.float32)
        
        # Check if GPU is available
        use_gpu = faiss.get_num_gpus() > 0
        
        if use_gpu:
            try:
                # Create GPU resource
                gpu_resource = faiss.StandardGpuResources()
                
                # Create CPU index
                cpu_index = faiss.IndexFlatL2(dim)
                
                # Move to GPU
                index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
                
                print(f"[GPU] Using GPU acceleration for exact k-NN")
            except Exception as e:
                print(f"[WARNING] GPU failed, falling back to CPU: {e}")
                use_gpu = False
        
        if not use_gpu:
            # Create CPU index
            index = faiss.IndexFlatL2(dim)
            print(f"[CPU] Using CPU for exact k-NN")
        
        # Add vectors to index
        index.add(data_train_f32)
        
        # Search
        distances, indices = index.search(data_test_f32, k)
        
        return indices
    
    @staticmethod
    def hnswflat_faiss(data_train, data_test, k):
        """HNSWFlat using Faiss with GPU support"""
        dim = data_train.shape[1]
        
        # Convert to float32 for Faiss
        data_train_f32 = data_train.astype(np.float32)
        data_test_f32 = data_test.astype(np.float32)
        
        # Check if GPU is available
        use_gpu = faiss.get_num_gpus() > 0
        
        if use_gpu:
            try:
                # Create GPU resource
                gpu_resource = faiss.StandardGpuResources()
                
                # Create CPU index
                cpu_index = faiss.IndexHNSWFlat(dim, 32)
                cpu_index.hnsw.efConstruction = 40
                cpu_index.hnsw.efSearch = 50
                
                # Move to GPU
                index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
                
                print(f"[GPU] Using GPU acceleration for HNSWFlat")
            except Exception as e:
                print(f"[WARNING] GPU failed, falling back to CPU: {e}")
                use_gpu = False
        
        if not use_gpu:
            # Create CPU index
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 50
            print(f"[CPU] Using CPU for HNSWFlat")
        
        # Add vectors to index
        index.add(data_train_f32)
        
        # Search
        distances, indices = index.search(data_test_f32, k)
        return indices
    
    @staticmethod
    def ivfpq_faiss(data_train, data_test, k):
        """IVFPQ using Faiss with GPU support"""
        dim = data_train.shape[1]
        n_train = data_train.shape[0]
        
        # Convert to float32 for Faiss
        data_train_f32 = data_train.astype(np.float32)
        data_test_f32 = data_test.astype(np.float32)
        
        # Determine parameters - adjust for small datasets
        if n_train < 200:
            nlist = min(5, max(1, n_train // 50))  # Very small nlist for small datasets
        else:
            nlist = min(100, max(1, n_train // 39))
        
        m_pq = min(8, dim) if dim >= 8 else dim
        nbits = 8
        
        # Check if GPU is available
        use_gpu = faiss.get_num_gpus() > 0
        
        if use_gpu:
            try:
                # Create GPU resource
                gpu_resource = faiss.StandardGpuResources()
                
                # Create CPU quantizer
                cpu_quantizer = faiss.IndexFlatL2(dim)
                
                # Create CPU IVFPQ index
                cpu_index = faiss.IndexIVFPQ(cpu_quantizer, dim, nlist, m_pq, nbits)
                
                # Move to GPU
                index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
                
                print(f"[GPU] Using GPU acceleration for IVFPQ")
            except Exception as e:
                print(f"[WARNING] GPU failed, falling back to CPU: {e}")
                use_gpu = False
        
        if not use_gpu:
            # Create CPU quantizer
            quantizer = faiss.IndexFlatL2(dim)
            
            # Create CPU IVFPQ index
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits)
            print(f"[CPU] Using CPU for IVFPQ")
        
        # Train and add vectors
        index.train(data_train_f32)
        index.add(data_train_f32)
        index.nprobe = min(nlist, 10)
        
        # Search
        distances, indices = index.search(data_test_f32, k)
        return indices
    
    @staticmethod
    def ivf_pqr_faiss(data_train, data_test, k):
        """IVF-PQR using Faiss with GPU support (Product Quantization with Residual)"""
        dim = data_train.shape[1]
        n_train = data_train.shape[0]
        
        # Convert to float32 for Faiss
        data_train_f32 = data_train.astype(np.float32)
        data_test_f32 = data_test.astype(np.float32)
        
        # Determine parameters for IVF-PQR - adjust for small datasets
        if n_train < 200:
            nlist = min(5, max(1, n_train // 50))  # Very small nlist for small datasets
        else:
            nlist = min(100, max(1, n_train // 39))
        
        m_pq = min(8, dim) if dim >= 8 else dim
        nbits = 8
        m_refine = min(4, dim) if dim >= 4 else dim  # Additional parameter for PQR
        
        # Check if GPU is available
        use_gpu = faiss.get_num_gpus() > 0
        
        if use_gpu:
            try:
                # Create GPU resource
                gpu_resource = faiss.StandardGpuResources()
                
                # Create CPU quantizer
                cpu_quantizer = faiss.IndexFlatL2(dim)
                
                # Create CPU IVF-PQR index with correct parameters
                cpu_index = faiss.IndexIVFPQR(cpu_quantizer, dim, nlist, m_pq, nbits, m_refine, nbits)
                
                # Move to GPU
                index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
                
                print(f"[GPU] Using GPU acceleration for IVF-PQR")
            except Exception as e:
                print(f"[WARNING] GPU failed, falling back to CPU: {e}")
                use_gpu = False
        
        if not use_gpu:
            # Create CPU quantizer
            quantizer = faiss.IndexFlatL2(dim)
            
            # Create CPU IVF-PQR index with correct parameters
            index = faiss.IndexIVFPQR(quantizer, dim, nlist, m_pq, nbits, m_refine, nbits)
            print(f"[CPU] Using CPU for IVF-PQR")
        
        # Train and add vectors
        index.train(data_train_f32)
        index.add(data_train_f32)
        index.nprobe = min(nlist, 10)
        
        # Search
        distances, indices = index.search(data_test_f32, k)
        return indices
    
    @staticmethod
    def ivf_opq_pq_faiss(data_train, data_test, k):
        """IVF-OPQ-PQ using Faiss with GPU support (Optimized Product Quantization)"""
        dim = data_train.shape[1]
        n_train = data_train.shape[0]
        
        # Convert to float32 for Faiss
        data_train_f32 = data_train.astype(np.float32)
        data_test_f32 = data_test.astype(np.float32)
        
        # Determine parameters for IVF-OPQ-PQ
        # For small datasets, use fewer clusters
        if n_train < 200:
            nlist = min(5, max(1, n_train // 50))  # Very small nlist for small datasets
        else:
            nlist = min(50, max(1, n_train // 10))
        
        m_pq = min(8, dim) if dim >= 8 else dim
        nbits = 8
        
        # Check if GPU is available
        use_gpu = faiss.get_num_gpus() > 0
        
        try:
            if use_gpu:
                try:
                    # Create GPU resource
                    gpu_resource = faiss.StandardGpuResources()
                    
                    # Create CPU quantizer (L2 distance)
                    cpu_quantizer = faiss.IndexFlatL2(dim)
                    
                    # Create OPQ matrix for preprocessing
                    opq_matrix = faiss.OPQMatrix(dim, m_pq)
                    
                    # Create CPU IVF-PQ index
                    cpu_index = faiss.IndexIVFPQ(cpu_quantizer, dim, nlist, m_pq, nbits)
                    
                    # Apply OPQ preprocessing
                    cpu_index = faiss.IndexPreTransform(opq_matrix, cpu_index)
                    
                    # Move to GPU
                    index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
                    
                    print(f"[GPU] Using GPU acceleration for IVF-OPQ-PQ")
                except Exception as e:
                    print(f"[WARNING] GPU failed, falling back to CPU: {e}")
                    use_gpu = False
            
            if not use_gpu:
                # Create CPU quantizer (L2 distance)
                quantizer = faiss.IndexFlatL2(dim)
                
                # Create OPQ matrix for preprocessing
                opq_matrix = faiss.OPQMatrix(dim, m_pq)
                
                # Create CPU IVF-PQ index
                index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits)
                
                # Apply OPQ preprocessing
                index = faiss.IndexPreTransform(opq_matrix, index)
                print(f"[CPU] Using CPU for IVF-OPQ-PQ")
            
            # Train and add vectors
            index.train(data_train_f32)
            index.add(data_train_f32)
            index.nprobe = min(nlist, 10)
            
            # Search
            distances, indices = index.search(data_test_f32, k)
            return indices
            
        except Exception as e:
            print(f"[WARNING] IVF-OPQ-PQ failed, falling back to IVFPQ: {e}")
            # Fallback to regular IVFPQ
            return IndexMethods.ivfpq_faiss(data_train, data_test, k)

def calculate_recall_at_k(true_indices, predicted_indices, k):
    """Calculate Recall@k"""
    recalls = []
    for i in range(len(true_indices)):
        true_set = set(true_indices[i])
        pred_set = set(predicted_indices[i])
        recall = len(true_set.intersection(pred_set)) / len(true_set)
        recalls.append(recall)
    return np.mean(recalls)

def evaluate_method(method_name, method_func, X_train, X_test, target_dim, index_methods, k_values, 
                    true_indices_orig=None):
    """
    Evaluate a single dimensionality reduction method with detailed timing and memory usage.
    
    Args:
        method_name: Name of the DR method
        method_func: Function to apply DR
        X_train, X_test: Original (unreduced) training and test data
        target_dim: Target dimension for DR
        index_methods: Dictionary of index methods
        k_values: List of k values for recall calculation
        true_indices_orig: Ground truth indices from ORIGINAL space (computed once, shared across all methods)
    
    Returns:
        results: Dictionary containing all evaluation metrics
    """
    import time
    
    print(f"  [INFO] Starting evaluation of {method_name}")
    print(f"  [INFO] Input dimensions: train={X_train.shape}, test={X_test.shape}")
    print(f"  [INFO] Target dimension: {target_dim}")
    
    results = {}
    
    try:
        # Apply dimensionality reduction with timing and memory monitoring
        print(f"  [STEP 1] Applying {method_name} dimensionality reduction...")
        start_time = time.time()
        start_memory = get_memory_usage()
        
        X_train_reduced, X_test_reduced = method_func(X_train, X_test, target_dim)
        
        dr_time = time.time() - start_time
        end_memory = get_memory_usage()
        dr_memory = end_memory - start_memory
        
        results['dr_time'] = dr_time
        results['dr_memory'] = dr_memory
        
        print(f"  [STEP 1] [OK] Completed in {dr_time:.4f}s, Memory: {dr_memory:.2f}MB")
        print(f"  [INFO] Reduced dimensions: train={X_train_reduced.shape}, test={X_test_reduced.shape}")
        
        # Ground truth should be from ORIGINAL space, not reduced space
        # This is passed in as a parameter (computed once, shared across all methods)
        if true_indices_orig is None:
            raise ValueError("true_indices_orig must be provided (ground truth from original space)")
        
        true_indices = true_indices_orig
        results['gt_time'] = 0  # GT time is tracked separately in main_evaluation
        results['gt_memory'] = 0
        
        print(f"  [STEP 2] Using ground truth from ORIGINAL space (pre-computed)")
        
        # Evaluate each index method ON REDUCED SPACE
        # Compare against ground truth from ORIGINAL space
        for index_name, index_func in index_methods.items():
            print(f"  [STEP 3] Evaluating {index_name} on reduced space...")
            results[index_name] = {}
            
            for k in k_values:
                print(f"    [INFO] Testing k={k}...")
                start_time = time.time()
                start_memory = get_memory_usage()
                
                try:
                    # Compute kNN in REDUCED space
                    pred_indices = index_func(X_train_reduced, X_test_reduced, k)
                    
                    search_time = time.time() - start_time
                    end_memory = get_memory_usage()
                    search_memory = end_memory - start_memory
                    
                    # Calculate Recall@k: Compare reduced-space kNN vs original-space ground truth
                    recall = calculate_recall_at_k(true_indices[:, :k], pred_indices, k)
                    
                    results[index_name][k] = {
                        'recall': recall,
                        'time': search_time,
                        'memory': search_memory,
                        'indices': pred_indices  # Store indices for caching
                    }
                    
                    print(f"    [OK] k={k}: Recall={recall:.4f}, Time={search_time:.4f}s, Memory={search_memory:.2f}MB")
                    
                except Exception as e:
                    print(f"    [ERROR] k={k}: {e}")
                    results[index_name][k] = {
                        'recall': np.nan,
                        'time': np.nan,
                        'memory': np.nan
                    }
                    
    except Exception as e:
        print(f"  [ERROR] Failed to evaluate {method_name}: {e}")
        results['dr_time'] = np.nan
        results['dr_memory'] = np.nan
        for index_name in index_methods.keys():
            if index_name != 'IndexFlat_kNN':
                results[index_name] = {k: {'recall': np.nan, 'time': np.nan, 'memory': np.nan} for k in k_values}
    
    print(f"  [INFO] Completed evaluation of {method_name}")
    
    # Store additional data for caching
    try:
        results['X_train_reduced'] = X_train_reduced
        results['X_test_reduced'] = X_test_reduced
        results['true_indices'] = true_indices
    except NameError:
        # X_train_reduced, X_test_reduced, or true_indices not defined
        pass
    
    return results

def save_results_to_csv(all_results, dataset_name, target_dim, b_percentage, alpha, k_values, output_dir="Result"):
    """Save results to CSV files with detailed metrics"""
    import pandas as pd
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for CSV
    rows = []
    
    for method_name, method_results in all_results.items():
        # Skip error entries
        if 'error' in method_results:
            continue
            
        # Extract DR metrics
        dr_time = method_results.get('dr_time', np.nan)
        dr_memory = method_results.get('dr_memory', np.nan)
        gt_time = method_results.get('gt_time', np.nan)
        gt_memory = method_results.get('gt_memory', np.nan)
        
        # Extract index method results
        for index_name, index_results in method_results.items():
            # Skip non-index keys
            if index_name in ['dr_time', 'dr_memory', 'gt_time', 'gt_memory', 
                            'X_train_reduced', 'X_test_reduced', 'true_indices']:
                continue
            
            # index_results should be a dict mapping k -> {recall, time, memory}
            if not isinstance(index_results, dict):
                continue
                
            for k in k_values:
                if k in index_results and isinstance(index_results[k], dict):
                    recall = index_results[k].get('recall', np.nan)
                    search_time = index_results[k].get('time', np.nan)
                    search_memory = index_results[k].get('memory', np.nan)
                    
                    rows.append({
                        'dataset': dataset_name,
                        'method': method_name,
                        'index_method': index_name,
                        'target_dim': target_dim,
                        'b_percentage': b_percentage,
                        'alpha': alpha,
                        'k': k,
                        'recall_at_k': recall,
                        'dr_time': dr_time,
                        'dr_memory_mb': dr_memory,
                        'gt_time': gt_time,
                        'gt_memory_mb': gt_memory,
                        'search_time': search_time,
                        'search_memory_mb': search_memory,
                        'total_time': dr_time + gt_time + search_time if not np.isnan(dr_time + gt_time + search_time) else np.nan,
                        'total_memory_mb': dr_memory + gt_memory + search_memory if not np.isnan(dr_memory + gt_memory + search_memory) else np.nan
                    })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Generate filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/results_{dataset_name}_TD{target_dim}_b{b_percentage}_alpha{alpha}_{timestamp}.csv"
    
    df.to_csv(filename, index=False)
    print(f"[SAVE] Results saved to: {filename}")
    
    return filename, df

def save_reduced_data_and_results(method_name, X_train_reduced, X_test_reduced, true_indices, index_results, 
                                  dataset_name, target_dim, b_percentage, alpha, k_values, output_dir="Result/cache"):
    """Save reduced data and k-NN results to cache for future use"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectory for this method
    method_dir = os.path.join(output_dir, f"{dataset_name}_{method_name}")
    os.makedirs(method_dir, exist_ok=True)
    
    # Save reduced data
    train_reduced_file = os.path.join(method_dir, f"train_reduced_TD{target_dim}.npy")
    test_reduced_file = os.path.join(method_dir, f"test_reduced_TD{target_dim}.npy")
    np.save(train_reduced_file, X_train_reduced)
    np.save(test_reduced_file, X_test_reduced)
    
    # Save ground truth
    gt_file = os.path.join(method_dir, f"ground_truth_k{max(k_values)}.npy")
    np.save(gt_file, true_indices)
    
    # Save index results for each index method and k
    for index_name, index_result in index_results.items():
        if index_name not in ['dr_time', 'dr_memory', 'gt_time', 'gt_memory']:
            for k in k_values:
                if k in index_result:
                    indices = index_result[k].get('indices', None) if isinstance(index_result[k], dict) else index_result[k]
                    if indices is not None:
                        result_file = os.path.join(method_dir, f"{index_name}_k{k}_indices.npy")
                        np.save(result_file, indices)
    
    return method_dir

def save_summary_report(all_results, dataset_name, target_dim, b_percentage, alpha, k_values, output_dir="Result"):
    """Save a summary report with key metrics"""
    import pandas as pd
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate summary
    summary_rows = []
    
    for method_name, method_results in all_results.items():
        dr_time = method_results.get('dr_time', np.nan)
        dr_memory = method_results.get('dr_memory', np.nan)
        
        # Calculate average recall across all index methods and k values
        recalls = []
        search_times = []
        search_memories = []
        
        for index_name, index_results in method_results.items():
            # Skip non-index method keys
            if index_name in ['dr_time', 'dr_memory', 'gt_time', 'gt_memory', 
                            'gt_time_orig', 'gt_memory_orig',
                            'X_train_reduced', 'X_test_reduced', 'true_indices']:
                continue
            
            # Skip if not a dictionary (should be dict mapping k -> results)
            if not isinstance(index_results, dict):
                continue
                
            for k in k_values:
                if k in index_results and isinstance(index_results[k], dict):
                    recall = index_results[k].get('recall', np.nan)
                    search_time = index_results[k].get('time', np.nan)
                    search_memory = index_results[k].get('memory', np.nan)
                    
                    if not np.isnan(recall):
                        recalls.append(recall)
                    if not np.isnan(search_time):
                        search_times.append(search_time)
                    if not np.isnan(search_memory):
                        search_memories.append(search_memory)
        
        avg_recall = np.mean(recalls) if recalls else np.nan
        avg_search_time = np.mean(search_times) if search_times else np.nan
        avg_search_memory = np.mean(search_memories) if search_memories else np.nan
        
        summary_rows.append({
            'dataset': dataset_name,
            'method': method_name,
            'target_dim': target_dim,
            'b_percentage': b_percentage,
            'alpha': alpha,
            'avg_recall_at_k': avg_recall,
            'dr_time': dr_time,
            'dr_memory_mb': dr_memory,
            'avg_search_time': avg_search_time,
            'avg_search_memory_mb': avg_search_memory,
            'total_time': dr_time + avg_search_time if not np.isnan(dr_time + avg_search_time) else np.nan,
            'total_memory_mb': dr_memory + avg_search_memory if not np.isnan(dr_memory + avg_search_memory) else np.nan
        })
    
    # Create DataFrame and save
    df_summary = pd.DataFrame(summary_rows)
    
    # Generate filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/summary_{dataset_name}_TD{target_dim}_b{b_percentage}_alpha{alpha}_{timestamp}.csv"
    
    df_summary.to_csv(filename, index=False)
    print(f"[SAVE] Summary saved to: {filename}")
    
    return filename, df_summary

def main_evaluation(dataset_name, train_file, test_file, target_dim, b_percentage, alpha, k_values, save_results=True, output_dir="Result"):
    """Main evaluation function"""
    print(f"\n=== Evaluating {dataset_name} ===")
    print(f"Target dim: {target_dim}, b: {b_percentage}%, alpha: {alpha}")
    
    # Check GPU availability
    check_gpu_availability()
    
    # Load data
    X_train = np.load(train_file)
    X_test = np.load(test_file)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Define methods
    methods = {
        'MPAD': lambda X_tr, X_te, td: (lambda: (lambda mpad: (mpad.fit_transform(X_tr), mpad.transform(X_te)))(MPAD(b_percentage, alpha, td)))(),
        'PCA': BaselineMethods.run_pca,
        'UMAP': BaselineMethods.run_umap,
        'Isomap': BaselineMethods.run_isomap,
        'KernelPCA': BaselineMethods.run_kernel_pca,
        'RandomProjection': BaselineMethods.run_random_projection,
        # 'tSNE': BaselineMethods.run_tsne,  # Removed due to dimension limitations
        'NMF': BaselineMethods.run_nmf,
        'LLE': BaselineMethods.run_lle,
        'FeatureAgglomeration': BaselineMethods.run_feature_agglomeration,
        'Autoencoder': BaselineMethods.run_autoencoder,
        'VAE': BaselineMethods.run_vae,
    }
    
    # Define index methods
    index_methods = {
        'IndexFlat_kNN': IndexMethods.exact_knn,
        'HNSWFlat': IndexMethods.hnswflat_faiss,
        'IVFPQ': IndexMethods.ivfpq_faiss,
        'IVF_PQR': IndexMethods.ivf_pqr_faiss,
        'IVF_OPQ_PQ': IndexMethods.ivf_opq_pq_faiss,
    }
    
    # ===== CRITICAL: Calculate ground truth in ORIGINAL SPACE (once for all methods) =====
    print(f"\n{'='*80}")
    print("COMPUTING GROUND TRUTH IN ORIGINAL SPACE")
    print(f"{'='*80}")
    print(f"[INFO] Computing exact kNN on ORIGINAL data (train={X_train.shape}, test={X_test.shape})")
    print(f"[INFO] This ground truth will be used to evaluate ALL dimensionality reduction methods")
    
    import time
    start_time = time.time()
    start_memory = get_memory_usage()
    
    true_indices_orig = IndexMethods.exact_knn(X_train, X_test, max(k_values))
    
    gt_time = time.time() - start_time
    end_memory = get_memory_usage()
    gt_memory = end_memory - start_memory
    
    print(f"[OK] Ground truth computed in {gt_time:.4f}s, Memory: {gt_memory:.2f}MB")
    print(f"[INFO] Ground truth shape: {true_indices_orig.shape}")
    print(f"{'='*80}\n")
    
    # Evaluate all methods
    all_results = {}
    total_methods = len(methods)
    
    print(f"\n[PROGRESS] Starting evaluation of {total_methods} dimensionality reduction methods")
    print(f"[PROGRESS] Each method will be tested with {len(index_methods)} index methods and {len(k_values)} k values")
    print(f"[PROGRESS] All methods will be compared against the ORIGINAL-space ground truth\n")
    
    for i, (method_name, method_func) in enumerate(methods.items(), 1):
        print(f"\n[PROGRESS] Method {i}/{total_methods}: {method_name}")
        print("=" * 60)
        
        results = evaluate_method(method_name, method_func, X_train, X_test, target_dim, 
                                 index_methods, k_values, true_indices_orig=true_indices_orig)
        results['gt_time_orig'] = gt_time  # Store original space GT time
        results['gt_memory_orig'] = gt_memory
        all_results[method_name] = results
        
        # Save reduced data and results to cache
        if save_results:
            X_train_reduced = results.pop('X_train_reduced', None)
            X_test_reduced = results.pop('X_test_reduced', None)
            true_indices = results.pop('true_indices', None)
            
            if X_train_reduced is not None and X_test_reduced is not None and true_indices is not None:
                try:
                    cache_dir = save_reduced_data_and_results(
                        method_name, X_train_reduced, X_test_reduced, true_indices, results,
                        dataset_name, target_dim, b_percentage, alpha, k_values,
                        output_dir=os.path.join(output_dir, "cache")
                    )
                    print(f"[SAVE] Cached reduced data: {cache_dir}")
                except Exception as e:
                    print(f"[WARNING] Failed to cache data: {e}")
        
        print(f"[PROGRESS] Completed {method_name} ({i}/{total_methods})")
    
    print(f"\n[PROGRESS] All {total_methods} methods completed!")
    
    # Save results if requested
    if save_results:
        print(f"\n[SAVE] Saving results to {output_dir}/")
        
        # Save detailed results
        detailed_file, detailed_df = save_results_to_csv(
            all_results, dataset_name, target_dim, b_percentage, alpha, k_values, output_dir
        )
        
        # Save summary report
        summary_file, summary_df = save_summary_report(
            all_results, dataset_name, target_dim, b_percentage, alpha, k_values, output_dir
        )
        
        print(f"[SAVE] Detailed results: {detailed_file}")
        print(f"[SAVE] Summary report: {summary_file}")
        
        return all_results, detailed_file, summary_file
    else:
        return all_results

if __name__ == "__main__":
    # Example usage
    dataset_name = "Fasttext"
    train_file = "training_vectors_01pct_Fasttext.npy"
    test_file = "testing_vectors_01pct_Fasttext.npy"
    target_dim = 128
    b_percentage = 1.0
    alpha = 0.1
    k_values = [1, 3, 6, 10, 15]
    
    # Run evaluation with result saving
    results, detailed_file, summary_file = main_evaluation(
        dataset_name, train_file, test_file, target_dim, b_percentage, alpha, k_values,
        save_results=True, output_dir="Result"
    )
    
    # Print summary results
    print("\n=== Summary Results ===")
    for method_name, method_results in results.items():
        dr_time = method_results.get('dr_time', np.nan)
        dr_memory = method_results.get('dr_memory', np.nan)
        print(f"\n{method_name}:")
        print(f"  DR Time: {dr_time:.4f}s")
        print(f"  DR Memory: {dr_memory:.2f}MB")
        
        for index_name, index_results in method_results.items():
            if index_name in ['dr_time', 'dr_memory', 'gt_time', 'gt_memory']:
                continue
            print(f"  {index_name}:")
            for k in k_values:
                if k in index_results:
                    recall = index_results[k].get('recall', np.nan)
                    search_time = index_results[k].get('time', np.nan)
                    search_memory = index_results[k].get('memory', np.nan)
                    print(f"    k={k}: Recall={recall:.4f}, Time={search_time:.4f}s, Memory={search_memory:.2f}MB")
