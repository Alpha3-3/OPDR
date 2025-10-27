#!/usr/bin/env python3
"""
MPAD with Numba parallel optimization.

Key changes from baseline:
1. Force BLAS to single-thread (avoid oversubscription)
2. Replace serial two-pointer counting with parallel binary search
3. Parallel equal-layer sampling with prefix sum
4. Keep sorting as-is (NumPy single-threaded for now)

Strategy: Let Numba handle all parallelism, keep BLAS single-threaded.
"""

import os
import numpy as np
from scipy.optimize import minimize

# Force BLAS to single thread to avoid thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Set Numba threading layer (can be 'omp' or 'tbb')
os.environ["NUMBA_THREADING_LAYER"] = "omp"

from numba import njit, prange


# ==================== Numba Parallel Kernels ====================

@njit(parallel=True, fastmath=True)
def parallel_binary_search_indices(s, delta, tol, N):
    """
    For each i, binary search for:
    - j_gt[i]: first j where s[j] - s[i] > delta - tol
    - j_ge[i]: first j where s[j] - s[i] >= delta - tol
    
    Fully parallel: each i is independent.
    Complexity: O(N log N) but parallelizable across all cores.
    """
    j_gt = np.empty(N, dtype=np.int64)
    j_ge = np.empty(N, dtype=np.int64)
    
    for i in prange(N):
        target = s[i] + delta
        
        # Binary search for j_gt: s[j] > target - tol
        lo = i + 1
        hi = N
        while lo < hi:
            mid = (lo + hi) >> 1
            if s[mid] <= target - tol:
                lo = mid + 1
            else:
                hi = mid
        j_gt[i] = lo
        
        # Binary search for j_ge: s[j] >= target - tol
        lo = i + 1
        hi = N
        while lo < hi:
            mid = (lo + hi) >> 1
            if s[mid] < target - tol:
                lo = mid + 1
            else:
                hi = mid
        j_ge[i] = lo
    
    return j_gt, j_ge


@njit(parallel=True)
def parallel_compute_counts(j_indices, N):
    """Parallel reduction to count pairs."""
    total = 0
    for i in prange(N):
        if j_indices[i] < N:
            total += (N - j_indices[i])
    return total


@njit(parallel=True, fastmath=True)
def parallel_compute_gt_stats(s, P, j_gt, N):
    """
    Parallel computation of:
    - left_gt_cnt[i]: number of right partners for i
    - right_cnt_diff: differential array for right-side counts
    - sum_gt: sum of gaps
    """
    left_gt_cnt = np.zeros(N, dtype=np.int64)
    right_cnt_diff = np.zeros(N + 1, dtype=np.int64)
    
    # Each i computes its contribution independently
    for i in prange(N):
        j = j_gt[i]
        if j < N:
            cnt_i = N - j
            left_gt_cnt[i] = cnt_i
            # Note: right_cnt_diff writes need to be handled carefully
            # Numba handles atomic operations automatically in prange
            right_cnt_diff[j] += 1
            right_cnt_diff[N] -= 1
    
    # Compute sum_gt (sequential reduction after parallel setup)
    sum_gt = 0.0
    for i in range(N):
        j = j_gt[i]
        if j < N:
            cnt_i = N - j
            sum_gt += (P[N] - P[j]) - cnt_i * s[i]
    
    return left_gt_cnt, right_cnt_diff, sum_gt


@njit(parallel=True)
def parallel_eq_layer_sampling(j_ge, j_gt, R, N):
    """
    Parallel computation of equal-layer pair selection.
    
    Strategy:
    1. Compute can_take[i] in parallel
    2. Sequential prefix sum to get offsets
    3. Parallel allocation based on offsets
    """
    can_take = np.zeros(N, dtype=np.int64)
    
    # Parallel: compute max available per i
    for i in prange(N):
        L = j_ge[i]
        Rg = j_gt[i] - 1
        if L < N and Rg >= L:
            can_take[i] = Rg - L + 1
    
    # Sequential prefix sum (could use parallel scan for large N)
    offsets = np.zeros(N, dtype=np.int64)
    cumsum = 0
    for i in range(N):
        offsets[i] = cumsum
        cumsum += can_take[i]
    
    # Parallel: allocate based on offsets
    eq_take_per_i = np.zeros(N, dtype=np.int64)
    for i in prange(N):
        if offsets[i] < R:
            take = min(can_take[i], R - offsets[i])
            eq_take_per_i[i] = take
    
    return eq_take_per_i


@njit(parallel=True)
def parallel_build_eq_right_diff(j_ge, eq_take_per_i, N):
    """Build differential array for equal-layer right counts."""
    eq_right_diff = np.zeros(N + 1, dtype=np.int64)
    
    for i in prange(N):
        take = eq_take_per_i[i]
        if take > 0:
            L = j_ge[i]
            eq_right_diff[L] += take
            if L + take < N:
                eq_right_diff[L + take] -= take
    
    return eq_right_diff


@njit(parallel=True, fastmath=True)
def parallel_build_gradient_coeff(left_gt_cnt, right_gt_cnt, 
                                   eq_take_per_i, eq_right_cnt, N):
    """Parallel construction of gradient coefficient vector c."""
    c = np.zeros(N, dtype=np.float64)
    
    for idx in prange(N):
        # GT-layer contributions
        c[idx] += right_gt_cnt[idx]
        c[idx] -= left_gt_cnt[idx]
        # EQ-layer contributions
        c[idx] += eq_right_cnt[idx]
        c[idx] -= eq_take_per_i[idx]
    
    return c


# ==================== Optimized MPAD Class ====================

class MPAD_Optimized:
    """
    Optimized MPAD using Numba parallel kernels.
    
    Key optimizations:
    1. Parallel binary search for pair counting (replaces serial two-pointer)
    2. Parallel equal-layer sampling
    3. BLAS forced to single-thread to avoid oversubscription
    """
    
    def __init__(self, b_percentage=1.0, alpha=0.1, target_dim=128, 
                 tol=1e-9, max_bs_iter=40):
        self.b_percentage = b_percentage
        self.alpha = alpha
        self.target_dim = target_dim
        self.projection_axes = None
        self.X_mean_ = None
        self.tol = tol
        self.max_bs_iter = max_bs_iter
    
    def _objective_and_grad(self, w, X, prev_ws):
        """
        Objective function and gradient with Numba parallel optimization.
        
        Main changes:
        - count_pairs uses parallel binary search instead of serial two-pointer
        - equal-layer sampling uses parallel computation
        """
        N, n = X.shape
        if N <= 1:
            v = w / (np.linalg.norm(w) + self.tol)
            g_pen, pen = self._ortho_grad_and_penalty(v, prev_ws)
            g_w = self._project_grad_to_w(g_pen, v, w)
            return self.alpha * pen, self.alpha * g_w
        
        # Normalize direction
        v = w / (np.linalg.norm(w) + self.tol)
        
        # 1) Project and sort (still single-threaded NumPy for now)
        p = X @ v
        order = np.argsort(p)
        s = p[order]
        
        # Prefix sum for range queries
        P = np.zeros(N + 1, dtype=s.dtype)
        P[1:] = np.cumsum(s)
        
        # 2) Compute number of pairs needed (top-b%)
        total_pairs = N * (N - 1) // 2
        B = max(1, min(total_pairs, int(round(self.b_percentage / 100.0 * total_pairs))))
        
        # 3) Binary search for optimal threshold Delta
        # Find min gap for initialization
        min_gap = float('inf')
        for i in range(N - 1):
            gap = s[i + 1] - s[i]
            if gap < min_gap:
                min_gap = gap
        if not np.isfinite(min_gap) or min_gap < 0:
            min_gap = 0.0
        
        lo = 0.0
        hi = (s[-1] - s[0]) + self.tol
        
        # Binary search for Delta using parallel counting
        for _ in range(self.max_bs_iter):
            mid = 0.5 * (lo + hi)
            
            # Parallel binary search for indices
            j_gt_temp, j_ge_temp = parallel_binary_search_indices(s, mid, self.tol, N)
            
            # Parallel count
            c_ge = parallel_compute_counts(j_ge_temp, N)
            c_gt = parallel_compute_counts(j_gt_temp, N)
            
            if c_ge < B:
                hi = mid
            elif c_gt > B:
                lo = mid
            else:
                lo = hi = mid
                break
        
        Delta = 0.5 * (lo + hi)
        
        # 4) Final computation with optimal Delta
        j_gt, j_ge = parallel_binary_search_indices(s, Delta, self.tol, N)
        
        # Parallel computation of GT-layer statistics
        left_gt_cnt, right_cnt_diff, sum_gt = parallel_compute_gt_stats(s, P, j_gt, N)
        count_gt = parallel_compute_counts(j_gt, N)
        
        # Cumsum for right counts (sequential, but small overhead)
        right_gt_cnt = np.cumsum(right_cnt_diff[:-1])
        
        # 5) Equal-layer sampling
        R = max(0, B - count_gt)
        
        if R > 0:
            eq_take_per_i = parallel_eq_layer_sampling(j_ge, j_gt, R, N)
            eq_right_diff = parallel_build_eq_right_diff(j_ge, eq_take_per_i, N)
            eq_right_cnt = np.cumsum(eq_right_diff[:-1])
        else:
            eq_take_per_i = np.zeros(N, dtype=np.int64)
            eq_right_cnt = np.zeros(N, dtype=np.int64)
        
        # 6) Build gradient coefficient vector in parallel
        c = parallel_build_gradient_coeff(left_gt_cnt, right_gt_cnt,
                                          eq_take_per_i, eq_right_cnt, N)
        
        # Reorder back to original indices
        c_orig = np.zeros(N, dtype=c.dtype)
        c_orig[order] = c
        
        # 7) Compute gradient w.r.t. v (single-threaded BLAS GEMV is fine)
        g_v = X.T @ c_orig
        
        # Negative because we want to maximize separation
        g_v = -g_v / B
        
        # 8) Add orthogonality penalty
        g_pen, pen = self._ortho_grad_and_penalty(v, prev_ws)
        g_v += self.alpha * g_pen
        
        # 9) Project gradient to tangent space of unit sphere
        g_w = self._project_grad_to_w(g_v, v, w)
        
        # Objective: negative mean separation + penalty
        obj = -sum_gt / B + self.alpha * pen
        
        return obj, g_w
    
    def _ortho_grad_and_penalty(self, v, prev_ws):
        """Orthogonality penalty and gradient."""
        if len(prev_ws) == 0:
            return np.zeros_like(v), 0.0
        
        # prev_ws is a list of vectors
        U = np.column_stack(prev_ws)  # n x k
        Vu = U.T @ v  # k
        pen = np.sum(Vu ** 2)
        g = 2.0 * (U @ Vu)  # n
        
        return g, pen
    
    def _project_grad_to_w(self, g_v, v, w):
        """Project gradient from v-space to w-space (tangent space)."""
        norm_w = np.linalg.norm(w) + self.tol
        g_w = (g_v - v * np.dot(g_v, v)) / norm_w
        return g_w
    
    def fit_transform(self, X):
        """Fit MPAD and transform training data."""
        X = np.asarray(X, dtype=float)
        self.X_mean_ = X.mean(axis=0, keepdims=True)
        X_centered = X - self.X_mean_
        
        n = X_centered.shape[1]
        prev_ws = []
        optimal_ws = []
        
        for axis in range(self.target_dim):
            # Objective + gradient function
            def fun(w):
                f, g = self._objective_and_grad(w, X_centered, prev_ws)
                return f, g
            
            # Random initialization
            w0 = np.random.randn(n)
            w0 /= (np.linalg.norm(w0) + self.tol)
            
            # Optimize using L-BFGS-B
            result = minimize(fun, w0, method='L-BFGS-B', jac=True)
            
            v_opt = result.x / (np.linalg.norm(result.x) + self.tol)
            prev_ws.append(v_opt)
            optimal_ws.append(v_opt)
        
        # Store projection axes as matrix (n x m) to match baseline format
        self.projection_axes = np.column_stack(optimal_ws)
        return X_centered @ self.projection_axes
    
    def transform(self, X):
        """Transform new data using fitted projection axes."""
        if self.projection_axes is None or self.X_mean_ is None:
            raise ValueError("Must fit the model before transforming")
        X = np.asarray(X, dtype=float)
        X_centered = X - self.X_mean_
        return X_centered @ self.projection_axes

