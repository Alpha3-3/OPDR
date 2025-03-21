import numpy as np
import pyopencl as cl
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

def load_vectors(fname):
    """
    Load word vectors from a file.

    Args:
        fname (str): Path to the file containing word vectors.

    Returns:
        dict: A dictionary where keys are words and values are vectors (as numpy arrays).
    """
    try:
        with open(fname, 'r', encoding='utf-8', errors='ignore') as fin:
            n, d = map(int, fin.readline().split())
            print(f"Number of vectors: {n}, Dimension of each vector: {d}")

            data = {}
            for line in tqdm(fin, total=n, desc="Loading vectors", unit="vec"):
                tokens = line.rstrip().split(' ')
                word, vector = tokens[0], np.array(tokens[1:], dtype=np.float32)
                data[word] = vector
        return data
    except FileNotFoundError:
        print(f"File not found: {fname}")
        return {}
    except Exception as e:
        print(f"Error loading vectors: {e}")
        return {}

def reduce_dimensions_by_grouping_opencl(vectors, num_groups):
    """
    Reduce dimensions by grouping based on average absolute differences using OpenCL and PCA.

    Args:
        vectors (np.ndarray): Array of shape (m, n), where m is the number of vectors and n is the number of dimensions.
        num_groups (int): Desired number of groups (new dimensions).

    Returns:
        np.ndarray: Reduced vectors.
        list: List of groups (indices of original dimensions).
    """
    m, n = vectors.shape
    avg_abs_diffs = np.zeros(n, dtype=np.float32)  # To store the average absolute differences for each dimension

    # OpenCL setup
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # Kernel for computing average absolute differences for each dimension
    kernel_code = """
    __kernel void calc_avg_abs_diff(__global const float *vectors, __global float *avg_abs_diffs, int m, int n) {
        int dim = get_global_id(0);  // Current dimension

        if (dim >= n) return;

        float sum_diff = 0.0f;

        // Compute mean of the current dimension
        float mean_dim = 0.0f;
        for (int i = 0; i < m; ++i) {
            mean_dim += vectors[i * n + dim];
        }
        mean_dim /= m;

        // Compute sum of absolute differences from the mean
        for (int i = 0; i < m; ++i) {
            float diff = fabs(vectors[i * n + dim] - mean_dim);
            sum_diff += diff;
        }

        avg_abs_diffs[dim] = sum_diff / m;
    }
    """
    program = cl.Program(context, kernel_code).build()

    # Create buffers
    mf = cl.mem_flags
    vectors_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vectors)
    avg_abs_diffs_buf = cl.Buffer(context, mf.WRITE_ONLY, n * 4)  # Buffer for all dimensions (float32)

    # Execute the kernel to compute average absolute differences
    global_size = (n,)  # Process all dimensions
    program.calc_avg_abs_diff(queue, global_size, None, vectors_buf, avg_abs_diffs_buf, np.int32(m), np.int32(n))

    # Retrieve the average absolute differences
    cl.enqueue_copy(queue, avg_abs_diffs, avg_abs_diffs_buf)
    queue.finish()

    # Sort dimensions by average absolute differences in descending order
    sorted_indices = np.argsort(-avg_abs_diffs)
    sorted_averages = avg_abs_diffs[sorted_indices]

    # Initialize groups and their sum of averages
    groups = [[] for _ in range(num_groups)]
    group_sums = np.zeros(num_groups)

    # Distribute dimensions into groups to balance the sum of averages
    for idx, avg in zip(sorted_indices, sorted_averages):
        # Find the group with the smallest total sum
        min_group = np.argmin(group_sums)
        groups[min_group].append(idx)
        group_sums[min_group] += avg**2

    # Debug: Check for empty groups
    print(f"Groups formed: {[len(group) for group in groups]}")

    # Create reduced vectors based on groups using t-SNE then rescale
    reduced_vectors = np.zeros((m, num_groups), dtype=np.float32)
    for i, group in enumerate(groups):
        if len(group) == 0:  # Handle empty group
            print(f"Warning: Group {i} is empty. Skipping.")
            continue

        # Extract the dimensions corresponding to the current group
        group_vectors = vectors[:, group]  # Shape: (m, len(group))

        # Use t-SNE to reduce the group to 1 dimension
        if group_vectors.shape[1] > 1:  # Apply t-SNE only if group has more than one dimension
            tsne = TSNE(n_components=1, random_state=42)
            reduced_group = tsne.fit_transform(group_vectors).flatten()  # Shape: (m,)
        else:
            reduced_group = group_vectors.flatten()  # If the group has only one dimension, use it directly

        # Store the reduced group as the i-th new dimension
        reduced_vectors[:, i] = reduced_group

    return reduced_vectors, groups
def apply_grouping_to_testing_set(test_vectors, groups):
    """
    Apply the grouping derived from the training set to the testing set.

    Args:
        test_vectors (np.ndarray): Original testing vectors.
        groups (list): Grouping of dimensions derived from the training set.

    Returns:
        np.ndarray: Transformed testing vectors.
    """
    m = test_vectors.shape[0]
    num_groups = len(groups)
    reduced_test_vectors = np.zeros((m, num_groups), dtype=np.float32)

    for i, group in enumerate(groups):
        reduced_test_vectors[:, i] = np.sum(test_vectors[:, group], axis=1)

    return reduced_test_vectors

if __name__ == "__main__":
    # File paths
    training_file = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'
    testing_file = 'testing_set_vectors.npy'

    # Load training vectors
    print("\nLoading training vectors:")
    training_vectors_dict = load_vectors(training_file)

    if training_vectors_dict:
        words = list(training_vectors_dict.keys())
        training_vectors = np.array(list(training_vectors_dict.values()))
        m, n = training_vectors.shape

        # Specify the number of groups (new dimensions)
        num_groups = 150  # Adjust as needed

        # Dimension reduction using OpenCL
        print("\nReducing dimensions of training set using grouping (OpenCL):")
        reduced_training_vectors, groups = reduce_dimensions_by_grouping_opencl(training_vectors, num_groups)

        # Save reduced training vectors
        np.save("10000reduced_vectors_groupByAbsDiffGPUAccPCA.npy", reduced_training_vectors)
        print("Reduced training vectors saved to 10000reduced_vectors_groupByAbsDiffGPUAccPCA.npy")

        # Load testing vectors
        print("\nLoading testing vectors:")
        test_vectors = np.load(testing_file)
        print(f"Testing vectors shape: {test_vectors.shape}")

        # Apply the same grouping to the testing set
        print("\nReducing dimensions of testing set using the same grouping:")
        reduced_testing_vectors = apply_grouping_to_testing_set(test_vectors, groups)

        # Save reduced testing vectors
        np.save("100testing_reduced_vectors_groupByAbsDiffGPUAccPCA.npy", reduced_testing_vectors)
        print("Reduced testing vectors saved to 100testing_reduced_vectors_groupByAbsDiffGPUAccPCA.npy")
        print("finished")