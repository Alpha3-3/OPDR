import numpy as np
import pyopencl as cl
from tqdm import tqdm
import os

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

def reduce_dimensions_top_opencl(vectors, num_groups):
    """
    Reduce dimensions by grouping based on average absolute differences using OpenCL.

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

    # Select the top dimensions with the largest average absolute differences
    top_indices = np.argsort(-avg_abs_diffs)[:num_groups]

    # Extract the corresponding vectors
    reduced_vectors = vectors[:, top_indices]

    return reduced_vectors, top_indices

def apply_top_selection_to_testing_set(test_vectors, top_indices):
    """
    Apply the top dimension selection to the testing set.

    Args:
        test_vectors (np.ndarray): Original testing vectors.
        top_indices (list): Indices of the top dimensions.

    Returns:
        np.ndarray: Reduced testing vectors containing only the selected dimensions.
    """
    return test_vectors[:, top_indices]

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

        # Dimension reduction using OpenCL (top dimensions)
        print("\nReducing dimensions of training set by selecting top dimensions (OpenCL):")
        reduced_training_vectors, top_indices = reduce_dimensions_top_opencl(training_vectors, num_groups)

        # Save reduced training vectors
        np.save("10000reduced_vectors_topAbsDiffGPUAccTop.npy", reduced_training_vectors)
        print("Reduced training vectors saved to 10000reduced_vectors_topAbsDiffGPUAccTop.npy")

        # Load testing vectors
        print("\nLoading testing vectors:")
        test_vectors = np.load(testing_file)
        print(f"Testing vectors shape: {test_vectors.shape}")

        # Reduce dimensions of testing set using the same top indices
        print("\nReducing dimensions of testing set by selecting top dimensions:")
        reduced_testing_vectors = apply_top_selection_to_testing_set(test_vectors, top_indices)

        # Save reduced testing vectors
        np.save("100testing_reduced_vectors_topAbsDiffGPUAccTop.npy", reduced_testing_vectors)
        print("Reduced testing vectors saved to 100testing_reduced_vectors_topAbsDiffGPUAccTop.npy")

