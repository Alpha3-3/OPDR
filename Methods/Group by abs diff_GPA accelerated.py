import numpy as np
import pyopencl as cl
from tqdm import tqdm
import os

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
def load_vectors(fname):
    # Same as your original function
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
    m, n = vectors.shape
    avg_abs_diffs = np.zeros(n, dtype=np.float32)  # To store the filtered averages for each dimension

    # OpenCL setup
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # Kernel for absolute differences
    kernel_code = """
    __kernel void calc_abs_diff(__global const float *vectors, __global float *abs_diffs, int m, int n) {
        int dim = get_global_id(0);  // Current dimension
        int i = get_global_id(1);    // Current row (vector index)

        if (dim >= n || i >= m) return;

        // Compute absolute differences
        for (int j = 0; j < m; ++j) {
            if (i != j) {
                float diff = fabs(vectors[i * n + dim] - vectors[j * n + dim]);
                abs_diffs[i * m + j] = diff;  // Store pairwise differences for this dimension
            }
        }
    }
    """
    program = cl.Program(context, kernel_code).build()

    # Create buffers
    mf = cl.mem_flags
    vectors_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vectors)
    abs_diffs_buf = cl.Buffer(context, mf.WRITE_ONLY, m * m * 4)  # Buffer for one dimension (float32)

    # Add a progress bar for tracking dimensions
    for dim in tqdm(range(n), desc="Processing dimensions", unit="dim"):
        # Execute the kernel for one dimension
        global_size = (1, m)  # Process one dimension across all rows
        program.calc_abs_diff(queue, global_size, None, vectors_buf, abs_diffs_buf, np.int32(m), np.int32(n))

        # Retrieve results for the current dimension
        abs_diffs = np.zeros((m, m), dtype=np.float32)
        cl.enqueue_copy(queue, abs_diffs, abs_diffs_buf)
        queue.finish()

        # Flatten and filter differences
        differences = abs_diffs.flatten()
        relative_diffs = differences / max(np.abs(vectors[:, dim]).mean(), 1e-8)  # Avoid divide-by-zero
        relative_diffs = np.sort(relative_diffs)

        # Remove top 1% max and min
        lower_bound = int(len(relative_diffs) * 0.01)
        upper_bound = int(len(relative_diffs) * 0.99)
        filtered_diffs = relative_diffs[lower_bound:upper_bound]

        # Compute the average
        avg_abs_diffs[dim] = np.mean(filtered_diffs)

    print("Average absolute differences after filtering (debug):", avg_abs_diffs)

    # Sort dimensions by average absolute differences
    sorted_indices = np.argsort(-avg_abs_diffs)
    sorted_averages = avg_abs_diffs[sorted_indices]
    print("Sorted averages (debug):", sorted_averages)

    # Initialize groups
    groups = [[] for _ in range(num_groups)]
    group_sums = np.zeros(num_groups)

    # Greedy grouping to balance square sums
    for avg, index in zip(sorted_averages, sorted_indices):
        min_group = np.argmin(group_sums)
        groups[min_group].append(index)
        group_sums[min_group] += avg**2

    # Calculate the new dimension values as square root of square sums
    new_dimension_values = np.sqrt(group_sums)

    # Create reduced vectors based on groups using square sum
    reduced_vectors = np.zeros((m, num_groups))
    for i, group in enumerate(groups):
        for dim in group:
            reduced_vectors[:, i] += vectors[:, dim] **2
        if group_sums[i] > 0:
            reduced_vectors[:, i] *= new_dimension_values[i] / np.sqrt(group_sums[i])
        else:
            print(f"Warning: Group {i} has a zero sum. Skipping adjustment.")


    return reduced_vectors, groups, new_dimension_values

if __name__ == "__main__":
    # File path for word vectors
    filename = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'

    # Load word vectors
    print("\nLoading word vectors:")
    vectors = load_vectors(filename)

    if vectors:
        vector_array = np.array(list(vectors.values()))
        words = list(vectors.keys())

        # Specify the number of groups (new dimensions)
        num_groups = 30  # Modify this as needed

        # Dimension reduction using OpenCL
        print("\nReducing dimensions using grouping (OpenCL):")
        reduced_vectors, groups, new_dimension_values = reduce_dimensions_by_grouping_opencl(vector_array, num_groups)

        print(f"New dimension values: {new_dimension_values}")
        print(f"Groups (indices of original dimensions): {groups}")

        # Save reduced vectors for later use
        np.save("10000reduced_vectors_groupByAbsDiffGPUAcc.npy", reduced_vectors)
        np.save("../words.npy", np.array(words))
        print("\nReduction completed. Reduced vectors saved to files.")
