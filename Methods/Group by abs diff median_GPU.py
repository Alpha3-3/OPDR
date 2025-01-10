import numpy as np
import pyopencl as cl
from tqdm import tqdm
import os

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

def load_vectors(fname):
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
    median_abs_diffs = np.zeros(n, dtype=np.float32)

    # OpenCL setup
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    kernel_code = """
    __kernel void calc_abs_diff(__global const float *vectors, __global float *abs_diffs, int m, int n) {
        int dim = get_global_id(0);  
        int i = get_global_id(1);

        if (dim >= n || i >= m) return;

        for (int j = 0; j < m; ++j) {
            if (i != j) {
                float diff = fabs(vectors[i * n + dim] - vectors[j * n + dim]);
                abs_diffs[i * m + j] = diff;
            }
        }
    }
    """
    program = cl.Program(context, kernel_code).build()

    mf = cl.mem_flags
    vectors_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vectors)
    abs_diffs_buf = cl.Buffer(context, mf.WRITE_ONLY, m * m * 4)

    for dim in tqdm(range(n), desc="Processing dimensions", unit="dim"):
        global_size = (1, m)
        program.calc_abs_diff(queue, global_size, None, vectors_buf, abs_diffs_buf, np.int32(m), np.int32(n))

        abs_diffs = np.zeros((m, m), dtype=np.float32)
        cl.enqueue_copy(queue, abs_diffs, abs_diffs_buf)
        queue.finish()

        differences = abs_diffs.flatten()
        relative_diffs = differences / max(np.abs(vectors[:, dim]).mean(), 1e-8)
        relative_diffs = np.sort(relative_diffs)

        lower_bound = int(len(relative_diffs) * 0.01)
        upper_bound = int(len(relative_diffs) * 0.99)
        filtered_diffs = relative_diffs[lower_bound:upper_bound]

        median_abs_diffs[dim] = np.median(filtered_diffs)

    sorted_indices = np.argsort(-median_abs_diffs)
    sorted_medians = median_abs_diffs[sorted_indices]

    groups = [[] for _ in range(num_groups)]
    group_sums = np.zeros(num_groups)

    for med, idx in zip(sorted_medians, sorted_indices):
        min_group = np.argmin(group_sums)
        groups[min_group].append(idx)
        group_sums[min_group] += med**2

    # Create reduced vectors based on groups
    reduced_vectors = np.zeros((m, num_groups), dtype=np.float32)
    for i, group in enumerate(groups):
        # Sum the selected dimensions
        reduced_vectors[:, i] = np.sum(vectors[:, group] , axis=1)

    return reduced_vectors, groups

def apply_grouping_to_testing_set(test_vectors, groups):
    m = test_vectors.shape[0]
    num_groups = len(groups)
    reduced_test_vectors = np.zeros((m, num_groups), dtype=np.float32)

    for i, group in enumerate(groups):
        reduced_test_vectors[:, i] = np.sum(test_vectors[:, group], axis=1)

    return reduced_test_vectors

if __name__ == "__main__":
    training_file = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'
    testing_file = 'testing_set_vectors.npy'

    print("\nLoading training vectors:")
    training_vectors_dict = load_vectors(training_file)

    if training_vectors_dict:
        words = list(training_vectors_dict.keys())
        training_vectors = np.array(list(training_vectors_dict.values()))

        num_groups = 150

        print("\nReducing dimensions of training set:")
        reduced_training_vectors, groups = reduce_dimensions_by_grouping_opencl(training_vectors, num_groups)

        np.save("10000reduced_vectors_groupByAbsDiffGPUAccMedian.npy", reduced_training_vectors)

        print("\nLoading testing vectors:")
        test_vectors = np.load(testing_file)

        print("\nReducing dimensions of testing set:")
        reduced_testing_vectors = apply_grouping_to_testing_set(test_vectors, groups)

        np.save("100testing_reduced_vectors_groupByAbsDiffGPUAccMedian.npy", reduced_testing_vectors)
