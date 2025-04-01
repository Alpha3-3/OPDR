import numpy as np
import pyopencl as cl
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

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

def reduce_dimensions_by_grouping_opencl(vectors, num_groups, batch_size=10, percentage=1.0):
    m, n = vectors.shape
    avg_abs_diffs = np.zeros(n, dtype=np.float32)

    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    kernel_code = """
    __kernel void calc_top_pct_pairwise_abs_diff(__global const float *vectors, 
                                                 __global float *avg_abs_diffs, 
                                                 __global float *diffs_buffer,  
                                                 int m, int n, 
                                                 float percentage) {
        int dim = get_global_id(0);

        if (dim >= n) return;

        int max_pairs = (m * (m - 1)) / 2;
        int offset = dim * max_pairs;
        int index = 0;

        for (int i = 0; i < m; ++i) {
            for (int j = i + 1; j < m; ++j) {
                diffs_buffer[offset + index++] = fabs(vectors[i * n + dim] - vectors[j * n + dim]);
            }
        }

        float sum_diffs = 0.0f;
        for (int i = 0; i < index; ++i) {
            sum_diffs += diffs_buffer[offset + i];
        }

        avg_abs_diffs[dim] = sum_diffs / index;
    }
    """

    try:
        program = cl.Program(context, kernel_code).build()
    except Exception as e:
        print(f"Kernel build failed: {e}")
        return None, None

    mf = cl.mem_flags
    vectors_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vectors)

    max_pairs = (m * (m - 1)) // 2
    groups = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        current_batch_size = batch_end - batch_start

        avg_abs_diffs_buf = cl.Buffer(context, mf.WRITE_ONLY, current_batch_size * 4)
        diffs_buffer = cl.Buffer(context, mf.READ_WRITE, size=current_batch_size * max_pairs * 4)

        global_size = (current_batch_size,)

        program.calc_top_pct_pairwise_abs_diff(queue, global_size, None,
                                               vectors_buf, avg_abs_diffs_buf,
                                               diffs_buffer,
                                               np.int32(m), np.int32(n), np.float32(percentage))

        avg_abs_diffs_batch = np.zeros(current_batch_size, dtype=np.float32)
        cl.enqueue_copy(queue, avg_abs_diffs_batch, avg_abs_diffs_buf)
        queue.finish()

        avg_abs_diffs[batch_start:batch_end] = avg_abs_diffs_batch

    return avg_abs_diffs, groups

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
        m, n = training_vectors.shape

        num_groups = 150
        batch_size = 10
        percentage = 10.0

        print(f"\nReducing dimensions of training set using grouping (OpenCL):")
        reduced_training_vectors, groups = reduce_dimensions_by_grouping_opencl(training_vectors, num_groups, batch_size, percentage)

        if reduced_training_vectors is not None:
            np.save("10000reduced_vectors_groupByLocalAbsDiffGPUAcc.npy", reduced_training_vectors)
            print("Reduced training vectors saved.")

            print("\nLoading testing vectors:")
            test_vectors = np.load(testing_file)
            print(f"Testing vectors shape: {test_vectors.shape}")

            print("\nReducing dimensions of testing set:")
            reduced_testing_vectors = apply_grouping_to_testing_set(test_vectors, groups)
            np.save("100testing_reduced_vectors_groupByLocalAbsDiffGPUAcc.npy", reduced_testing_vectors)
            print("Reduced testing vectors saved.")
