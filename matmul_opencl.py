import pyopencl as cl
import numpy as np
import datetime


# Test Matrices
matrix1 = np.ones([5, 5], np.float32)
matrix2 = np.ones([5, 1], np.float32)

ctx = cl.create_some_context()   # create context
queue = cl.CommandQueue(ctx)    # create queue


# Define the kernel and build it
prg = cl.Program(ctx, """
        // First naive implementation
        __kernel void multiplymatrices(const unsigned int M, const unsigned int N, const unsigned int K,
                              const __global float* A,
                              const __global float* B,
                              __global float* C) {

            // Thread identifiers
            const int globalRow = get_global_id(0); // Row ID of C (0..M)
            const int globalCol = get_global_id(1); // Col ID of C (0..N)

            // Compute a single element (loop over K)
            float acc = 0.0f;
            for (int k=0; k<K; k++) {
                //acc += A[k*M + globalRow] * B[globalCol*K + k];
                acc += A[globalRow*K + k] * B[globalCol+ k*N];
            }

            // Store the result
            C[globalCol + globalRow*N] = acc;
        }
    """).build()


# Convert into python function
# Doesnt delete objects
# Cannot make too many calls
# Correction Necessary
def mat_mul(mat_A, mat_B):
    # Create buffers to store variables
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mat_A) # input buffer
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mat_B) # input buffer
    n1 = len(mat_A)
    n2 = len(mat_B.T)
    mat_C = np.zeros([n1, n2], np.float32)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, mat_C.nbytes ) # Output Buffer

    # Deploy the kernel
    prg.multiplymatrices(queue, mat_C.shape, None,np.int32(n1), np.int32(n2),
                         np.int32(len(mat_A.T)) ,a_buf, b_buf, dest_buf)
    # Copy from the destination buffer
    cl.enqueue_copy(queue, mat_C , dest_buf)
    return mat_C

if __name__ == "main":
    t0 = datetime.datetime.now()
    final_matrix = mat_mul(matrix1, matrix2)
    print(final_matrix)
    
    delta_t = datetime.datetime.now() - t0
    print('OpenCL Multiplication: ' + str(delta_t))