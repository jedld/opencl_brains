 __kernel void sigmoid(uint col_size, uint row_size, __global const float *A, __global float *C) {

    // Get the index of the current element to be processed
    int i = get_global_id(0) / row_size;
    int j = get_global_id(0) % col_size;
    
    C[i * col_size + j] = 1f / ( 1f + exp(-A[i * col_size + j])
}