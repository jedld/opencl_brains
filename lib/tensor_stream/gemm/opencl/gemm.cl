 __kernel void matrix(uint col_size, uint row_size, __global const float *A, __global const float *B, __global float *C) {

    // Get the index of the current element to be processed
    int i = get_global_id(0) / row_size;
    int j = get_global_id(0) % col_size;
    
        
    int value = 0;
    for(int k = 0; k < col_size; k++)
    {
        value += A[i * col_size + k] *  B[k * col_size + j];
    }
    C[i * col_size + j] = value;
}