 __kernel void matrix(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[globalRow*K + k] * B[k*N + globalCol];
    }
 
    // Store the result
    C[globalRow*N + globalCol] = acc;
}