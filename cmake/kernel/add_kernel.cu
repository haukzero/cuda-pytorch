#include "my_add.h"

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

void add2(float* c, const float* a, const float* b, int n) {
    int block_size = 1024;
    int grid_size = (n + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size>>>(a, b, c, n);
}
