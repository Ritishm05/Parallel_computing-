#include <stdio.h>

#define N 1024

__global__ void sum(int *out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        int sum = (N * (N + 1)) / 2;
        *out = sum;
    }
}

int main() {
    int *h_out;
    int *d_out;

    h_out = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&d_out, sizeof(int));
    sum<<<1,32>>>(d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum of first %d integers using formula: %d\n", N, *h_out);

    free(h_out);
    cudaFree(d_out);

    return 0;
}
