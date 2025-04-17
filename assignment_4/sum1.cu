#include <stdio.h>
#define N 1024

__global__ void ans(int *inp, int *out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        int ans = 0;
        for (int i = 0; i < N; i++) {
            ans += inp[i];
        }
        *out = ans;
    }
}

int main() {
    int *h_inp, *h_out;
    int *d_inp, *d_out;

    size_t size = N * sizeof(int);
    h_inp = (int *)malloc(size);
    h_out = (int *)malloc(sizeof(int));

    for (int i = 0; i < N; i++) {
        h_inp[i] = i + 1;
    }

    cudaMalloc((void **)&d_inp, size);
    cudaMalloc((void **)&d_out, sizeof(int));

    cudaMemcpy(d_inp, h_inp, size, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));
    ans<<<1, 32>>>(d_inp, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("ans of first %d integers is: %d\n", N, *h_out);

    free(h_inp);
    free(h_out);
    cudaFree(d_inp);
    cudaFree(d_out);

    return 0;
}
