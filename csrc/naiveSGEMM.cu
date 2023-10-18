#include <ATen/ATen.h>
#include <iostream>

__global__ void naive_gemm_inner(const float *__restrict__ a, const float *__restrict__ b, float *c, int M, int N, int K)
{
    uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;

    float res = 0;
#pragma unroll
    for (uint32_t i{0}; i < K; ++i)
        res += a[row * K + i] * b[i * N + col];
    c[row * N + col] = res;
}

int main()
{
    int M = 4096;
    int K = 4096;
    int N = 4096;
    auto a = at::rand({M, K}, at::TensorOptions().dtype(at::kFloat));
    auto b = at::rand({K, N}, at::TensorOptions().dtype(at::kFloat));
    auto c = at::zeros({M, N}, at::TensorOptions().dtype(at::kFloat));

    // allocate and copy to device
    auto nBytes = a.element_size() * a.numel();
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, nBytes);
    cudaMalloc((void **)&d_b, nBytes);
    cudaMalloc((void **)&d_c, c.element_size() * c.numel());

    cudaMemcpy(d_a, a.data_ptr(), nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data_ptr(), nBytes, cudaMemcpyHostToDevice);

    dim3 block(64, 4);
    dim3 grid((a.sizes()[0] + block.x - 1) / block.x, (a.sizes()[1] + block.y - 1) / block.y);

    naive_gemm_inner<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    std::cout << (c.allclose(a.matmul(b)) ? "Inner Success" : "Inner Failed") << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
}
