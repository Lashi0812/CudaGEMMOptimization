#include <cute/tensor.hpp>
#include <ATen/ATen.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cute;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, const float *A, const float *B, float *C)
{
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    // if statement is necessary to make things work under tile quantization
    if (cRow < M && cCol < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i)
        {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = tmp;
    }
}

template <typename ALayout, typename BLayout, typename CLayout>
__global__ void sgemm_global_mem_coalesce_using_cute(const float *__restrict__ A, const float *__restrict__ B, float *C,
                                                     ALayout la, BLayout lb, CLayout lc)
{
    auto gA = make_tensor(make_gmem_ptr(A), la);
    auto gB = make_tensor(make_gmem_ptr(B), lb);

    // auto threadLayout = make_tile(make_layout(_2{}), make_layout(_2{}));
    auto threadLayout = make_tile(make_layout(_32{}), make_layout(_8{}));

    auto gC_coord = zipped_divide(lc, threadLayout)(make_coord(threadIdx.x, threadIdx.y), make_coord(blockIdx.x, blockIdx.y));
    auto tAgA = gA(_, get<1>(gA.get_flat_coord(gC_coord)));
    auto tBgB = gB(get<0>(gB.get_flat_coord(gC_coord)), _);

    float regC{0};
    for (uint k{0}; k < size<1>(gA); ++k)
    {
        regC += tAgA(k) * tBgB(k);
    }
    C[gC_coord] = regC;
}

void host_sgemm_global_mem_coalesce(at::Tensor &a, at::Tensor &b, at::Tensor &c,
                                    float *d_a, float *d_b, float *d_c,
                                    int M, int N, int K)
{
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);

    c.zero_();
    cudaMemset(d_c, 0, c.element_size() * c.numel());
    sgemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(M, N, K, d_a, d_b, d_c);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    std::cout << (c.allclose(a.matmul(b)) ? "Coalesce Success" : "Coalesce Failed") << std::endl;
}

void host_sgemm_global_mem_coalesce_using_cute(at::Tensor &a, at::Tensor &b, at::Tensor &c,
                                               float *d_a, float *d_b, float *d_c,
                                               int m, int n, int k)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    auto lc = make_layout(make_shape(M, N));
    auto la = make_layout(make_shape(M, K));
    auto lb = make_layout(make_shape(K, N));

    // dim3 block(4);
    // dim3 grid(CEIL_DIV(M, 2), CEIL_DIV(N, 2));

    dim3 block(32, 8);
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 8));

    c.zero_();
    cudaMemset(d_c, 0, c.element_size() * c.numel());
    sgemm_global_mem_coalesce_using_cute<<<grid, block>>>(d_a, d_b, d_c, la, lb, lc);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    std::cout << (c.allclose(a.matmul(b)) ? "Cute Coalesce Success" : "Cute Coalesce Failed") << std::endl;
    // std::cout << a << std::endl;
    // std::cout << b << std::endl;
    // std::cout << a.matmul(b) << std::endl;
    // std::cout << c << std::endl;
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

    host_sgemm_global_mem_coalesce(a, b, c, d_a, d_b, d_c, M, N, K);
    host_sgemm_global_mem_coalesce_using_cute(a, b, c, d_a, d_b, d_c, M, N, K);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
}