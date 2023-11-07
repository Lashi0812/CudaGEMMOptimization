#include <cute/tensor.hpp>
#include <ATen/ATen.h>
#include <iostream>

using namespace cute;

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

template <class ALayout, class BLayout, class CLayout>
__global__ void naive_gemm_inner_using_cute(const float *__restrict__ a, const float *__restrict__ b, float *c,
                                            ALayout la, BLayout lb, CLayout lc)
{
    auto gA = make_tensor(make_gmem_ptr(a), la);
    auto gB = make_tensor(make_gmem_ptr(b), lb);
    // auto gC = make_tensor(make_gmem_ptr(c), lc);

    auto threadBlock = make_tile(make_layout(blockDim.y), make_layout(blockDim.x));

    // auto tCgC = zipped_divide(gC, threadBlock)(make_coord(threadIdx.y, threadIdx.x), make_coord(blockIdx.y, blockIdx.x));
    auto tCgC_coord = zipped_divide(lc, threadBlock)(make_coord(threadIdx.y, threadIdx.x), make_coord(blockIdx.y, blockIdx.x));
    auto tAgA = gA(_, get<0>(gA.get_flat_coord(tCgC_coord)));
    auto tBGB = gB(get<1>(gB.get_flat_coord(tCgC_coord)), _);

    // print("%d \n", tCgC_coord);
    // print_tensor(tAgA);
    // print_tensor(tBGB);

    float regC{0};
    for (uint k{0}; k < size<1>(gA); ++k)
    {
        regC += tAgA(k) * tBGB(k);
    }
    c[lc(make_coord(get<1>(gB.get_flat_coord(tCgC_coord)),get<0>(gB.get_flat_coord(tCgC_coord))))] = regC;
}

void host_naive_gemm_inner(at::Tensor const &a, at::Tensor const &b, at::Tensor &c,
                           float *d_a, float *d_b, float *d_c,
                           int M, int N, int K)
{
    dim3 block(64, 4);
    dim3 grid((a.sizes()[0] + block.x - 1) / block.x, (a.sizes()[1] + block.y - 1) / block.y);

    c.zero_();
    cudaMemset(d_c, 0, c.element_size() * c.numel());
    naive_gemm_inner<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    std::cout << (c.allclose(a.matmul(b)) ? "Inner Success" : "Inner Failed") << std::endl;
    // std::cout << a << std::endl;
    // std::cout << b << std::endl;
    // std::cout << c << std::endl;
}

void host_naive_gemm_inner_using_cute(at::Tensor &a, at::Tensor &b, at::Tensor &c,
                                      float *d_a, float *d_b, float *d_c,
                                      int m, int n, int k)
{
    auto M = int(m);
    auto K = int(n);
    auto N = int(k);

    dim3 block(64, 4);
    dim3 grid((a.sizes()[0] + block.x - 1) / block.x, (a.sizes()[1] + block.y - 1) / block.y);

    auto lA = make_layout(make_shape(M, K));
    auto lB = make_layout(make_shape(N, K));
    auto lC = make_layout(make_shape(M, N));
    
    c.zero_();
    cudaMemset(d_c, 0, c.element_size() * c.numel());
    naive_gemm_inner_using_cute<<<grid, block>>>(d_a, d_b, d_c, lA, lB, lC);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    // std::cout << c << std::endl;
    std::cout << (c.allclose(a.matmul(b)) ? "Cute Inner Success" : "Cute Inner Failed") << std::endl;
}

int main()
{
    auto M = 4096;
    auto K = 4096;
    auto N = 4096;

    // auto a = at::arange(M * K, at::TensorOptions().dtype(at::kFloat)).reshape({M, K});
    // auto b = at::arange(K * N, at::TensorOptions().dtype(at::kFloat)).reshape({K, N});

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

    host_naive_gemm_inner(a, b, c, d_a, d_b, d_c, M, N, K);

    host_naive_gemm_inner_using_cute(a, b, c, d_a, d_b, d_c, M, N, K);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
}
