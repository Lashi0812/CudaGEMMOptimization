#include <ATen/ATen.h>
#include <iostream>

// #define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const uint BDIMX, const uint BDIMY>
__global__ void sgemm_shared_mem_block(const float *__restrict__ A, const float *__restrict__ B, float *C, int M, int N, int K)
{
    
    __shared__ float AS[BDIMY * BDIMX];
    __shared__ float BS[BDIMY * BDIMX];

    A += blockIdx.y * BDIMY * K;
    B += blockIdx.x * BDIMX;
    C += blockIdx.y * BDIMY * N + blockIdx.x * BDIMX;

    float tmp = 0.0f;
    for (uint32_t blkIdx{0}; blkIdx < K; blkIdx += BDIMX)
    {
        AS[threadIdx.y * BDIMX + threadIdx.x] = A[threadIdx.y * K + threadIdx.x];
        BS[threadIdx.y * BDIMX + threadIdx.x] = B[threadIdx.y * N + threadIdx.x];

        __syncthreads();
        A += BDIMX;
        B += BDIMX * N;

        for (uint32_t dotIdx{0}; dotIdx < BDIMX; ++dotIdx)
        {
            tmp += AS[threadIdx.y * BDIMX + dotIdx] * BS[dotIdx * BDIMX + threadIdx.x];
        }
        __syncthreads();
    }
    C[threadIdx.y * N + threadIdx.x] = tmp;
    
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
    
    
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / (block.x), (N + block.y - 1) / (block.y));

    cudaMemset(d_c, 0, c.element_size() * c.numel());
    sgemm_shared_mem_block<16, 16><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
    
}