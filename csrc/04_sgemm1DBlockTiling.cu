#include <ATen/ATen.h>
#include <iostream>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TH>
__global__ void sgemm1DBlockTiling(const float *__restrict__ A, const float *__restrict__ B, float *C, int M, int N, int K)
{
    __shared__ float AS[BM * BK];
    __shared__ float BS[BK * BN];
        
    // move the mat to start of the each block
    // BM -> TH * blockDim.y BN -> blockDim.x
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    
    uint threadRow = threadIdx.x / BN;
    uint threadCol = threadIdx.x % BN;

    uint innerRowA = threadIdx.x / BK;
    uint innerColA = threadIdx.x % BK;

    float tmp[TH] = {0.0f};
    for (uint blkIdx{0}; blkIdx < K; blkIdx += BK)
    {
        // Global to shared mem
        // each block will have 512 threads
        // each thread move single element from A and B
        // shape A->64*8 B->8*64
        AS[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        BS[threadRow * BN + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();

        // MOVE the mat pointer to next block
        A += BK;
        B += BK * N;

        // do the dot product
        for (uint dotIdx{0}; dotIdx < BK; ++dotIdx)
        {
            float tmpB = BS[dotIdx * BN + threadCol];
            for (uint workIdx{0}; workIdx < TH; ++workIdx)
            {
                tmp[workIdx] += AS[(threadRow * TH + workIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }
    // write back the results
    for (uint resIdx = 0; resIdx < TH; ++resIdx)
    {
        C[(threadRow * TH + resIdx) * N + threadCol] = tmp[resIdx];
    }
    
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
    
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 block((BM * BN) / TM);
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    
    cudaMemset(d_c, 0, c.element_size() * c.numel());
    sgemm1DBlockTiling<BM, BN, BK, TM><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
}