#include <ATen/ATen.h>
#include <iostream>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void sgemmVectorize(float *__restrict__ A, float *__restrict__ B, float *C, int M, int N, int K)
{
    // 1. allocate the share mem
    __shared__ float AS[BM * BK];
    __shared__ float BS[BK * BN];

    // 2. start point for all mat pointer to this block
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    
    // indexing
    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    float regA[TM]{0.0};
    float regB[TN]{0.0};
    float regC[TM * TN]{0.0};

    // loop through K
    for (uint blkIdx{0}; blkIdx < K; blkIdx += BK)
    {
        // move the data from global to shared mem
        // use the reinterpret cast to treat 1st 8 bytes(ie 4 elements) as single unit as float4
        // so we don't need loop as we do in pervious kernel 256*8 = 2048 bytes (ie 1024 elements)
        // we are going to transpose the A block load
        float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        AS[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        AS[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        AS[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        AS[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        reinterpret_cast<float4 *>(&BS[innerRowB * BN + innerColB * 4])[0] = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];

        __syncthreads();

        // more the pointer to next block
        A += BK;
        B += BK * N;

        // loop through within each block
        for (uint dotIdx{0}; dotIdx < BK; ++dotIdx)
        {
            // move data form shared mem to register
            for (uint i{0}; i < TM; ++i)
                regA[i] = AS[dotIdx * BM + threadRow * TM + i];
            for (uint i{0}; i < TN; ++i)
                regB[i] = BS[dotIdx * BN + threadCol * TN + i];

            // do the dot  product
            for (uint regAIdx{0}; regAIdx < TM; ++regAIdx)
                for (uint regBIdx{0}; regBIdx < TN; ++regBIdx)
                    regC[regAIdx * TN + regBIdx] += regA[regAIdx] * regB[regBIdx];
        }
        __syncthreads();
    }
    // write back the results
    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1)
    {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4)
        {
            // load C vector into registers
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            // perform GEMM update in reg
            tmp.x = regC[resIdxM * TN + resIdxN];
            tmp.y = regC[resIdxM * TN + resIdxN + 1];
            tmp.z = regC[resIdxM * TN + resIdxN + 2];
            tmp.w = regC[resIdxM * TN + resIdxN + 3];
            // write back
            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
                tmp;
        }
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

    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 16;
    const uint TM = 8;
    const uint TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));

    cudaMemset(d_c, 0, c.element_size() * c.numel());
    sgemmVectorize<BM, BN, BK, TM, TN><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
}