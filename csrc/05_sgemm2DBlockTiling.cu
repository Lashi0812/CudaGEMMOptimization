#include <ATen/ATen.h>
#include <iostream>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// BM 128 BN 128 BK 8 TM 8 TN 8
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm2DBlockTiling(const float *__restrict__ A, const float *__restrict__ B, float *C, int M, int N, int K)
{
    // 1. allocate the shared memory
    // each with 128*8 = 1024 elements
    // so each block need 2kb of shared mem
    __shared__ float AS[BM * BK];
    __shared__ float BS[BK * BN];
    

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // 2. each block will calculate the BM*BN (128x128) in C mat
    // move the all mat pointer at start
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // jump for inner blocks
    const uint strideA = blockDim.x / BK;
    const uint strideB = blockDim.x / BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;

    // register to store A,B,C
    float regA[TM] = {0.0f};
    float regB[TN] = {0.0f};
    float regC[TM * TN] = {0.0f};
    

    for (uint blkIdx{0}; blkIdx < K; blkIdx += BK)
    {
        // 3. move the  block from global to shared mem
        // each has (BM*BN)/(TM*TN)  (ie 256) threads but we need to move (BM*BK) 1024 elements
        // so each thread to move (BM*BK) / (BM*BN)/(TM*TN) (ie 4 elements per threads)
        // we move by loop 1st 256 element is loaded using all thread thread in the block
        // next 256 element is loaded
        // outer loop(ie loop through all threads in block) taken care by CUDA driver
        // we need just think inner loop 0th thread -> 0,256,512,768
        // to do that we have to find how many rows we have to jump
        // A block has BK (8)cols so 1st 256 elements take 32 rows in A , so jump is 32 we call it strideA
        // B block has BN (128)cols so 1st 256 elements take 2 rows in B , so jump is 2 we call it strideB

        for (uint loadOffset{0}; loadOffset < BM; loadOffset += strideA)
            AS[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];

        for (uint loadOffset{0}; loadOffset < BK; loadOffset += strideB)
            BS[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];

        __syncthreads();

        // Advance the blocks
        A += BK;
        B += BK * N;

        // 4. loop through BK
        for (uint dotIdx{0}; dotIdx < BK; ++dotIdx)
        {
            // 5. move the data from shared to register
            // A first ROW and B 1st col
            for (uint i{0}; i < TM; ++i)
                regA[i] = AS[(threadRow * TM + i) * BK + dotIdx];
            for (uint i{0}; i < TN; ++i)
                regB[i] = BS[dotIdx * BN + threadCol * TN + i];

            // 6. do the dot product
            for (uint resM{0}; resM < TM; ++resM)
                for (uint resN{0}; resN < TN; ++resN)
                    regC[resM * TN + resN] += regA[resM] * regB[resN];
        }
        __syncthreads();
    }

    // write back the result
    for (uint row{0}; row < TM; ++row)
        for (uint col{0}; col < TN; ++col)
            C[(threadRow * TM + row) * N + threadCol * TN + col] = regC[row * TN + col];
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
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid5(CEIL_DIV(M, BM), CEIL_DIV(N, BN));

    cudaMemset(d_c, 0, c.element_size() * c.numel());
    sgemm2DBlockTiling<BM, BN, BK, TM, TN><<<grid5, block>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
}