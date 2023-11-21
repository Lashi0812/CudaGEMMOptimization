#include <cute/tensor.hpp>
#include <ATen/ATen.h>
#include <iostream>

using namespace cute;
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

template <typename ALayout, typename ABlockLayout, typename AThreadLayout,
          typename BLayout, typename BBlockLayout, typename BThreadLayout,
          typename CLayout, typename CBlockLayout, typename CThreadLayout,
          typename T>
__global__ void sgemm1DBlockTilingUsingCute(T const *__restrict__ A, T const *__restrict__ B, T *C,
                                            ALayout mA, ABlockLayout bA, AThreadLayout tA,
                                            BLayout mB, BBlockLayout bB, BThreadLayout tB,
                                            CLayout mC, CBlockLayout bC, CThreadLayout tC)
{
    __shared__ T smemA[cosize_v<ABlockLayout>];
    __shared__ T smemB[cosize_v<BBlockLayout>];

    // mat layouts in global mem
    auto tensorC = make_tensor(make_gmem_ptr(C), mC);
    auto tensorA = make_tensor(make_gmem_ptr(A), mA);
    auto tensorB = make_tensor(make_gmem_ptr(B), mB);

    // smem layouts
    auto sA_copy = make_tensor(make_smem_ptr(smemA), bA);
    auto sA_part = make_tensor(make_smem_ptr(smemA), make_layout(reverse(shape(bA)), GenRowMajor()));
    auto sB = make_tensor(make_smem_ptr(smemB), bB);

    // work done by the blocks
    auto gC = local_tile(tensorC, shape(bC), make_coord(blockIdx.x, blockIdx.y));
    auto gA = local_tile(tensorA, shape(bA), make_coord(_, blockIdx.y));
    auto gB = local_tile(tensorB, shape(bB), make_coord(blockIdx.x, _));

    // partition the block for data movement from gmem to smem by threads
    auto bAgA = local_partition(gA, bA, threadIdx.x);
    auto bAsA_copy = local_partition(sA_copy, bA, threadIdx.x);

    auto bBgB = local_partition(gB, bB, threadIdx.x);
    auto bBsB = local_partition(sB, bB, threadIdx.x);

    // partition the smem for data needed for to calculate the dot product
    auto bCgC = local_partition(gC, tC, threadIdx.x);
    auto needA = local_partition(sA_part, tC, threadIdx.x, Step<Underscore, _1>{});
    auto needB = local_partition(sB, tC, threadIdx.x, Step<_1, Underscore>{});

    // rmem layout
    auto rC = make_tensor_like(bCgC);
    auto rA = make_tensor_like(needA);
    auto rB = make_tensor_like(needB);

    auto k_tile_max = size<2>(bAgA);
    // loop over tile
    for (int k_tile{0}; k_tile < k_tile_max; ++k_tile)
    {
        copy(bAgA(_, _, k_tile), bAsA_copy);
        copy(bBgB(_, _, k_tile), bBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // loop over thread work
        // needB
        for (int bIdx{0}; bIdx < size<1>(needB); ++bIdx)
        {
            // needA
            rB(0) = needB(bIdx);
            for (int aIdx{0}; aIdx < size<0>(needA); ++aIdx)
            {
                rA(0) = needA(aIdx, bIdx);
                rC(aIdx) += rA(0) * rB(0);
                // rC(aIdx) += needA(aIdx, bIdx) * needB(bIdx);
            }
        }
        __syncthreads();
    }
    copy(rC, bCgC);
}

void host_sgemm1DBlockTiling(at::Tensor &a, at::Tensor &b, at::Tensor &c,
                             float *d_a, float *d_b, float *d_c,
                             int M, int N, int K)
{
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 block((BM * BN) / TM);
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));

    cudaMemset(d_c, 0, c.element_size() * c.numel());
    sgemm1DBlockTiling<BM, BN, BK, TM><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    std::cout << (c.allclose(a.matmul(b)) ? "SMEM 1D Success" : "SMEM 1D Failed") << std::endl;
}

void host_sgemm1DBlockTiling_using_cute(at::Tensor &a, at::Tensor &b, at::Tensor &c,
                                        float *d_a, float *d_b, float *d_c,
                                        int m, int n, int k)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    auto BM = Int<64>{};
    auto BN = Int<64>{};
    auto BK = Int<8>{};

    auto TM = Int<8>{};
    auto TN = Int<64>{};

    // mat layout
    auto mC = make_layout(make_shape(N, M));
    auto mA = make_layout(make_shape(K, M));
    auto mB = make_layout(make_shape(N, K));

    // block layout
    auto bC = make_layout(make_shape(BN, BM));
    auto bA = make_layout(make_shape(BK, BM));
    auto bB = make_layout(make_shape(BN, BK));

    // thread layout
    auto tC = make_layout(make_shape(TN, TM));
    auto tA = make_layout(make_shape(BK, BM));
    auto tB = make_layout(make_shape(BN, BK));

    dim3 block(size(tC));
    dim3 grid(ceil_div(size(M), BM), ceil_div(size(N), BN));

    c.zero_();
    cudaMemset(d_c, 0, c.numel() * c.element_size());

    // std::cout << block << std::endl;
    // std::cout << grid << std::endl;

    // std::cout << a << std::endl;
    // std::cout << b << std::endl;
    // std::cout << a.matmul(b) << std::endl;
    sgemm1DBlockTilingUsingCute<<<grid, block>>>(d_a, d_b, d_c,
                                                 mA, bA, tA,
                                                 mB, bB, tB,
                                                 mC, bC, tC);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    // std::cout << c << std::endl;
    std::cout << (c.allclose(a.matmul(b)) ? "Cute SMEM 1D Success" : "Cute SMEM 1D Failed") << std::endl;

    // std::cout << at::isclose(c, a.matmul(b)).count_nonzero() << std::endl;
    // std::cout << c.numel() - at::isclose(c, a.matmul(b)).count_nonzero() << std::endl;
}

int main(int argc, char *argv[])
{
    int mat_size{4096};
    if (argc >= 2)
    {
        mat_size = atoi(argv[1]);
    }
    int M = mat_size;
    int K = mat_size;
    int N = mat_size;

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

    host_sgemm1DBlockTiling(a, b, c, d_a, d_b, d_c, M, N, K);
    host_sgemm1DBlockTiling_using_cute(a, b, c, d_a, d_b, d_c, M, N, K);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
}