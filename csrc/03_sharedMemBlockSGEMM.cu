#include <cute/tensor.hpp>
#include <cute/underscore.hpp>
#include <ATen/ATen.h>
#include <iostream>

using namespace cute;
using X = Underscore;

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

template <typename ALayout, typename ABlockLayout, typename AThreadLayout,
          typename BLayout, typename BBlockLayout, typename BThreadLayout,
          typename CLayout, typename CBlockLayout, typename CThreadLayout,
          typename T>
__global__ static __launch_bounds__(decltype(size(CBlockLayout{}))::value) void sgemm_share_mem_block_using_cute(T const *A, T const *B, T *C,
                                                                                                                 ALayout mA, ABlockLayout bA, AThreadLayout tA,
                                                                                                                 BLayout mB, BBlockLayout bB, BThreadLayout tB,
                                                                                                                 CLayout mC, CBlockLayout bC, CThreadLayout tC)
{
    T regA{0}, regB{0}, regC{0};
    __shared__ T smemA[cosize_v<ABlockLayout>];
    __shared__ T smemB[cosize_v<BBlockLayout>];

    // gmem layout
    auto tensorC = make_tensor(make_gmem_ptr(C), mC);
    auto tensorA = make_tensor(make_gmem_ptr(A), mA);
    auto tensorB = make_tensor(make_gmem_ptr(B), mB);

    // smem layout
    auto sA = make_tensor(make_smem_ptr(smemA), bA);
    auto sB = make_tensor(make_smem_ptr(smemB), bB);

    // local tile work of the block
    auto gC = local_tile(tensorC, shape(bC), make_coord(blockIdx.x, blockIdx.y));
    auto gA = local_tile(tensorA, shape(bA), make_coord(_, blockIdx.y));
    auto gB = local_tile(tensorB, shape(bB), make_coord(blockIdx.x, _));

    // local partition work for thr thread
    auto bCgC = local_partition(gC, bC, threadIdx.x);

    auto bAgA = local_partition(gA, bA, threadIdx.x);
    auto bAsA = local_partition(sA, bA, threadIdx.x);

    auto bBgB = local_partition(gB, bB, threadIdx.x);
    auto bBsB = local_partition(sB, bB, threadIdx.x);

    // element need for dot product
    auto needA = local_tile(sA, shape(tA), make_coord(_, get<1>(bC.get_flat_coord(threadIdx.x))));
    auto needB = local_tile(sB, shape(tB), make_coord(get<0>(bC.get_flat_coord(threadIdx.x)), _));

    // rmem layout
    auto rC = make_tensor(make_rmem_ptr(&regC), tC);
    auto rA = make_tensor(make_rmem_ptr(&regA), tA);
    auto rB = make_tensor(make_rmem_ptr(&regB), tB);

    auto k_tile_max = size<2>(bAgA);
    auto k_max = size<2>(needA);

    if (thread0())
    {
        print("C Info : \n");
        print("\t\t Tensor layout in global mem: ");
        print(mC);
        print("\n");
        print("\t\t Tensor layout in rmem: ");
        print(rC);
        print("\n");
        print("\t\tWork for the block : ");
        print(gC);
        print("\n");
        print("\t\tWork for the thread : ");
        print(bCgC);
        print("\n");

        print("A Info : \n");
        print("\t\t Tensor layout in global mem: ");
        print(mA);
        print("\n");
        print("\t\t Tensor layout in smem: ");
        print(sA);
        print("\n");
        print("\t\t Tensor layout in rmem: ");
        print(rA);
        print("\n");
        print("\t\tCopy Work for the block : ");
        print(gA);
        print("\n");
        print("\t\tCopy Work for the thread from Global: ");
        print(bAgA);
        print("\n");
        print("\t\tCopy Work for the thread to shared : ");
        print(bAsA);
        print("\n");
        print("\t\tElement needed for dot product ");
        print(needA);
        print("\n");

        print("B Info : \n");
        print("\t\t Tensor layout in global mem: ");
        print(mB);
        print("\n");
        print("\t\t Tensor layout in smem: ");
        print(sB);
        print("\n");
        print("\t\t Tensor layout in rmem: ");
        print(rB);
        print("\n");
        print("\t\tCopy Work for the block : ");
        print(gB);
        print("\n");
        print("\t\tCopy Work for the thread from Global: ");
        print(bBgB);
        print("\n");
        print("\t\tCopy Work for the thread to shared : ");
        print(bBsB);
        print("\n");
        print("\t\tElement needed for dot product ");
        print(needB);
        print("\n");
    }

    for (int k_tile{0}; k_tile < k_tile_max; ++k_tile)
    {
        // copy_if(TrivialPredTensor{}, bAgA(_, _, k_tile), bAsA);
        // copy_if(TrivialPredTensor{}, bBgB(_, _, k_tile), bBsB);

        copy(bAgA(_, _, k_tile), bAsA);
        copy(bBgB(_, _, k_tile), bBsB);
        cp_async_fence();
        cp_async_wait<0>();

        // for (int i{0}; i < size(bA); ++i)
        // {
        //     bAsA(i) = bAgA(_, _, k_tile)(i);
        //     bBsB(i) = bBgB(_, _, k_tile)(i);
        // }
        __syncthreads();

        for (int k{0}; k < k_max; ++k)
        {
            // copy_if(TrivialPredTensor{}, needA(_, _, k), rA);
            // copy_if(TrivialPredTensor{}, needB(_, _, k), rB);

            copy(needA(_, _, k), rA);
            copy(needB(_, _, k), rB);

            // for (int i{0}; i < size(tA); ++i)
            // {
            //     rA(i) = needA(_, _, k)(i);
            //     rB(i) = needB(_, _, k)(i);
            // }

            regC += regA * regB;
        }
        __syncthreads();
    }
    copy(rC, bCgC);
    // for (int i{0}; i < size(rC); ++i)
    // {
    //     bCgC(i) = rC(i);
    // }
}

void host_sgemm_shared_mem_block(at::Tensor &a, at::Tensor &b, at::Tensor &c,
                                 float *d_a, float *d_b, float *d_c,
                                 int M, int N, int K)
{
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / (block.x), (N + block.y - 1) / (block.y));

    cudaMemset(d_c, 0, c.element_size() * c.numel());
    sgemm_shared_mem_block<16, 16><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    std::cout << (c.allclose(a.matmul(b)) ? "SMEM Success" : "SMEM Failed") << std::endl;
}

void host_sgemm_shared_mem_block_using_cute(at::Tensor &a, at::Tensor &b, at::Tensor &c,
                                            float *d_a, float *d_b, float *d_c,
                                            int m, int n, int k)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    auto BM = Int<8>{};
    auto BN = Int<8>{};
    auto BK = Int<8>{};

    // layouts for C
    auto mC = make_layout(make_shape(M, N));
    auto bC = make_layout(make_shape(BM, BN));
    auto tC = make_layout(make_shape(_1{}, _1{}));

    // layout for A
    auto mA = make_layout(make_shape(K, M));
    auto bA = make_layout(make_shape(BK, BM));
    auto tA = make_layout(make_shape(_1{}, _1{}));

    // layout for B
    auto mB = make_layout(make_shape(N, K));
    auto bB = make_layout(make_shape(BN, BK));
    auto tB = make_layout(make_shape(_1{}, _1{}));

    dim3 block(size(bC));
    dim3 grid(ceil_div(size(M), BM),
              ceil_div(size(N), BN));

    std::cout << block << std::endl;
    std::cout << grid << std::endl;
    
    c.zero_();
    cudaMemset(d_c, 0, c.element_size() * c.numel()); 
    sgemm_share_mem_block_using_cute<<<grid, block>>>(d_a, d_b, d_c,
                                                      mA, bA, tA,
                                                      mB, bB, tB,
                                                      mC, bC, tC);
    cudaMemcpy(c.data_ptr(), d_c, c.element_size() * c.numel(), cudaMemcpyDeviceToHost);
    std::cout << (c.allclose(a.matmul(b)) ? "Cute SMEM Success" : "Cute SMEM Failed") << std::endl;

    auto org_ans = a.matmul(b);
    auto failed_index = (at::isclose(c, a.matmul(b)) == 0);
    auto non_zero = failed_index.nonzero();
    auto uni_row = at::_unique2(non_zero.index({at::indexing::Slice(at::indexing::None), at::indexing::Slice(0, 1)}), true, false, true);
    auto uni_col = at::_unique2(non_zero.index({at::indexing::Slice(at::indexing::None), at::indexing::Slice(1)}), true, false, true);
    auto uni_row_tensor = at::stack(at::TensorList({get<0>(uni_row), get<2>(uni_row)}), 1);
    auto uni_col_tensor = at::stack(at::TensorList({get<0>(uni_col), get<2>(uni_col)}), 1);

    std::cout << at::stack(at::TensorList({non_zero.index({at::indexing::Slice(at::indexing::None), at::indexing::Slice(0, 1)}).squeeze(),
                                           non_zero.index({at::indexing::Slice(at::indexing::None), at::indexing::Slice(1)}).squeeze(),
                                           org_ans.index({failed_index}).squeeze(), c.index({failed_index}).squeeze(),
                                           (org_ans.index({failed_index}) - c.index({failed_index})).squeeze()}),
                           1)
              << std::endl;

    std::cout << uni_row_tensor << std::endl;
    std::cout << uni_col_tensor << std::endl;

    
    std::cout << at::isclose(c, a.matmul(b)).count_nonzero() << std::endl;
    std::cout << c.numel() - at::isclose(c, a.matmul(b)).count_nonzero() << std::endl;
}

int main(int argc, char *argv[])
{
    int mat_size{192};
    if (argc >= 2)
    {
        mat_size = atoi(argv[1]);
    }
    int M = mat_size;
    int K = mat_size;
    int N = mat_size;
    at::manual_seed(42);

    auto a = at::rand({M, K}, at::TensorOptions().dtype(at::kFloat));
    auto b = at::rand({K, N}, at::TensorOptions().dtype(at::kFloat));
    auto c = at::zeros({M, N}, at::TensorOptions().dtype(at::kFloat));

    // auto a = at::fill(at::zeros({M, N}, at::TensorOptions().dtype(at::kFloat)), 0.5);
    // auto b = at::fill(at::zeros({K, N}, at::TensorOptions().dtype(at::kFloat)), 0.5);
    // auto c = at::zeros({M, N}, at::TensorOptions().dtype(at::kFloat));

    // allocate and copy to device
    auto nBytes = a.element_size() * a.numel();
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, nBytes);
    cudaMalloc((void **)&d_b, nBytes);
    cudaMalloc((void **)&d_c, c.element_size() * c.numel());

    cudaMemcpy(d_a, a.data_ptr(), nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data_ptr(), nBytes, cudaMemcpyHostToDevice);

    c = a.matmul(b);
    cudaMemcpy(d_c, c.data_ptr(), nBytes, cudaMemcpyHostToDevice);

    // host_sgemm_shared_mem_block(a, b, c, d_a, d_b, d_c, M, N, K);
    host_sgemm_shared_mem_block_using_cute(a, b, c, d_a, d_b, d_c, M, N, K);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
}