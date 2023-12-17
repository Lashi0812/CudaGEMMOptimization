#include <ATen/ATen.h>
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

auto nBytes(const at::Tensor &tensor)
{
    return tensor.numel() * tensor.element_size();
}

///////////////////////////////////////////////////////////////////////////////
//                      Kernels
///////////////////////////////////////////////////////////////////////////////

template <typename BlockLayout, typename ThreadLayout>
__global__ void kernel_partition_arrangement(float *in, float *out,
                                             BlockLayout block_layout, ThreadLayout thread_layout)
{
    __shared__ float smem[cosize_v<BlockLayout>];

    auto g_in = make_tensor(make_gmem_ptr(in), block_layout);
    auto s_in = make_tensor(make_smem_ptr(smem), block_layout);
    auto g_out = make_tensor(make_gmem_ptr(out), block_layout);

    auto g_in_src = local_partition(g_in, thread_layout, threadIdx.x);
    auto s_in_dst = local_partition(s_in, thread_layout, threadIdx.x);

    auto g_out_dst = local_partition(g_out, thread_layout, threadIdx.x);

    auto r_in = make_tensor_like(s_in_dst);

    copy(g_in_src, s_in_dst);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    copy(s_in_dst, r_in);
    for (int i{0}; i < size(r_in); ++i)
    {
        r_in(i) *= 10;
    }
    copy(r_in, g_out_dst);
}

template <typename BlockLayout, typename ThreadTile>
__global__ void kernel_tile_arrangement(float *in, float *out,
                                        BlockLayout block_layout, ThreadTile thread_tile)
{
    __shared__ float smem[cosize_v<BlockLayout>];

    auto g_in = make_tensor(make_gmem_ptr(in), block_layout);
    auto s_in = make_tensor(make_smem_ptr(smem), block_layout);
    auto g_out = make_tensor(make_gmem_ptr(out), block_layout);

    auto g_in_src = local_tile(g_in, thread_tile, threadIdx.x);
    auto s_in_dst = local_tile(s_in, thread_tile, threadIdx.x);

    auto g_out_dst = local_tile(g_out, thread_tile, threadIdx.x);

    auto r_in = make_tensor_like(s_in_dst);

    copy(g_in_src, s_in_dst);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    copy(s_in_dst, r_in);
    for (int i{0}; i < size(r_in); ++i)
    {
        r_in(i) *= 10;
    }
    copy(r_in, g_out_dst);
}

///////////////////////////////////////////////////////////////////////////////
//                      Host Functions
///////////////////////////////////////////////////////////////////////////////

void test_partition_arrangement(at::Tensor &in, at::Tensor &out, float *d_in, float *d_out)
{
    auto M = Int<128>{};
    auto N = Int<8>{};
    auto TM = Int<32>{};
    auto TN = Int<8>{};

    auto block_layout = make_layout(make_shape(N, M));
    auto thread_layout = make_layout(make_shape(TN, TM));

    dim3 block(size(thread_layout));
    dim3 grid(1);
    kernel_partition_arrangement<<<grid, block>>>(d_in, d_out,
                                                  block_layout,
                                                  thread_layout);
    cudaMemcpy(out.data_ptr(), d_out, nBytes(out), cudaMemcpyDeviceToHost);
    std::cout << (out.allclose(in * 10) ? "Partition Success" : "Partition Failed") << std::endl;
}

void test_tile_arrangement(at::Tensor &in, at::Tensor &out, float *d_in, float *d_out)
{
    auto M = Int<128>{};
    auto N = Int<8>{};
    auto TM = Int<1>{};
    auto TN = Int<4>{};

    auto block_layout = make_layout(make_shape(N, M));
    auto thread_tile = make_shape(TN, TM);

    dim3 block((M * N) / (TN * TM));
    dim3 grid(1);
    kernel_tile_arrangement<<<grid, block>>>(d_in, d_out,
                                             block_layout,
                                             thread_tile);
    cudaMemcpy(out.data_ptr(), d_out, nBytes(out), cudaMemcpyDeviceToHost);
    std::cout << (out.allclose(in * 10) ? "TIle Success" : "TIle Failed") << std::endl;
}

int main()
{
    auto in = at::arange(128 * 8, at::TensorOptions().dtype(at::kFloat));
    auto out = at::zeros_like(in);

    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, nBytes(in));
    cudaMalloc((void **)&d_out, nBytes(out));

    cudaMemcpy(d_in, in.data_ptr(), nBytes(in), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, nBytes(out));

    test_partition_arrangement(in, out, d_in, d_out);
    test_tile_arrangement(in, out, d_in, d_out);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaDeviceReset();
}