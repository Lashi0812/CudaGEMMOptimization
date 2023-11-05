#include <ATen/ATen.h>
#include <iostream>
#include <cuda_runtime.h>

template <uint BN, uint BK>
__global__ void loadSmem32(float *in, float *out)
{
    __shared__ float BS[BK * BN];

    uint innerRowB = threadIdx.x / BN;
    uint innerColB = threadIdx.x % BN;

    uint strideB = blockDim.x / BN;

    for (uint offset{0}; offset < BK; offset += strideB)
    {
        BS[(innerRowB + offset) * BN + innerColB] = in[(innerRowB + offset) * BN + innerColB];
    }
    __syncthreads();
    for (uint offset{0}; offset < BK; offset += strideB)
    {
        out[(innerRowB + offset) * BN + innerColB] = BS[(innerRowB + offset) * BN + innerColB];
    }
}

template<uint BN, uint BK>
__global__ void loadSmemVectorize(float *in, float *out)
{
    __shared__ float BS[BK * BN];

    uint innerRowB = threadIdx.x / (BN/4);
    uint innerColB = threadIdx.x % (BN/4);

    uint strideB = blockDim.x / (BN / 4);

    for (uint offset{0}; offset < BK; offset += strideB)
    {
        reinterpret_cast<float4 *>(&BS[(innerRowB + offset) * BN + 4 * innerColB])[0] = reinterpret_cast<float4 *>(&in[(innerRowB + offset) * BN + 4 * innerColB])[0];
    }
    __syncthreads();
    innerRowB = threadIdx.x / BN;
    innerColB = threadIdx.x % BN;
    strideB = blockDim.x / BN;
    for (uint offset{0}; offset < BK; offset += strideB)
    {
        out[(innerRowB + offset) * BN + innerColB] = BS[(innerRowB + offset) * BN + innerColB];
    }
}

int main()
{
    const uint BN = 128;
    const uint BK = 16;


    auto in = at::rand({BK, BN}, at::TensorOptions().dtype(at::kFloat));
    auto out = at::zeros({BK, BN}, at::TensorOptions().dtype(at::kFloat));

    auto nBytes = in.element_size() * in.numel();
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, nBytes);
    cudaMalloc((void **)&d_out, nBytes);

    cudaMemcpy(d_in, in.data_ptr(), nBytes, cudaMemcpyHostToDevice);

    cudaMemset(d_out,0,nBytes);
    loadSmem32<BN, BK><<<1, 128>>>(d_in, d_out);
    cudaMemcpy(out.data_ptr(), d_out, nBytes, cudaMemcpyDeviceToHost);
    std::cout << (out.allclose(in) ? "normal load Success" : "normal load Fail") << std::endl;

    cudaMemset(d_out,0,nBytes);
    out.zero_();
    loadSmemVectorize<BN, BK><<<1, 128>>>(d_in, d_out);
    cudaMemcpy(out.data_ptr(), d_out, nBytes, cudaMemcpyDeviceToHost);
    std::cout << (out.allclose(in) ? "Vectorize load Success" : "Vectorize load Fail") << std::endl;
    cudaDeviceReset();
}