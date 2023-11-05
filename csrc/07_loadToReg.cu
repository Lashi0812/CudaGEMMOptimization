#include <ATen/ATen.h>
#include <iostream>
#include <cuda_runtime.h>

template <int BN, int BK, int TN>
__global__ void naiveLoadToReg(float *in, float *out)
{
    __shared__ float BS[BK * BN];

    const uint threadCol = threadIdx.x % (BN / TN);

    uint innerRowB = threadIdx.x / (BN / 4);
    uint innerColB = threadIdx.x % (BN / 4);

    uint strideB = blockDim.x / (BN / 4);

    for (uint offset{0}; offset < BK; offset += strideB)
    {
        reinterpret_cast<float4 *>(&BS[(innerRowB + offset) * BN + 4 * innerColB])[0] = reinterpret_cast<float4 *>(&in[(innerRowB + offset) * BN + 4 * innerColB])[0];
    }
    __syncthreads();

    float regB{0.0f};
    regB = BS[threadCol * TN];
    out[threadIdx.x] = regB;
}

template <int BN, int BK, int TN, int WN>
__global__ void warpLoadToReg(float *in, float *out)
{
    __shared__ float BS[BK * BN];

    const uint warpIdx = threadIdx.x / warpSize;
    // const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);

    const uint threadIdxInWarp = threadIdx.x % warpSize;
    // const uint threadRowInWarp = threadIdxInWarp / (WN / TN);
    const uint threadColInWarp = threadIdxInWarp % (WN / TN);

    uint innerRowB = threadIdx.x / (BN / 4);
    uint innerColB = threadIdx.x % (BN / 4);

    uint strideB = blockDim.x / (BN / 4);

    for (uint offset{0}; offset < BK; offset += strideB)
    {
        reinterpret_cast<float4 *>(&BS[(innerRowB + offset) * BN + 4 * innerColB])[0] = reinterpret_cast<float4 *>(&in[(innerRowB + offset) * BN + 4 * innerColB])[0];
    }
    __syncthreads();

    float regB{0.0f};
    // regB = BS[threadCol * TN];
    regB = BS[warpCol * WN + threadColInWarp * TN];
    out[threadIdx.x] = regB;
}

template <int BN, int BK, int TN, int WN, int WNITER>
__global__ void subWarpLoadToReg(float *in, float *out)
{
    __shared__ float BS[BK * BN];

    const uint warpIdx = threadIdx.x / warpSize;
    // const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);

    const uint WSUBN = WN / WNITER;

    const uint threadIdxInWarp = threadIdx.x % warpSize;
    // const uint threadRowInWarp = threadIdxInWarp / (WN / TN);
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);

    uint innerRowB = threadIdx.x / (BN / 4);
    uint innerColB = threadIdx.x % (BN / 4);

    uint strideB = blockDim.x / (BN / 4);

    for (uint offset{0}; offset < BK; offset += strideB)
    {
        reinterpret_cast<float4 *>(&BS[(innerRowB + offset) * BN + 4 * innerColB])[0] = reinterpret_cast<float4 *>(&in[(innerRowB + offset) * BN + 4 * innerColB])[0];
    }
    __syncthreads();

    float regB{0.0f};
    // regB = BS[threadCol * TN];
    regB = BS[warpCol * WN + threadColInWarp * TN];
    out[threadIdx.x] = regB;
}

int main()
{
    const int BN = 128;
    const int BK = 16;
    const int TN = 16;
    const int WN = 64;
    const int WNITER = 4;

    auto in = at::arange(BK * BN, at::TensorOptions().dtype(at::kFloat)).reshape({BK, BN});
    auto out = at::zeros(BN, at::TensorOptions().dtype(at::kFloat));

    auto inBytes = in.element_size() * in.numel();
    auto outBytes = out.element_size() * out.numel();
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, inBytes);
    cudaMalloc((void **)&d_out, outBytes);

    cudaMemcpy(d_in, in.data_ptr(), inBytes, cudaMemcpyHostToDevice);

    cudaMemset(d_out, 0, outBytes);
    out.zero_();
    naiveLoadToReg<BN, BK, TN><<<1, 128>>>(d_in, d_out);
    cudaMemcpy(out.data_ptr(), d_out, outBytes, cudaMemcpyDeviceToHost);
    // std::cout << out << std::endl;

    cudaMemset(d_out, 0, outBytes);
    out.zero_();
    warpLoadToReg<BN, BK, TN, WN><<<1, 128>>>(d_in, d_out);
    cudaMemcpy(out.data_ptr(), d_out, outBytes, cudaMemcpyDeviceToHost);
    // std::cout << out << std::endl;

    const int TN2 = 4;
    cudaMemset(d_out, 0, outBytes);
    out.zero_();
    subWarpLoadToReg<BN, BK, TN2, WN,WNITER><<<1, 128>>>(d_in, d_out);
    cudaMemcpy(out.data_ptr(), d_out, outBytes, cudaMemcpyDeviceToHost);
    // std::cout << out << std::endl;
    cudaDeviceReset();
}