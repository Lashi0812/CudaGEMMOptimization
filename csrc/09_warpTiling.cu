#include <ATen/ATen.h>
#include <iostream>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32;

namespace wt {
template <const uint BM, const uint BN, const uint BK, const uint strideA, const uint strideB>
__device__ void loadFromGMEM(
  float     *chunkA,
  float     *chunkB,
  float     *AS,
  float     *BS,
  const uint K,
  const uint N,
  const uint innerRowA,
  const uint innerColA,
  const uint innerRowB,
  const uint innerColB) {
    // move A chunk into reg  in vectorize and then store in SMEM in transposed manner
    for (uint offset{0}; offset + strideA <= BM; offset += strideA) {
        const float4 tmp =
          reinterpret_cast<float4 *>(&chunkA[(innerRowA + offset) * K + innerColA * 4])[0];
        // transpose and store in smem
        // todo :: bank conflict while storing the data
        AS[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        AS[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        AS[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        AS[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    // move B chunk into reg and then store in SMEM
    for (uint offset{0}; offset + strideB <= BK; offset += strideB) {
        reinterpret_cast<float4 *>(&BS[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(&chunkB[(innerRowB + offset) * N + innerColB * 4])[0];
    }
}

template <
  const uint BM,
  const uint BN,
  const uint BK,
  const uint WMITER,
  const uint WM,
  const uint WSUBM,
  const uint WNITER,
  const uint WN,
  const uint WSUBN,
  const uint TM,
  const uint TN>
__device__ void processFromSMEM(
  float     *regM,
  float     *regN,
  float     *threadResults,
  float     *AS,
  float     *BS,
  const uint warpRow,
  const uint warpCol,
  const uint threadRowInWarp,
  const uint threadColInWarp) {
    // loop through BK
    for (uint dotIdx{0}; dotIdx < BK; ++dotIdx) {
        // loop through warp sub tile iter
        for (uint wSubRowIdx{0}; wSubRowIdx < WMITER; ++wSubRowIdx) {
            // loop through thread
            for (uint i{0}; i < TM; ++i) {
                regM[wSubRowIdx * TM + i] =
                  AS[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
            }
        }

        // loop through warp sub tile iter
        for (uint wSubColIdx{0}; wSubColIdx < WNITER; ++wSubColIdx) {
            // loop through thread
            for (uint i{0}; i < TN; ++i) {
                regN[wSubColIdx * TN + i] =
                  BS[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            }
        }

        // do outer product
        for (uint wSubRowIdx{0}; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint wSubColIdx{0}; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint resIdxM{0}; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN{0}; resIdxN < TN; ++resIdxN) {
                        threadResults
                          [(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) +
                           resIdxN] +=
                          regM[wSubRowIdx * TM + resIdxM] * regN[wSubColIdx * TN + resIdxN];
                    }
                }
            }
        }
    }
}

} // namespace wt

template <
  const uint BM,
  const uint BN,
  const uint BK,
  const uint WM,
  const uint WN,
  const uint WNITER,
  const uint TM,
  const uint TN,
  const uint N_THREADS>
__global__ void __launch_bounds__(N_THREADS) warpTile(
  float *__restrict__ A,
  float *__restrict__ B,
  float     *C,
  const uint M,
  const uint N,
  const uint K) {

    // warp arrangement in block
    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);

    // size of warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    const uint     WSUBM  = WM / WMITER;
    const uint     WSUBN  = WN / WNITER;

    // thread arrangement within the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);

    // allocate shared mem
    __shared__ float AS[BM * BK];
    __shared__ float BS[BK * BN];

    // move the A mat pointer to start of the Row
    A += blockIdx.y * BM * K;
    // move the B mat pointer to start of the col
    B += blockIdx.x * BN;
    // move the C mat pointer to C tile then move it warp tile in that  C tile
    C += (blockIdx.y * BM + warpRow * WM) * N + blockIdx.x * BN + warpCol * WN;

    // thread arrangement for moving the A block tile from global mem to shared mem
    // change A/(B/C) into (A*C)/B
    const uint     innerRowA = threadIdx.x / (BK / 4);
    const uint     innerColA = threadIdx.x % (BK / 4);
    constexpr uint strideA   = (N_THREADS * 4) / BK;
    // thread arrangement for moving the B block tile from global mem to shared mem
    const uint     innerRowB = threadIdx.x / (BN / 4);
    const uint     innerColB = threadIdx.x % (BN / 4);
    constexpr uint strideB   = N_THREADS / (BN / 4);

    // allocate the register for thread tiling
    float threadResults[TM * WMITER * TN * WNITER] = {0.0};
    float regM[TM * WMITER]                        = {0.0};
    float regN[TN * WNITER]                        = {0.0};

    // loop through the chunks
    for (uint chunk{0}; chunk < K; chunk += BK) {
        // move the global to shared mem
        wt::loadFromGMEM<BM, BN, BK, strideA, strideB>(
          A, B, AS, BS, K, N, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
        // move from shared mem to reg then do outer product
        wt::processFromSMEM<BM, BN, BK, WMITER, WM, WSUBM, WNITER, WN, WSUBN, TM, TN>(
          regM, regN, threadResults, AS, BS, warpRow, warpCol, threadRowInWarp, threadColInWarp);
        // move the A and B mat to next chunk
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // move C pointer to current warp subtile
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    reinterpret_cast<float4 *>(&C_interim
                                                 [(threadRowInWarp * TM + resIdxM) * N +
                                                  threadColInWarp * TN + resIdxN])[0] =
                      reinterpret_cast<float4 *>(&threadResults
                                                   [(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                                    wSubColIdx * TN + resIdxN])[0];
                }
            }
        }
    }
}

void host_warpTiling(
  at::Tensor &a,
  at::Tensor &b,
  at::Tensor &c,
  float      *d_a,
  float      *d_b,
  float      *d_c,
  int         M,
  int         N,
  int         K) {
    const uint N_THREADS = 128;
    // block level
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 16;
    // warp level
    const uint WM     = 64;
    const uint WN     = 64;
    const uint WNITER = 4;
    // thread level
    const uint TM     = 8;
    const uint TN     = 4;
    auto       nBytes = a.numel() * a.element_size();

    // execution configuration
    dim3 block(N_THREADS);
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));

    // launch the kernel
    warpTile<BM, BN, BK, WM, WN, WNITER, TM, TN, N_THREADS>
      <<<grid, block>>>(d_a, d_b, d_c, M, N, K);

    // copy the result
    cudaMemcpy(c.data_ptr(), d_c, nBytes, cudaMemcpyDeviceToHost);

    // check the answer
    std::cout << (c.allclose(a.matmul(b)) ? "SUCCESS" : "FAIL") << std::endl;
}

int main() {
    const uint M = 4096;
    const uint K = 4096;
    const uint N = 4096;

    auto A = at::rand({M, K}, at::TensorOptions().dtype(at::kFloat));
    auto B = at::rand({K, N}, at::TensorOptions().dtype(at::kFloat));
    auto C = at::rand({M, N}, at::TensorOptions().dtype(at::kFloat));

    float *d_A, *d_B, *d_C;
    auto   nBytes = A.numel() * A.element_size();

    cudaMalloc((void **)&d_A, nBytes);
    cudaMalloc((void **)&d_B, nBytes);
    cudaMalloc((void **)&d_C, nBytes);

    cudaMemcpy(d_A, A.data_ptr(), nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data_ptr(), nBytes, cudaMemcpyHostToDevice);

    host_warpTiling(A, B, C, d_A, d_B, d_C, M, N, K);

    // free the memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
}