#include "cute/tensor.hpp"
#include "cute/int_tuple.hpp"
#include "cute/numeric/tfloat.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/tuple_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits.hpp"
#include <ATen/ATen.h>
#include <iostream>
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/stride.hpp"
#include "cute/swizzle.hpp"
#include "cute/underscore.hpp"
#include "cute/util/debug.hpp"

using namespace cute;
using X = Underscore;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32;

// Utility
constexpr int exponent(int result) {
    int exponent = 0;
    while (result > 1) {
        result = result >> 1; // Right shift the result by 1 bit
        exponent++;
    }
    return exponent;
}

template <class IntT>
Stride<IntT, Int<1>> make_cute_packed_stride(Stride<IntT, Int<1>> s, Shape<int, int> shape_MK) {
    static_assert(
      std::is_integral_v<IntT>,
      "Stride must have an integral type so it can be set dynamically. Static strides not "
      "supported.");
    auto s_copy    = s;
    get<0>(s_copy) = static_cast<IntT>(get<1>(shape_MK));
    return s_copy;
}

template <class IntT>
Stride<Int<1>, IntT> make_cute_packed_stride(Stride<Int<1>, IntT> s, Shape<int, int> shape_MK) {
    static_assert(
      std::is_integral_v<IntT>,
      "Stride must have an integral type so it can be set dynamically. Static strides not "
      "supported.");
    auto s_copy    = s;
    get<1>(s_copy) = static_cast<IntT>(get<0>(shape_MK));
    return s_copy;
}

template <class L>
struct TagToStrideA {
    using type = L;
};

template <>
struct TagToStrideA<GenRowMajor> {
    using type = Stride<int32_t, Int<1>>;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<GenColMajor> {
    using type = Stride<Int<1>, int32_t>;
};

template <class L>
struct TagToStrideB {
    using type = L;
};

template <>
struct TagToStrideB<GenRowMajor> {
    using type = Stride<Int<1>, int32_t>;
};

template <>
struct TagToStrideB<GenColMajor> {
    using type = Stride<int32_t, Int<1>>;
};

template <class LayoutTag>
struct TagToStrideC : TagToStrideA<LayoutTag> {};

// Convenience aliases
template <class LayoutTag>
using TagToStrideA_t = typename TagToStrideA<LayoutTag>::type;

template <class LayoutTag>
using TagToStrideB_t = typename TagToStrideB<LayoutTag>::type;

template <class LayoutTag>
using TagToStrideC_t = typename TagToStrideC<LayoutTag>::type;

template <int I, class Shape>
using get_t = decltype(get<I>(declval<Shape>()));

template <typename T, typename sizeBK, typename Major, int ThrCount>
struct OperandA;

template <typename T, typename sizeBK, typename Major, int ThrCount>
struct OperandB;

// 128x128xBK K-Major (Row Major)
template <typename T, typename sizeBK, int ThrCount>
struct OperandA<T, sizeBK, GenRowMajor, ThrCount> {
    static int constexpr M         = exponent(16 / sizeof(T));
    static int constexpr S         = 7 - (exponent(sizeof(T)) + M);
    static int constexpr B         = (3 + exponent(sizeBK::value)) - (7 - exponent(sizeof(T)));
    static int constexpr Alignment = sizeof(uint128_t) / sizeof(T);
    static int constexpr ThrMode1  = sizeBK::value / Alignment;
    static int constexpr ThrMode0  = ThrCount / ThrMode1;

    using SmemLayout =
      decltype(composition(Swizzle<B, M, S>{}, Layout<Shape<_8, sizeBK>, Stride<sizeBK, _1>>{}));
    using SRCopyAtom  = Copy_Atom<SM75_U32x4_LDSM_N, T>;
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, T>{},
      Layout<Shape<Int<ThrMode0>, Int<ThrMode1>>, Stride<Int<ThrMode1>, _1>>{},
      Layout<Shape<_1, Int<Alignment>>>{}));
};

// 128x128xBK M-Major (Col Major)
template <typename T, typename sizeBK, int ThrCount>
struct OperandA<T, sizeBK, GenColMajor, ThrCount> {
    static int constexpr M         = exponent(16 / sizeof(T));
    static int constexpr S         = 7 - (exponent(sizeof(T)) + M);
    static int constexpr B         = (3 + exponent(sizeBK::value)) - (7 - exponent(sizeof(T)));
    static int constexpr Alignment = sizeof(uint128_t) / sizeof(T);
    static int constexpr ThrMode1  = sizeBK::value / Alignment;
    static int constexpr ThrMode0  = ThrCount / ThrMode1;

    using SmemLayout =
      decltype(composition(Swizzle<B, M, S>{}, Layout<Shape<sizeBK, _8>, Stride<_1, sizeBK>>{}));
    using SRCopyAtom  = Copy_Atom<SM75_U16x8_LDSM_T, T>;
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, T>{},
      Layout<Shape<Int<ThrMode0>, Int<ThrMode1>>, Stride<_1, Int<ThrMode0>>>{},
      Layout<Shape<Int<Alignment>, _1>>{}));
};

// Operand B - Column-Major (K-major)
template <typename T, typename sizeBK, int ThrCount>
struct OperandB<T, sizeBK, GenColMajor, ThrCount> : OperandA<T, sizeBK, GenRowMajor, ThrCount> {};

// Operand B - Row-Major (N-major)
template <typename T, typename sizeBK, int ThrCount>
struct OperandB<T, sizeBK, GenRowMajor, ThrCount> : OperandA<T, sizeBK, GenColMajor, ThrCount> {};

template <
  typename TAB,
  typename TC_,
  typename sizeBK,
  typename MMA_Op_,
  typename AMajor,
  typename BMajor>
struct GemmConfig {
    using TA                      = TAB;
    using TB                      = TAB;
    using TC                      = TC_;
    using StrideA                 = TagToStrideA_t<AMajor>;
    using StrideB                 = TagToStrideB_t<BMajor>;
    using StrideC                 = TagToStrideC_t<GenRowMajor>;
    using TileShape               = Shape<_128, _128, sizeBK>;
    static int constexpr ThrCount = 128;
    using LdMatrixElemShapeMNK    = Shape<_16, _16, Int<(16 * 2) / sizeof(TA)>>;
    using ValShape                = decltype(transform(
      LdMatrixElemShapeMNK{}, typename MMA_Traits<MMA_Op_>::Shape_MNK{}, divides{}));
    using TiledMMA = TiledMMA<MMA_Atom<MMA_Op_>, Layout<Shape<_2, _2, _1>>, Layout<ValShape>>;

    using OperandA_ = OperandA<TA, sizeBK, AMajor, ThrCount>;
    using OperandB_ = OperandB<TB, sizeBK, BMajor, ThrCount>;
};

template <typename Config>
struct MainLoop {
    using TA             = typename Config::TA;
    using TB             = typename Config::TB;
    using TC             = typename Config::TC;
    using StrideA        = typename Config::StrideA;
    using StrideB        = typename Config::StrideB;
    using StrideC        = typename Config::StrideC;
    using TiledMMA_      = typename Config::TiledMMA;
    using TileShapeMNK   = typename Config::TileShape;
    using Layout_SA      = decltype(tile_to_shape(
      typename Config::OperandA_::SmemLayout{},
      make_shape(get<0>(TileShapeMNK{}), get<2>(TileShapeMNK{}))));
    using Layout_SB      = decltype(tile_to_shape(
      typename Config::OperandB_::SmemLayout{},
      make_shape(get<1>(TileShapeMNK{}), get<2>(TileShapeMNK{}))));
    using GS_TiledCopy_A = typename Config::OperandA_::GSTiledCopy;
    using GS_TiledCopy_B = typename Config::OperandB_::GSTiledCopy;
    using SR_TiledCopy_A =
      decltype(make_tiled_copy_A(typename Config::OperandA_::SRCopyAtom{}, TiledMMA_{}));
    using SR_TiledCopy_B =
      decltype(make_tiled_copy_B(typename Config::OperandB_::SRCopyAtom{}, TiledMMA_{}));

    MainLoop() = default;

    struct Params {
        TA const *ptr_A;
        TB const *ptr_B;
        TC       *ptr_C;
        StrideA   dA;
        StrideB   dB;
        StrideC   dC;
    };
    using Arguments = Params;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const &_, Arguments const &args) {
        return args;
    }

    struct SharedStorage {
        cute::array_aligned<TA, cute::cosize_v<Layout_SA>> smem_a;
        cute::array_aligned<TB, cute::cosize_v<Layout_SB>> smem_b;
    };

    template <typename TensorA, typename TensorB, typename TensorC>
    CUTE_DEVICE void operator()(TensorA gA, TensorB gB, TensorC gC, char *smem) {
        SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem);

        auto           sA = make_tensor(make_smem_ptr(storage.smem_a.data()), Layout_SA{});
        auto           sB = make_tensor(make_smem_ptr(storage.smem_b.data()), Layout_SB{});
        GS_TiledCopy_A gs_tiledCopy_A;
        auto           gs_thr_copy_A = gs_tiledCopy_A.get_thread_slice(threadIdx.x);
        auto           tAgA          = gs_thr_copy_A.partition_S(gA);
        auto           tAsA          = gs_thr_copy_A.partition_D(sA);

        GS_TiledCopy_B gs_tiledCopy_B;
        auto           gs_thr_copy_B = gs_tiledCopy_B.get_thread_slice(threadIdx.x);
        auto           tBgB          = gs_thr_copy_B.partition_S(gB);
        auto           tBsB          = gs_thr_copy_B.partition_D(sB);

        TiledMMA_ tiledMAA;
        auto      thr_mma = tiledMAA.get_thread_slice(threadIdx.x);
        auto      tCrA    = thr_mma.partition_fragment_A(sA);
        auto      tCrB    = thr_mma.partition_fragment_B(sB);
        auto      tCrC    = thr_mma.partition_C(gC);

        SR_TiledCopy_A sr_tiledCopy_A;
        auto           sr_thr_copy_A = sr_tiledCopy_A.get_thread_slice(threadIdx.x);
        auto           tCsA          = sr_thr_copy_A.partition_S(sA);
        auto           tCrA_view     = sr_thr_copy_A.retile_D(tCrA);
        SR_TiledCopy_B sr_tiledCopy_B;
        auto           sr_thr_copy_B = sr_tiledCopy_B.get_thread_slice(threadIdx.x);
        auto           tCsB          = sr_thr_copy_B.partition_S(sB);
        auto           tCrB_view     = sr_thr_copy_B.retile_D(tCrB);

        auto fragC = partition_fragment_C(tiledMAA, take<0, 2>(TileShapeMNK{}));
        clear(fragC);

        // if (thread0()) {
        //     // clang-format off
        //     print("sA            : ");print(sA           );print("\n");
        //     print("sB            : ");print(sB           );print("\n");
        //     print("gs_thr_copy_A : ");print(gs_thr_copy_A);print("\n");
        //     print("tAgA          : ");print(tAgA         );print("\n");
        //     print("tAsA          : ");print(tAsA         );print("\n");
        //     print("gs_thr_copy_B : ");print(gs_thr_copy_B);print("\n");
        //     print("tBgB          : ");print(tBgB         );print("\n");
        //     print("tBsB          : ");print(tBsB         );print("\n");
        //     print("thr_mma       : ");print(thr_mma      );print("\n");
        //     print("tCrA          : ");print(tCrA         );print("\n");
        //     print("tCrB          : ");print(tCrB         );print("\n");
        //     print("tCrC          : ");print(tCrC         );print("\n");
        //     print("sr_thr_copy_A : ");print(sr_thr_copy_A);print("\n");
        //     print("tCsA          : ");print(tCsA         );print("\n");
        //     print("tCrA_view     : ");print(tCrA_view    );print("\n");
        //     print("sr_thr_copy_B : ");print(sr_thr_copy_B);print("\n");
        //     print("tCsB          : ");print(tCsB         );print("\n");
        //     print("tCrB_view     : ");print(tCrB_view    );print("\n");
        //     print("fragC         : ");print(fragC        );print("\n");
        //     // clang-format on
        // }

        for (int k_tile_iter{0}; k_tile_iter < size<2>(gA); ++k_tile_iter) {
            copy(gs_tiledCopy_A, tAgA(_, _, _, k_tile_iter), tAsA);
            copy(gs_tiledCopy_B, tBgB(_, _, _, k_tile_iter), tBsB);
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();

            copy(sr_tiledCopy_A, tCsA, tCrA_view);
            copy(sr_tiledCopy_B, tCsB, tCrB_view);

            gemm(tiledMAA, fragC, tCrA, tCrB, fragC);
            __syncthreads();
        }
        copy_aligned(fragC, tCrC);
    }
};

template <typename ProblemShapeMNK_, typename MainLoop_>
struct KernelOperator {
    using ProblemShapeMNK = ProblemShapeMNK_;
    using MainLoop        = MainLoop_;
    using MainLoopParams  = typename MainLoop_::Params;
    using TileShapeMNK    = typename MainLoop_::TileShapeMNK;
    using TiledMma        = typename MainLoop_::TiledMMA_;

    static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});

    KernelOperator() = default;
    struct Params {
        ProblemShapeMNK problem_shape;
        MainLoopParams  mainLoop;
    };

    using Arguments = Params;

    static Params to_underlying_arguments(Arguments const &args) {
        return {
          args.problem_shape, MainLoop::to_underlying_arguments(args.problem_shape, args.mainLoop)};
    }

    static dim3 get_grid_shape(Params const &params) {
        return dim3(
          size(ceil_div(shape<0>(params.problem_shape), shape<0>(TileShapeMNK{}))),
          size(ceil_div(shape<1>(params.problem_shape), shape<1>(TileShapeMNK{}))),
          1);
    }

    static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

    CUTE_DEVICE void operator()(Params params, char *smem) {
        auto mA = make_tensor(
          make_gmem_ptr(params.mainLoop.ptr_A),
          make_shape(get<0>(params.problem_shape), get<2>(params.problem_shape)),
          params.mainLoop.dA);
        auto mB = make_tensor(
          make_gmem_ptr(params.mainLoop.ptr_B),
          make_shape(get<1>(params.problem_shape), get<2>(params.problem_shape)),
          params.mainLoop.dB);
        auto mC = make_tensor(
          make_gmem_ptr(params.mainLoop.ptr_C),
          make_shape(get<0>(params.problem_shape), get<1>(params.problem_shape)),
          params.mainLoop.dC);

        auto blk_shape                   = TileShapeMNK{};
        auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
        auto blk_coord                   = make_coord(m_coord, n_coord, _);

        auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, X, _1>{});
        auto gB = local_tile(mB, blk_shape, blk_coord, Step<X, _1, _1>{});
        auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1, _1, X>{});

        // if (thread0()) {
        //     // clang-format off
        //     print("mA        : ");print(mA       );print("\n");
        //     print("mB        : ");print(mB       );print("\n");
        //     print("mC        : ");print(mC       );print("\n");
        //     print("blk_shape : ");print(blk_shape);print("\n");
        //     print("blk_coord : ");print(blk_coord);print("\n");
        //     print("gA        : ");print(gA       );print("\n");
        //     print("gB        : ");print(gB       );print("\n");
        //     print("gC        : ");print(gC       );print("\n");
        //     // clang-format on
        // }

        MainLoop mainLoop;
        mainLoop(gA, gB, gC, smem);
    }
};

template <typename Operator>
__global__ void kernel_mma(typename Operator::Params params) {
    extern __shared__ char smem[];
    Operator               op;
    op(params, smem);
}

template <typename Kernel_>
struct KernelAdapter {
    using Operator        = Kernel_;
    using Loop            = typename Operator::MainLoop;
    using Params_         = typename Operator::Params;
    using ProblemShapeMNK = typename Operator::ProblemShapeMNK;
    using TA              = typename Loop::TA;
    using TB              = typename Loop::TB;
    using TC              = typename Loop::TC;
    using StrideA         = typename Loop::StrideA;
    using StrideB         = typename Loop::StrideB;
    using StrideC         = typename Loop::StrideC;

    StrideA strideA;
    StrideB strideB;
    StrideC strideC;

    Params_ params_;

    void initialize(ProblemShapeMNK problem_size, TA const *d_A, TB const *d_B, TC *d_C) {

        auto M = cute::size<0>(problem_size);
        auto N = cute::size<1>(problem_size);
        auto K = cute::size<2>(problem_size);

        strideA = make_cute_packed_stride(StrideA{}, make_shape(M, K));
        strideB = make_cute_packed_stride(StrideB{}, make_shape(N, K));
        strideC = make_cute_packed_stride(StrideC{}, make_shape(M, N));

        auto args =
          typename Operator::Arguments{problem_size, {d_A, d_B, d_C, strideA, strideB, strideC}};

        params_ = Operator::to_underlying_arguments(args);
    }

    void run(ProblemShapeMNK problem_size, TA const *d_A, TB const *d_B, TC *d_C) {
        initialize(problem_size, d_A, d_B, d_C);
        static int constexpr SHARED_SIZE = sizeof(typename Loop::SharedStorage{});
        dim3 const block                 = Operator::get_block_shape();
        dim3 const grid                  = Operator::get_grid_shape(params_);
        kernel_mma<Operator><<<grid, block, SHARED_SIZE>>>(params_);
    }
};

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
    const uint TM = 8;
    const uint TN = 4;

    // execution configuration
    dim3 block(N_THREADS);
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    cudaMemset(d_c, 0, c.numel() * c.element_size());

    // launch the kernel
    warpTile<BM, BN, BK, WM, WN, WNITER, TM, TN, N_THREADS>
      <<<grid, block>>>(d_a, d_b, d_c, M, N, K);

    // copy the result
    cudaMemcpy(c.data_ptr(), d_c, c.numel() * c.element_size(), cudaMemcpyDeviceToHost);

    // check the answer
    std::cout << (c.allclose(a.matmul(b)) ? "SUCCESS" : "FAIL") << std::endl;
}

template <typename TA, typename TB, typename TC>
void host_warpTiling_using_cute(
  at::Tensor &a, at::Tensor &b, at::Tensor &c, TA *d_a, TB *d_b, TC *d_c, int M, int N, int K) {

    auto problemShapeMNK = make_shape(M, N, K);
    using Config = GemmConfig<TA, TC, _16, SM80_16x8x8_F32TF32TF32F32_TN, GenRowMajor, GenColMajor>;
    using Loop   = MainLoop<Config>;
    using Operator = KernelOperator<Shape<int, int, int>, Loop>;
    using Adapter  = KernelAdapter<Operator>;
    Adapter ap;

    cudaMemset(d_c, 0, c.numel() * c.element_size());
    ap.run(problemShapeMNK, d_a, d_b, d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data_ptr(), d_c, c.numel() * c.element_size(), cudaMemcpyDeviceToHost);

    auto cpu_ans = (a.matmul(b.mT()));

    std::cout << (c.allclose(cpu_ans, 1e-02) ? "MMA Success" : "MMA Failed") << std::endl;
    // std::cout << cpu_ans << std::endl;
    // std::cout << c << std::endl;
    // std::cout << (c.isclose(cpu_ans, 1e-02)) << std::endl;
}

int main(int argc, char *argv[]) {
    int M{128};
    int N{128};
    int K{16};
    if (argc >= 2) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    auto A = at::rand({M, K}, at::TensorOptions().dtype(at::kFloat));
    auto B = at::rand({N, K}, at::TensorOptions().dtype(at::kFloat));
    auto C = at::rand({M, N}, at::TensorOptions().dtype(at::kFloat));

    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, A.numel() * A.element_size());
    cudaMalloc((void **)&d_B, B.numel() * B.element_size());
    cudaMalloc((void **)&d_C, C.numel() * C.element_size());

    cudaMemcpy(d_A, A.data_ptr(), A.numel() * A.element_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data_ptr(), B.numel() * B.element_size(), cudaMemcpyHostToDevice);

    // host_warpTiling(A, B, C, d_A, d_B, d_C, M, N, K);
    host_warpTiling_using_cute(A, B, C, d_A, d_B, d_C, M, N, K);

    // free the memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
}