#include "cute/algorithm/copy.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/numeric/half.hpp"
#include "cute/numeric/integer_sequence.hpp"
#include "cute/tensor.hpp"
#include "cute/int_tuple.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/tuple_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits.hpp"
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
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

#define CUDA_CHECK(status)                                                                         \
    {                                                                                              \
        cudaError_t error = status;                                                                \
        if (error != cudaSuccess) {                                                                \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)                      \
                      << " at line: " << __LINE__ << std::endl;                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

using namespace cute;
using X = Underscore;

// Utility
constexpr int exponent(int result) {
    int exponent = 0;
    while (result > 1) {
        result = result >> 1;               // Right shift the result by 1 bit
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
    using type = Stride<int64_t, Int<1>>;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<GenColMajor> {
    using type = Stride<Int<1>, int64_t>;
};

template <class L>
struct TagToStrideB {
    using type = L;
};

template <>
struct TagToStrideB<GenRowMajor> {
    using type = Stride<Int<1>, int64_t>;
};

template <>
struct TagToStrideB<GenColMajor> {
    using type = Stride<int64_t, Int<1>>;
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
    static int constexpr Stages   = 3;
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
    using TA                    = typename Config::TA;
    using TB                    = typename Config::TB;
    using TC                    = typename Config::TC;
    using StrideA               = typename Config::StrideA;
    using StrideB               = typename Config::StrideB;
    using StrideC               = typename Config::StrideC;
    using TiledMMA_             = typename Config::TiledMMA;
    using TileShapeMNK          = typename Config::TileShape;
    static int constexpr Stages = Config::Stages;

    using Layout_SA      = decltype(tile_to_shape(
      typename Config::OperandA_::SmemLayout{},
      make_shape(get<0>(TileShapeMNK{}), get<2>(TileShapeMNK{}), Int<Stages>{})));
    using Layout_SB      = decltype(tile_to_shape(
      typename Config::OperandB_::SmemLayout{},
      make_shape(get<1>(TileShapeMNK{}), get<2>(TileShapeMNK{}), Int<Stages>{})));
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

    template <typename TensorA, typename TensorB, typename TensorC, typename KTileIterator>
    CUTE_DEVICE void operator()(
      TensorA       gA,
      TensorB       gB,
      TensorC       gC,
      char         *smem,
      KTileIterator k_tile_iter,
      int           k_tile_count,
      uint          threadId) {
        SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem);

        auto sA = make_tensor(
          make_smem_ptr(storage.smem_a.data()), Layout_SA{});               // (BM,BK,Stages)
        auto sB = make_tensor(
          make_smem_ptr(storage.smem_b.data()), Layout_SB{});               // (BN,BK,Stages)

        GS_TiledCopy_A gs_tiledCopy_A;
        GS_TiledCopy_B gs_tiledCopy_B;
        auto           gs_thr_copy_A = gs_tiledCopy_A.get_thread_slice(threadId);
        auto           gs_thr_copy_B = gs_tiledCopy_B.get_thread_slice(threadId);

        auto tAgA = gs_thr_copy_A.partition_S(gA);               // (ACpy,RestM,RestK,k_tile)
        auto tAsA = gs_thr_copy_A.partition_D(sA);               // (ACpy,RestM,RestK,Stages)

        auto tBgB = gs_thr_copy_B.partition_S(gB);               // (ACpy,RestN,RestK,k_tile)
        auto tBsB = gs_thr_copy_B.partition_D(sB);               // (ACpy,RestN,RestK,Stages)

        // Prefetch
        // Write all but not last buffer
        CUTE_UNROLL
        for (int k_pipe{0}; k_pipe < Stages - 1; ++k_pipe) {
            copy(gs_tiledCopy_A, tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, k_pipe));
            copy(gs_tiledCopy_B, tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) {
                ++k_tile_iter;
            }
        }

        TiledMMA_ tiledMAA;
        auto      thr_mma = tiledMAA.get_thread_slice(threadId);
        auto tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));               // (mma,restM,restK)
        auto tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));               // (mma,restN,restK)
        auto tCrC = thr_mma.partition_C(gC);                                 // (mma,restM,restN)

        SR_TiledCopy_A sr_tiledCopy_A;
        SR_TiledCopy_B sr_tiledCopy_B;
        auto           sr_thr_copy_A = sr_tiledCopy_A.get_thread_slice(threadId);
        auto           sr_thr_copy_B = sr_tiledCopy_B.get_thread_slice(threadId);

        auto tCsA      = sr_thr_copy_A.partition_S(sA);               // (Cpy,RestM,RestK,Stages)
        auto tCrA_view = sr_thr_copy_A.retile_D(tCrA);                // (Cpy,RestM,RestK)
        auto tCsB      = sr_thr_copy_B.partition_S(sB);               // (Cpy,RestN,RestK,Stages)
        auto tCrB_view = sr_thr_copy_B.retile_D(tCrB);                // (Cpy,RestN,RestK)

        auto fragC = partition_fragment_C(tiledMAA, take<0, 2>(TileShapeMNK{}));
        clear(fragC);

        // if (thread0()) {
        //     // clang-format off
        //         print("sA            : ");print(sA           );print("\n");
        //         print("sB            : ");print(sB           );print("\n");
        //         print("gs_thr_copy_A : ");print(gs_thr_copy_A);print("\n");
        //         print("tAgA          : ");print(tAgA         );print("\n");
        //         print("tAsA          : ");print(tAsA         );print("\n");
        //         print("gs_thr_copy_B : ");print(gs_thr_copy_B);print("\n");
        //         print("tBgB          : ");print(tBgB         );print("\n");
        //         print("tBsB          : ");print(tBsB         );print("\n");
        //         print("thr_mma       : ");print(thr_mma      );print("\n");
        //         print("tCrA          : ");print(tCrA         );print("\n");
        //         print("tCrB          : ");print(tCrB         );print("\n");
        //         print("tCrC          : ");print(tCrC         );print("\n");
        //         print("sr_thr_copy_A : ");print(sr_thr_copy_A);print("\n");
        //         print("tCsA          : ");print(tCsA         );print("\n");
        //         print("tCrA_view     : ");print(tCrA_view    );print("\n");
        //         print("sr_thr_copy_B : ");print(sr_thr_copy_B);print("\n");
        //         print("tCsB          : ");print(tCsB         );print("\n");
        //         print("tCrB_view     : ");print(tCrB_view    );print("\n");
        //         print("fragC         : ");print(fragC        );print("\n");
        //     // clang-format on
        // }

        // current read and write index
        int smem_pipe_read  = 0;
        int smem_pipe_write = Stages - 1;

        Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
        Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

        auto k_inner_max = size<2>(tCrA);

        if (k_inner_max > 1) {
            // wait until the first buffer is loaded
            cp_async_wait<Stages - 2>();
            __syncthreads();

            copy(sr_tiledCopy_A, tCsA_p(_, _, Int<0>{}), tCrA_view(_, _, Int<0>{}));
            copy(sr_tiledCopy_B, tCsB_p(_, _, Int<0>{}), tCrB_view(_, _, Int<0>{}));
        }

        CUTE_NO_UNROLL
        for (; k_tile_count > -(Stages - 1); --k_tile_count) {

            for_each(make_int_sequence<k_inner_max>{}, [&](auto k_inner) {
                // switch next buffer when k is last in inner product
                if (k_inner == k_inner_max - 1) {
                    tCsA_p = tCsA(_, _, _, smem_pipe_read);
                    tCsB_p = tCsB(_, _, _, smem_pipe_read);

                    // sync threads before use it
                    cp_async_wait<Stages - 2>();
                    __syncthreads();
                }

                // load A,B from smem to rmem for outer product
                auto k_inner_next = (k_inner + Int<1>{}) % k_inner_max;
                copy(sr_tiledCopy_A, tCsA_p(_, _, k_inner_next), tCrA_view(_, _, k_inner_next));
                copy(sr_tiledCopy_B, tCsB_p(_, _, k_inner_next), tCrB_view(_, _, k_inner_next));

                // before compute write to last buffer
                if (k_inner == 0) {
                    copy(
                      gs_tiledCopy_A, tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, smem_pipe_write));
                    copy(
                      gs_tiledCopy_B, tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, smem_pipe_write));
                    cp_async_fence();
                    if (k_tile_count > 0) {
                        ++k_tile_iter;
                    }

                    smem_pipe_write = smem_pipe_read;
                    ++smem_pipe_read;
                    smem_pipe_read = (smem_pipe_read == Stages) ? 0 : smem_pipe_read;
                }
                gemm(tiledMAA, fragC, tCrA(_, _, k_inner), tCrB(_, _, k_inner), fragC);
            });
        }
        copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, fragC, tCrC);
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
        //         print("mA        : ");print(mA       );print("\n");
        //         print("mB        : ");print(mB       );print("\n");
        //         print("mC        : ");print(mC       );print("\n");
        //         print("blk_shape : ");print(blk_shape);print("\n");
        //         print("blk_coord : ");print(blk_coord);print("\n");
        //         print("gA        : ");print(gA       );print("\n");
        //         print("gB        : ");print(gB       );print("\n");
        //         print("gC        : ");print(gC       );print("\n");
        //     // clang-format on
        // }

        auto k_tile_iter  = make_coord_iterator(shape<2>(gA));
        auto k_tile_count = size<2>(gA);

        MainLoop mainLoop;
        mainLoop(gA, gB, gC, smem, k_tile_iter, k_tile_count, threadIdx.x);
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

        params_       = Operator::to_underlying_arguments(args);
        int smem_size = sizeof(typename Loop::SharedStorage{});

        CUDA_CHECK(cudaFuncSetAttribute(
          kernel_mma<Operator>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    void run(ProblemShapeMNK problem_size, TA const *d_A, TB const *d_B, TC *d_C) {
        initialize(problem_size, d_A, d_B, d_C);
        static int constexpr SHARED_SIZE = sizeof(typename Loop::SharedStorage{});
        dim3 const block                 = Operator::get_block_shape();
        dim3 const grid                  = Operator::get_grid_shape(params_);
        std::cout << "Launch Configuration : \n"
                  << "\tGrid : " << grid << "\tBlock : " << block << "\tSmem : " << SHARED_SIZE
                  << std::endl;
        kernel_mma<Operator><<<grid, block, SHARED_SIZE>>>(params_);
    }
};

template <typename TA, typename TB, typename TC>
void host_warpTiling_using_cute(
  at::Tensor &a, at::Tensor &b, at::Tensor &c, TA *d_a, TB *d_b, TC *d_c, int M, int N, int K) {

    auto problemShapeMNK = make_shape(M, N, K);
    using Config = GemmConfig<TA, TC, _32, SM80_16x8x8_F32TF32TF32F32_TN, GenRowMajor, GenColMajor>;
    using Loop   = MainLoop<Config>;
    using Operator = KernelOperator<Shape<int, int, int>, Loop>;
    using Adapter  = KernelAdapter<Operator>;
    Adapter ap;

    CUDA_CHECK(cudaMemset(d_c, 0, c.numel() * c.element_size()));
    ap.run(problemShapeMNK, d_a, d_b, d_c);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data_ptr(), d_c, c.numel() * c.element_size(), cudaMemcpyDeviceToHost));

    auto cpu_ans = (a.matmul(b.mT()));

    std::cout << (c.allclose(cpu_ans, 1e-02) ? "MMA Success" : "MMA Failed") << std::endl;
    // std::cout << cpu_ans << std::endl;
    // std::cout << c << std::endl;
    // std::cout << (c.isclose(cpu_ans, 1e-02)) << std::endl;
}

int main(int argc, char *argv[]) {
    int M{4096};
    int N{4096};
    int K{4096};
    if (argc >= 2) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    auto A = at::rand({M, K}, at::TensorOptions().dtype(at::kFloat));
    auto B = at::rand({N, K}, at::TensorOptions().dtype(at::kFloat));
    auto C = at::rand({M, N}, at::TensorOptions().dtype(at::kFloat));

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc((void **)&d_A, A.numel() * A.element_size()));
    CUDA_CHECK(cudaMalloc((void **)&d_B, B.numel() * B.element_size()));
    CUDA_CHECK(cudaMalloc((void **)&d_C, C.numel() * C.element_size()));

    CUDA_CHECK(cudaMemcpy(d_A, A.data_ptr(), A.numel() * A.element_size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data_ptr(), B.numel() * B.element_size(), cudaMemcpyHostToDevice));

    host_warpTiling_using_cute(A, B, C, d_A, d_B, d_C, M, N, K);

    // free the memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaDeviceReset());
}