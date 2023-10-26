cmake_minimum_required(VERSION 3.26)
project(GEMM CUDA CXX)
set(CMAKE_PREFIX_PATH /root/miniconda3/envs/nlp/lib/python3.11/site-packages/torch/share/cmake)


find_package(CUDAToolkit REQUIRED)
set(TORCH_CUDA_ARCH_LIST 8.9)
find_package(Torch REQUIRED)

macro(add_executable_for_torch target_name)
    add_executable(${target_name} "./csrc/${target_name}.cu")
    set_target_properties(${target_name} PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 89)
    target_compile_options(${target_name} PRIVATE -O3 -lineinfo )
    target_link_libraries(${target_name} "${TORCH_LIBRARIES}")
endmacro()

# gemmCompLoad
add_executable_for_torch(naiveSGEMM)
add_executable_for_torch(coalesceMemoryAccess)
add_executable_for_torch(sharedMemBlockSGEMM)
add_executable_for_torch(sgemm1DBlockTiling)
add_executable_for_torch(sgemm2DBlockTiling)
add_executable_for_torch(sgemmVectorize)
add_executable_for_torch(loadToSMEM)
add_executable_for_torch(loadToReg)
add_executable_for_torch(warpTiling)