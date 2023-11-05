cmake_minimum_required(VERSION 3.26)
project(GEMM CUDA CXX)
set(CMAKE_PREFIX_PATH /root/miniconda3/envs/nlp/lib/python3.11/site-packages/torch/share/cmake)


find_package(CUDAToolkit REQUIRED)
set(TORCH_CUDA_ARCH_LIST 8.9)
find_package(Torch REQUIRED)

file(GLOB CUDA_SOURCE_FILES "./csrc/*.cu")
foreach(CUDA_SOURCE_FILE ${CUDA_SOURCE_FILES})
    # Extract the filename without the extension
    get_filename_component(EXECUTABLE_NAME ${CUDA_SOURCE_FILE} NAME_WE)  

    # Create an executable for each source file
    add_executable(${EXECUTABLE_NAME} ${CUDA_SOURCE_FILE})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 89)
    # target_compile_options(${EXECUTABLE_NAME} PRIVATE -G -g)
    target_compile_options(${EXECUTABLE_NAME} PRIVATE -O3 -lineinfo )
    target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")
endforeach()