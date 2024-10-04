cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# 选择适合自己 gpu 的架构
set(CMAKE_CUDA_ARCHITECTURES 86)

# CXX 代表 C++ 
project(MyProj LANGUAGES CXX CUDA)

find_package(Python REQUIRED)
find_package(Torch REQUIRED)

find_library(
    TORCH_PYTHON_LIBRARY
    torch_python
    PATHS
    "${TORCH_INSTALL_PREFIX}/lib"
)

# 导入头文件位置
include_directories(include)
# 将 kernel 目录中所有的 C/C++/CUDA 文件添加到 SRCS 变量中
file(GLOB SRCS kernel/*.c*)

# 生成共享库 libadd2.so
add_library(add2 SHARED ${SRCS})
target_link_libraries(add2 "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
