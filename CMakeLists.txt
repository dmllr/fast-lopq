cmake_minimum_required(VERSION 3.9)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(fast-lopq LANGUAGES C CXX CUDA)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s") # strip
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(GNUInstallDirs)

include("cmake/ext-blaze.cmake")
include("cmake/ext-protobuf.cmake")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -W -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -W -Wall -Wextra")

add_subdirectory("fast-lopq")
add_subdirectory("fast-lopq-test")
