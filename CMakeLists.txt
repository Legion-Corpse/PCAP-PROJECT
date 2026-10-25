cmake_minimum_required(VERSION 3.24)
project(ooc_rtnn
  VERSION 1.0.0
  DESCRIPTION "Out-of-core RT-core accelerated fixed-radius neighbor search"
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Turing (sm_75) is the oldest architecture with RT cores and the oldest CUDA 13 supports.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

option(OOC_OPTIX_IR "Compile device programs to OptiX-IR (OFF = PTX)" ON)
set(OOC_OPTIX_VERSION "v9.1.0" CACHE STRING "Tag of github.com/NVIDIA/optix-dev to fetch when OptiX_INSTALL_DIR is unset")
set(OptiX_INSTALL_DIR "$ENV{OptiX_INSTALL_DIR}" CACHE PATH "Path to an OptiX SDK (directory that contains include/optix.h)")

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# ---------------------------------------------------------------------------
# OptiX headers. OptiX is header-only on the host side: the implementation
# ships with the NVIDIA display driver and is loaded by optixInit().
# ---------------------------------------------------------------------------
if(OptiX_INSTALL_DIR AND EXISTS "${OptiX_INSTALL_DIR}/include/optix.h")
  set(OPTIX_INCLUDE_DIR "${OptiX_INSTALL_DIR}/include")
else()
  include(FetchContent)
  FetchContent_Declare(optix_dev
    GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
    GIT_TAG        ${OOC_OPTIX_VERSION}
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(optix_dev)
  set(OPTIX_INCLUDE_DIR "${optix_dev_SOURCE_DIR}/include")
endif()
message(STATUS "OptiX headers: ${OPTIX_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Common compile settings
# ---------------------------------------------------------------------------
add_library(ooc_options INTERFACE)
target_compile_definitions(ooc_options INTERFACE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
target_include_directories(ooc_options INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OPTIX_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS})
if(MSVC)
  # /fp:precise (the default) never contracts a*b+c into an FMA, which keeps the
  # host distance test bit-identical to the device one (see common/vec_math.h).
  target_compile_options(ooc_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:__cplusplus /Zc:preprocessor /utf-8 /fp:precise /openmp:llvm>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8 -Xcompiler=/Zc:__cplusplus -Xcompiler=/Zc:preprocessor>)
else()
  target_compile_options(ooc_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -ffp-contract=off>)
endif()
target_compile_options(ooc_options INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda -lineinfo>)

# ---------------------------------------------------------------------------
# OptiX device programs -> OptiX-IR/PTX -> embedded byte array
# ---------------------------------------------------------------------------
set(OOC_DEVICE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/device_programs.cu)
if(OOC_OPTIX_IR)
  set(OOC_DEVICE_EXT optixir)
  set(OOC_DEVICE_FLAG -optix-ir)
else()
  set(OOC_DEVICE_EXT ptx)
  set(OOC_DEVICE_FLAG -ptx)
endif()
set(OOC_DEVICE_OUT ${CMAKE_CURRENT_BINARY_DIR}/device_programs.${OOC_DEVICE_EXT})
set(OOC_DEVICE_EMBED ${CMAKE_CURRENT_BINARY_DIR}/generated/device_programs_embed.cpp)
if(MSVC)
  set(OOC_NVCC_HOST_FLAGS -ccbin "${CMAKE_CXX_COMPILER}")
endif()
add_custom_command(
  OUTPUT ${OOC_DEVICE_OUT}
  COMMAND ${CMAKE_CUDA_COMPILER} ${OOC_DEVICE_FLAG} ${OOC_NVCC_HOST_FLAGS}
          -arch=compute_75 -std=c++17 -rdc=true -lineinfo -Wno-deprecated-gpu-targets
          -I${OPTIX_INCLUDE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/src
          ${OOC_DEVICE_SRC} -o ${OOC_DEVICE_OUT}
  DEPENDS ${OOC_DEVICE_SRC}
          ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/launch_params.h
          ${CMAKE_CURRENT_SOURCE_DIR}/src/common/vec_math.h
  COMMENT "Compiling OptiX device programs (${OOC_DEVICE_EXT})"
  VERBATIM)
add_custom_command(
  OUTPUT ${OOC_DEVICE_EMBED}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${OOC_DEVICE_OUT} -DOUTPUT=${OOC_DEVICE_EMBED} -DSYMBOL=g_devicePrograms
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
  DEPENDS ${OOC_DEVICE_OUT} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
  COMMENT "Embedding OptiX device programs"
  VERBATIM)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(ooc_core STATIC
  src/common/dataset.cpp
  src/common/cli.cpp
  src/common/host_utils.cpp
  src/cpu/cpu_search.cpp
  src/cpu/verify.cpp
  src/cuda/kernels.cu
  src/cuda/grid_search.cu
  src/rt/optix_engine.cpp
  src/rt/incore_search.cpp
  src/ooc/kdtree.cpp
  src/ooc/tile_store.cpp
  src/ooc/planner.cpp
  src/ooc/gpu_order.cu
  src/ooc/executor.cpp
  src/ooc/cost_model.cpp
  src/ooc/ooc_search.cpp
  ${OOC_DEVICE_EMBED})
target_link_libraries(ooc_core PUBLIC ooc_options CUDA::cudart OpenMP::OpenMP_CXX)
set_target_properties(ooc_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(ooc_rtnn src/app/main.cpp)
target_link_libraries(ooc_rtnn PRIVATE ooc_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
  add_executable(ooc_tests tests/unit_tests.cpp tests/gpu_tests.cpp tests/perf_tests.cpp tests/test_main.cpp)
  target_link_libraries(ooc_tests PRIVATE ooc_core)
  add_test(NAME unit COMMAND ooc_tests --filter cpu)
  add_test(NAME gpu  COMMAND ooc_tests --filter gpu)
endif()
