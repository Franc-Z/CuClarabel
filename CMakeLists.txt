cmake_minimum_required(VERSION 3.18)
project(Clarabel LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Target executable
add_executable(clarabel ${SRC_FILES})

# CUDA libraries
find_package(CUDA REQUIRED)
target_link_libraries(clarabel PRIVATE ${CUDA_LIBRARIES} cudart cusparse cublas cusolver)

# Thrust library
find_package(Thrust REQUIRED)
target_link_libraries(clarabel PRIVATE Thrust::Thrust)

# Compiler flags
target_compile_options(clarabel PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Werror>)
target_compile_options(clarabel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -Wall,-Wextra,-Werror>)

# Build and clean targets
add_custom_target(build ALL DEPENDS clarabel)
add_custom_target(clean-all COMMAND ${CMAKE_COMMAND} --build . --target clean)
