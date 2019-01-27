cmake_minimum_required(VERSION 2.9)

SET(PROJECT_NAME dijkstra)

project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include_directories(".")

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_executable(${PROJECT_NAME} src/main.cpp src/parallel.cpp src/parallel_cl.cpp src/parallel_omp.cpp src/sequential.cpp common/graph.cpp)
target_link_libraries(${PROJECT_NAME} ${OPENCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenMP_CXX_LIBRARIES})
configure_file(src/gpu/dijkstra.cl ${CMAKE_CURRENT_BINARY_DIR}/dijkstra.cl COPYONLY)

target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
target_compile_options(${PROJECT_NAME} PRIVATE -Wno-unused-command-line-argument)
target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall)
target_compile_options(${PROJECT_NAME} PRIVATE -Wextra)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)

# Print shortest paths as part of all the functions
target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_PATH_PRINT=0)
# Print results
target_compile_definitions(${PROJECT_NAME} PRIVATE PRINT_RESULTS=0)
target_compile_definitions(${PROJECT_NAME} PRIVATE PRINT_GRAPH_DATA=0)
target_compile_definitions(${PROJECT_NAME} PRIVATE OMP_THREADS=4)