cmake_minimum_required(VERSION 3.9)

SET(PROJECT_NAME dijkstra)

SET(ENABLE_CUDA 0)

if (ENABLE_CUDA)
    project(${PROJECT_NAME} LANGUAGES CXX CUDA)
else()
    project(${PROJECT_NAME} LANGUAGES CXX)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include_directories(".")

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)
# set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

if (ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})

    # Target for CUDA part
    add_library(${PROJECT_NAME}_cuda STATIC src/gpu/dijkstra.cu src/gpu/Utilities.cu)
    target_compile_features(${PROJECT_NAME}_cuda PUBLIC cxx_std_11)
    set_target_properties(${PROJECT_NAME}_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    CUDA_ADD_CUBLAS_TO_TARGET(${PROJECT_NAME}_cuda)
endif()

# Target for main executable
add_executable(${PROJECT_NAME} src/main.cpp src/parallel.cpp src/parallel_cl.cpp src/parallel_omp.cpp src/sequential.cpp common/graph.cpp src/parallel_acc.cpp)
target_link_libraries(${PROJECT_NAME} ${OPENCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
configure_file(src/gpu/dijkstra.cl ${CMAKE_CURRENT_BINARY_DIR}/dijkstra.cl COPYONLY)
configure_file(misc/plot.gp ${CMAKE_CURRENT_BINARY_DIR}/plot.gp COPYONLY)
if (ENABLE_CUDA)
    target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_cuda)
endif()

find_package(OpenACC)
if (OpenACC_CXX_FOUND)
    message("OpenACC support is enabled")
    target_compile_options(${PROJECT_NAME} PRIVATE ${OpenACC_CXX_FLAGS})
endif()

target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall)
target_compile_options(${PROJECT_NAME} PRIVATE -Wextra)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)

# Print shortest paths as part of all the functions
target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_PATH_PRINT=0)
# Print results
target_compile_definitions(${PROJECT_NAME} PRIVATE PRINT_RESULTS=0)
target_compile_definitions(${PROJECT_NAME} PRIVATE PRINT_GRAPH_DATA=0)
target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_CUDA=${ENABLE_CUDA})

# algorithm parameters
# target_compile_definitions(${PROJECT_NAME} PRIVATE TOTAL_VERTICES=1024)
# target_compile_definitions(${PROJECT_NAME} PRIVATE MAX_NEIGHBOUR_VERTICES=512)
