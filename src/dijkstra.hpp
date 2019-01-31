#pragma once

#include <vector>
#include <cfloat>

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include "common/graph.hpp"


typedef enum ocl_init_result_e
{
    OCL_INIT_SUCCESS,
    OCL_INIT_GPU_ONLY,
    OCL_INIT_CPU_ONLY,
    OCL_INIT_NO_DEVICES,
    OCL_INIT_NO_PLATFORM,
} ocl_init_result_t;

std::vector<float> dijkstra_sequential(const Graph &graph,
                                       int source_vertex);

std::vector<float> dijkstra_omp(const Graph &graph,
                                int source_vertex);

ocl_init_result_t dijkstra_init_contexts(cl_context &gpu_context, cl_context &cpu_context);

std::vector<float> dijkstra_opencl(const Graph &graph, int source_vertex, cl_context &opencl_context);

std::vector<float> dijkstra_cuda(const Graph &graph, int sourceVertex);

std::vector<float> dijkstra_acc(const __restrict Graph &graph, int source_vertex);