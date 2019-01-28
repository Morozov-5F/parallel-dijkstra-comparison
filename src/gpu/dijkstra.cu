#include "src/dijkstra.hpp"

#include "Utilities.cuh"

#define NUM_ASYNCHRONOUS_ITERATIONS 20  // Number of async loop iterations before attempting to read results back

#define BLOCK_SIZE 16

bool allFinalizedVertices(std::vector<bool> &finalizedVertices, int numVertices) {

    for (int i = 0; i < numVertices; i++)  if (finalizedVertices[i] == true) { return false; }

    return true;
}

__global__ void initializeArrays(bool * __restrict__ d_finalizedVertices, float* __restrict__ d_shortestDistances, float* __restrict__ d_updatingShortestDistances,
                                 const int sourceVertex, const int numVertices) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (sourceVertex == tid) {

            d_finalizedVertices[tid]            = true;
            d_shortestDistances[tid]            = 0.f;
            d_updatingShortestDistances[tid]    = 0.f; }

        else {

            d_finalizedVertices[tid]            = false;
            d_shortestDistances[tid]            = FLT_MAX;
            d_updatingShortestDistances[tid]    = FLT_MAX;
        }
    }
}

__global__  void Kernel1(const int * __restrict__ vertexArray, const int* __restrict__ edgeArray,
                         const float * __restrict__ weightArray, bool * __restrict__ finalizedVertices, float* __restrict__ shortestDistances,
                         float * __restrict__ updatingShortestDistances, const int numVertices, const int numEdges) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (finalizedVertices[tid] == true) {

            finalizedVertices[tid] = false;

            int edgeStart = vertexArray[tid], edgeEnd;

            if (tid + 1 < (numVertices)) edgeEnd = vertexArray[tid + 1];
            else                         edgeEnd = numEdges;

            for (int edge = edgeStart; edge < edgeEnd; edge++) {
                int nid = edgeArray[edge];
                atomicMin(&updatingShortestDistances[nid], shortestDistances[tid] + weightArray[edge]);
            }
        }
    }
}

__global__  void Kernel2(const int * __restrict__ vertexArray, const int * __restrict__ edgeArray, const float* __restrict__ weightArray,
                         bool * __restrict__ finalizedVertices, float* __restrict__ shortestDistances, float* __restrict__ updatingShortestDistances,
                         const int numVertices) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (shortestDistances[tid] > updatingShortestDistances[tid]) {
            shortestDistances[tid] = updatingShortestDistances[tid];
            finalizedVertices[tid] = true; }

        updatingShortestDistances[tid] = shortestDistances[tid];
    }
}

std::vector<float> dijkstra_cuda(const Graph &graph, int sourceVertex)
{

    // --- Create device-side adjacency-list, namely, vertex array Va, edge array Ea and weight array Wa from G(V,E,W)
    int     *d_vertexArray;         gpuErrchk(cudaMalloc(&d_vertexArray, sizeof(int)   * graph.vertex_array.size()));
    int     *d_edgeArray;           gpuErrchk(cudaMalloc(&d_edgeArray,   sizeof(int)   * graph.edge_array.size()));
    float   *d_weightArray;         gpuErrchk(cudaMalloc(&d_weightArray, sizeof(float) * graph.weight_array.size()));

    // --- Copy adjacency-list to the device
    gpuErrchk(cudaMemcpy(d_vertexArray, graph.vertex_array.data(), sizeof(int) * graph.vertex_array.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_edgeArray,   graph.edge_array.data(), sizeof(int) * graph.edge_array.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_weightArray, graph.weight_array.data(), sizeof(float) * graph.weight_array.size(), cudaMemcpyHostToDevice));

    // --- Create mask array Ma, cost array Ca and updating cost array Ua of size V
    bool    *d_finalizedVertices;           gpuErrchk(cudaMalloc(&d_finalizedVertices, sizeof(bool)   * graph.vertex_array.size()));
    float   *d_shortestDistances;           gpuErrchk(cudaMalloc(&d_shortestDistances, sizeof(float) * graph.vertex_array.size()));
    float   *d_updatingShortestDistances;   gpuErrchk(cudaMalloc(&d_updatingShortestDistances, sizeof(float) * graph.vertex_array.size()));

    bool * h_finalizedVertices = new bool[graph.vertex_array.size()];

    // --- Initialize mask Ma to false, cost array Ca and Updating cost array Ua to \u221e
    initializeArrays <<<iDivUp(graph.vertex_array.size(), BLOCK_SIZE), BLOCK_SIZE >>>(d_finalizedVertices, d_shortestDistances,
                               d_updatingShortestDistances, sourceVertex, graph.vertex_array.size());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // --- Read mask array from device -> host
    gpuErrchk(cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * graph.vertex_array.size(), cudaMemcpyDeviceToHost));

    while (std::all_of(h_finalizedVertices, h_finalizedVertices + graph.vertex_array.size(), [](bool x) { return !x; })) {

        // --- In order to improve performance, we run some number of iterations without reading the results.  This might result
        //     in running more iterations than necessary at times, but it will in most cases be faster because we are doing less
        //     stalling of the GPU waiting for results.
        for (int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++) {

            Kernel1 <<<iDivUp(graph.vertex_array.size(), BLOCK_SIZE), BLOCK_SIZE >>>(d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices, d_shortestDistances,
                                                            d_updatingShortestDistances, graph.vertex_array.size(), graph.edge_array.size());
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
            Kernel2 <<<iDivUp(graph.vertex_array.size(), BLOCK_SIZE), BLOCK_SIZE >>>(d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices, d_shortestDistances, d_updatingShortestDistances,
                                                            graph.vertex_array.size());
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }

        gpuErrchk(cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * graph.vertex_array.size(), cudaMemcpyDeviceToHost));
    }

    // --- Copy the result to host
    std::vector<float> shortest_distance(graph.vertex_array.size());
    gpuErrchk(cudaMemcpy(shortest_distance.data(), d_shortestDistances, sizeof(float) * shortest_distance.size(), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_vertexArray));
    gpuErrchk(cudaFree(d_edgeArray));
    gpuErrchk(cudaFree(d_weightArray));
    gpuErrchk(cudaFree(d_finalizedVertices));
    gpuErrchk(cudaFree(d_shortestDistances));
    gpuErrchk(cudaFree(d_updatingShortestDistances));

    return shortest_distance;
}