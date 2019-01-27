#include <iostream>
#include <iomanip>

#include <omp.h>

#include "src/dijkstra.hpp"

void print_results(const std::string& msg, const std::vector<float> &res, int source_vertex)
{
    #if PRINT_RESULTS != 0
    std::cout << std::endl << msg << std::endl;
    for (auto k = 0ULL; k < res.size(); k++)
    {
        std::cout << "From vertex " << source_vertex << " to vertex " << k << " = " << res[k] << std::endl;
    }
    #endif
}

void print_duration(const std::string& name, std::chrono::time_point<std::chrono::high_resolution_clock> start,
                    std::chrono::time_point<std::chrono::high_resolution_clock> finish)
{
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << std::fixed << std::setw( 11 ) << std::setprecision( 6 ) << "Duration of " << name << " algorithm: " << elapsed.count() << " seconds" << std::endl;
}

int main() {

    // --- Number of graph vertices
    int num_vertices = 1024 * 5;

    // --- Number of edges per graph vertex
    int neighbors_per_vertex = 128;

    // --- Source vertex
    int sourceVertex = 0;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish;

    std::cout << "Dijkstra's algorithm test, " << num_vertices << " vertices, " << neighbors_per_vertex << " neighbors per vertex" << std::endl;
    cl_context gpu_context, cpu_context;
    bool gpu_found = false, cpu_found = false;
    auto init_res = dijkstra_init_contexts(gpu_context, cpu_context);

    switch (init_res)
    {
        case OCL_INIT_NO_DEVICES:
            std::cerr << "No OpenCL devices found" << std::endl;
            exit(1);
        case OCL_INIT_NO_PLATFORM:
            std::cerr << "No OpenCL platform found" << std::endl;
            exit(1);
        case OCL_INIT_CPU_ONLY:
            cpu_found = true;
            std::cerr << "Suitable GPU for OpenCL was not found on the system" << std::endl;
            break;
        case OCL_INIT_GPU_ONLY:
            gpu_found = true;
            std::cerr << "Suitable CPU for OpenCL was not found on the system" << std::endl;
            break;
        case OCL_INIT_SUCCESS:
            gpu_found = cpu_found = true;
            break;
    }

      // --- Allocate memory for arrays
    std::cout << "Generating graph...";
    std::cout.flush();
    Graph graph(num_vertices, neighbors_per_vertex);
    std::cout << "\tDone" << std::endl;

    // --- Displaying the adjacency list and constructing the adjacency matrix
    #if PRINT_GRAPH_DATA != 0
        graph.PrintVertexData();
        graph.DisplayWeightMatrix();
    #endif

    // --- Running sequential Dijkstra on the CPU
    start = std::chrono::high_resolution_clock::now();
    auto shortest_distances_seq = dijkstra_sequential(graph, sourceVertex);
    finish = std::chrono::high_resolution_clock::now();
    print_results("CPU results", shortest_distances_seq, sourceVertex);
    print_duration("CPU", start, finish);
    shortest_distances_seq.clear();

    // --- Running parallel Dijkstra on the CPU with OMP
    start = std::chrono::high_resolution_clock::now();
    auto shortest_distances_omp = dijkstra_omp(graph, sourceVertex);
    finish = std::chrono::high_resolution_clock::now();
    print_results("CPU results (OpenMP)", shortest_distances_omp, sourceVertex);
    print_duration("CPU (OpenMP)", start, finish);
    shortest_distances_omp.clear();

    // --- Running parallel Dijkstra on the CPU with OpenCL
    if (cpu_found)
    {
        start = std::chrono::high_resolution_clock::now();
        auto shortest_distances_ocl_cpu = dijkstra_opencl(graph, sourceVertex, cpu_context);
        finish = std::chrono::high_resolution_clock::now();
        print_results("CPU results (OpenCL)", shortest_distances_ocl_cpu, sourceVertex);
        print_duration("CPU (OpenCL)", start, finish);
        shortest_distances_ocl_cpu.clear();
    }

    // --- Running parallel Dijkstra on the GPU with OpenCL
    if (gpu_found)
    {
        start = std::chrono::high_resolution_clock::now();
        auto shortest_distances_ocl_gpu = dijkstra_opencl(graph, sourceVertex, gpu_context);
        finish = std::chrono::high_resolution_clock::now();
        print_results("GPU results (OpenCL)", shortest_distances_ocl_gpu, sourceVertex);
        print_duration("GPU (OpenCL)", start, finish);
        shortest_distances_ocl_gpu.clear();
    }


    // --- Allocate space for the h_shortestDistancesGPU
    // float *h_shortestDistancesGPU = (float*)malloc(sizeof(float) * graph.numVertices);
    // dijkstraGPU(&graph, sourceVertex, h_shortestDistancesGPU);

    // printf("\nGPU results\n");
    // for (int k = 0; k < numVertices; k++) printf("From vertex %i to vertex %i = %f\n", sourceVertex, k, h_shortestDistancesGPU[k]);

    return 0;
}