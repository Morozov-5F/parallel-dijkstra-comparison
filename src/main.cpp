#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>


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
#else
    (void)msg;
    (void)res;
    (void)source_vertex;
#endif
}

void print_duration(const std::string& name, std::chrono::time_point<std::chrono::high_resolution_clock> start,
                    std::chrono::time_point<std::chrono::high_resolution_clock> finish,
                    std::ofstream &output_file)
{
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << std::fixed << std::setw( 11 ) << std::setprecision( 6 ) << "Duration of " << name << " algorithm: " << elapsed.count() << " seconds" << std::endl;

    if (output_file.good())
    {
        output_file << std::fixed << std::setw( 11 ) << std::setprecision( 6 ) << " " << elapsed.count();
    }
}

int main() {

    // --- Number of graph vertices
#ifdef TOTAL_VERTICES
    int num_vertices = TOTAL_VERTICES;
#else
    int num_vertices = 1024 * 20;
#endif
    // --- Number of edges per graph vertex
#ifdef MAX_NEIGHBOUR_VERTICES
    int neighbors_per_vertex = MAX_NEIGHBOUR_VERTICES;
#else
    int neighbors_per_vertex = 128;
#endif
    // --- Source vertex
    int sourceVertex = 0;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish;

    std::ofstream out_file("output.dat");

    std::cout << std::fixed << std::setw( 11 ) << std::setprecision( 6 ) << "Dijkstra's algorithm test, " << num_vertices << " vertices, " << neighbors_per_vertex << " neighbors per vertex" << std::endl;
    cl_context gpu_context, cpu_context;
    bool gpu_found = false, cpu_found = false;
    auto init_res = dijkstra_init_contexts(gpu_context, cpu_context);

    switch (init_res)
    {
        case OCL_INIT_NO_DEVICES:
            std::cerr << "No OpenCL devices found" << std::endl;
	    break;
        case OCL_INIT_NO_PLATFORM:
            std::cerr << "No OpenCL platform found" << std::endl;
	    break;
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

#if _OPENACC
    acc_init(acc_device_nvidia);
#endif

#if 0
      // --- Allocate memory for arrays
    std::cout << "Generating graph...";
    std::cout.flush();
    Graph graph(num_vertices, neighbors_per_vertex);
    std::cout << "\tDone" << std::endl;
#endif

    // --- Displaying the adjacency list and constructing the adjacency matrix
    #if PRINT_GRAPH_DATA != 0
        graph.PrintVertexData();
        graph.DisplayWeightMatrix();
    #endif

    for (int i = 1024; i < num_vertices; i += 1024)
    {
        std::cout << "Generating graph with " << i << " vertices and " << i / neighbors_per_vertex << " neighbors per vertex...";
        std::cout.flush();
        Graph graph(i, i / neighbors_per_vertex);
        std::cout << "\tDone" << std::endl;

        out_file << i;

        // --- Running sequential Dijkstra on the CPU
        start = std::chrono::high_resolution_clock::now();
        auto shortest_distances_seq = dijkstra_sequential(graph, sourceVertex);
        finish = std::chrono::high_resolution_clock::now();
        print_results("CPU results", shortest_distances_seq, sourceVertex);
        print_duration("CPU", start, finish, out_file);
        shortest_distances_seq.clear();

        // --- Running parallel Dijkstra on the CPU with OMP
        start = std::chrono::high_resolution_clock::now();
        auto shortest_distances_omp = dijkstra_omp(graph, sourceVertex);
        finish = std::chrono::high_resolution_clock::now();
        print_results("CPU results (OpenMP)", shortest_distances_omp, sourceVertex);
        print_duration("CPU (OpenMP)", start, finish, out_file);
        shortest_distances_omp.clear();

        // --- Running parallel Dijkstra on the CPU with OpenCL
        if (cpu_found)
        {
            start = std::chrono::high_resolution_clock::now();
            auto shortest_distances_ocl_cpu = dijkstra_opencl(graph, sourceVertex, cpu_context);
            finish = std::chrono::high_resolution_clock::now();
            print_results("CPU results (OpenCL)", shortest_distances_ocl_cpu, sourceVertex);
            print_duration("CPU (OpenCL)", start, finish, out_file);
            shortest_distances_ocl_cpu.clear();
        }

        // --- Running parallel Dijkstra on the GPU with OpenCL
        if (gpu_found)
        {
            start = std::chrono::high_resolution_clock::now();
            auto shortest_distances_ocl_gpu = dijkstra_opencl(graph, sourceVertex, gpu_context);
            finish = std::chrono::high_resolution_clock::now();
            print_results("GPU results (OpenCL)", shortest_distances_ocl_gpu, sourceVertex);
            print_duration("GPU (OpenCL)", start, finish, out_file);
            shortest_distances_ocl_gpu.clear();
        }

    #if ENABLE_CUDA == 1
        std::chrono::high_resolution_clock::now();
        auto shortest_distances_cuda_gpu = dijkstra_cuda(graph, sourceVertex);
        finish = std::chrono::high_resolution_clock::now();
        print_results("GPU results (CUDA)", shortest_distances_cuda_gpu, sourceVertex);
        print_duration("GPU (CUDA)", start, finish, out_file);
        shortest_distances_cuda_gpu.clear();
    #endif

    #if 1
        std::chrono::high_resolution_clock::now();
        auto shortest_distances_openacc_gpu = dijkstra_acc(graph, sourceVertex);
        finish = std::chrono::high_resolution_clock::now();
        print_results("GPU results (OpenACC)", shortest_distances_openacc_gpu, sourceVertex);
        print_duration("GPU (OpenACC)", start, finish, out_file);
        shortest_distances_openacc_gpu.clear();
    #endif

        out_file << std::endl;
    }

    return 0;
}
