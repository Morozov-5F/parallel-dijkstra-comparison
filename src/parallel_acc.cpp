#include "dijkstra.hpp"
#include <algorithm>

// #include <openacc.h>

#define NUM_ASYNCHRONOUS_ITERATIONS 20  // Number of async loop iterations before attempting to read results back

std::vector<float> dijkstra_acc(const __restrict Graph &graph, int source_vertex)
{
    auto number_of_vetecies = graph.vertex_array.size();

    // finalized_verticies -- true if vertex i is included in the shortest path tree
    //                      or the shortest distance from the source node to i is
    //                      finalized
    std::vector<bool>  finalized_verticies(number_of_vetecies, false);
    std::vector<float> distances(number_of_vetecies, FLT_MAX);
    std::vector<float> updating_distances(number_of_vetecies, FLT_MAX);

    // bool * d_finalized_verticies = new bool[number_of_vetecies];
    // float * d_distances = new float[number_of_vetecies];
    // float * d_updating_distances = new float[number_of_vetecies];

    // memset(d_distances, FLT_MAX, number_of_vetecies);
    // memset(d_updating_distances, FLT_MAX, number_of_vetecies);

    // distances of the source vertex from itself is always 0
    finalized_verticies[source_vertex] = true;
    distances[source_vertex] = 0.f;
    updating_distances[source_vertex] = 0.f;

    // --- Dijkstra iterations
#pragma acc data copy(finalized_verticies.data(), distances.data(), updating_distances.dat())
    while (std::find(finalized_verticies.begin(), finalized_verticies.end(), true) != finalized_verticies.end())
    {
        for (int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++)
        {
            // Kernel 1
            #pragma acc kernels loop independent
            {
                for (auto i = 0ULL; i < number_of_vetecies; ++i)
                {
                    if (!finalized_verticies[i])
                    {
                        continue;
                    }

                    auto edge_start = graph.vertex_array[i];
                    auto edge_end = 0;

                    if (i + 1 < number_of_vetecies)
                    {
                        edge_end = graph.vertex_array[i + 1];
                    }
                    else
                    {
                        edge_end = graph.edge_array.size();
                    }

                    for (auto edge = edge_start; edge < edge_end; edge++)
                    {
                        auto nid = graph.edge_array[edge];
                        if (updating_distances[nid] > distances[i] + graph.weight_array[edge])
                        {
                            updating_distances[nid] = distances[i] + graph.weight_array[edge];
                        }
                    }
                }
            }
            // Kernel 2
            #pragma acc kernels loop independent
            {
                for (auto i = 0ULL; i < number_of_vetecies; ++i)
                {
                    if (distances[i] > updating_distances[i])
                    {
                        distances[i] = updating_distances[i];
                        finalized_verticies[i] = true;
                    }

                    updating_distances[i] = distances[i];
                }
            }
        }
    }
    return distances;
}