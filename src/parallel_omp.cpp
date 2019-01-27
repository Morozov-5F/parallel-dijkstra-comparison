#include "src/dijkstra.hpp"

#include <iostream>
#include <chrono>

#include <omp.h>

std::vector<float> dijkstra_omp(const Graph &graph,
                                int source_vertex)
{
    auto number_of_vetecies = graph.vertex_array.size();

    // finalized_verticies -- true if vertex i is included in the shortest path tree
    //                      or the shortest distance from the source node to i is
    //                      finalized
    std::vector<bool>  finalized_verticies(number_of_vetecies, false);
    std::vector<float> distances(number_of_vetecies, FLT_MAX);

    std::vector<std::vector<int>> actual_path(number_of_vetecies);
    // distances of the source vertex from itself is always 0
    distances[source_vertex] = 0.f;

    omp_set_dynamic(0);     // Explicitly disable dynamic teams

    // --- Dijkstra iterations
    for (auto iter_count = 0ULL; iter_count < number_of_vetecies - 1; ++iter_count)
    {
        // parallel min_distances funciton
        int current_vertex = graph.MinDistancesOMP(distances, finalized_verticies, source_vertex);

        finalized_verticies[current_vertex] = true;

        #pragma omp parallel shared(graph, finalized_verticies, distances, actual_path, number_of_vetecies)
        {
            // For all unvisited neighbors of current vertex
            #pragma omp for
            for (auto v = 0UL; v < number_of_vetecies; ++v)
            {
                if (0 == graph.weight_matrix[current_vertex * number_of_vetecies + v] ||
                    FLT_MAX == distances[current_vertex])
                {
                    continue;
                }

                if (finalized_verticies[v])
                {
                    continue;
                }

                if (distances[current_vertex] + graph.weight_matrix[current_vertex * number_of_vetecies + v] < distances[v])
                {
                    distances[v] = distances[current_vertex] + graph.weight_matrix[current_vertex * number_of_vetecies + v];

                    actual_path[v].push_back(current_vertex);
                }
            }
            #pragma omp barrier
        }
    }

#if ENABLE_PATH_PRINT != 0
    std::cout << "Actual paths: " << std::endl;
    for (auto i = 0ULL; i < actual_path.size(); ++i)
    {
        std::cout << "Path from " << source_vertex << " to " << i  << ": ";
        if (actual_path[i].size() == 0)
        {
            std::cout << "None (Direct?)" << std::endl;
            continue;
        }

        // std::cout << source_vertex << " -> ";
        for (auto j = 0ULL; j < actual_path[i].size(); ++j)
        {
            std::cout << actual_path[i][j] << " -> ";
        }
        std::cout << i << std::endl;
    }
#endif

    return distances;
}