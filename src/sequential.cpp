#include "src/dijkstra.hpp"

#include <iostream>
#include <chrono>

std::vector<float> dijkstra_sequential(const Graph &graph,
                                       int source_vertex)
{
    auto number_of_vertexes = graph.vertex_array.size();

    // finalized_verticies -- true if vertex i is included in the shortest path tree
    //                      or the shortest distance from the source node to i is
    //                      finalized
    std::vector<bool>  finalized_verticies(number_of_vertexes, false);
    std::vector<float> distances(number_of_vertexes, FLT_MAX);

    std::vector<std::vector<int>> actual_path(number_of_vertexes);
    for(auto&& path : actual_path)
    {
        path.push_back(source_vertex);
    }
    // distances of the source vertex from itself is always 0
    distances[source_vertex] = 0.f;

    // --- Dijkstra iterations
    for (auto iter_count = 0; iter_count < number_of_vertexes - 1; ++iter_count)
    {
        int current_vertex = graph.MinDistances(distances, finalized_verticies, source_vertex);

        finalized_verticies[current_vertex] = true;

        // For all unvisited neighbors of current vertex
        for (auto v = 0; v < number_of_vertexes; ++v)
        {
            if (0 == graph.weight_matrix[current_vertex * number_of_vertexes + v] ||
                FLT_MAX == distances[current_vertex])
            {
                continue;
            }

            if (finalized_verticies[v])
            {
                continue;
            }

            if (distances[current_vertex] + graph.weight_matrix[current_vertex * number_of_vertexes + v] < distances[v])
            {
                distances[v] = distances[current_vertex] + graph.weight_matrix[current_vertex * number_of_vertexes + v];

                if (!actual_path[v].size() || *actual_path[v].end() != current_vertex)
                {
                    actual_path[v].push_back(current_vertex);
                }
            }
        }
    }

    std::cout << "Actual paths: " << std::endl;
    for (auto i = 0; i < actual_path.size(); ++i)
    {
        std::cout << "Path from " << source_vertex << " to " << i  << ": ";
        if (actual_path[i].size() == 0)
        {
            std::cout << "None (Direct?)" << std::endl;
            continue;
        }

        // std::cout << source_vertex << " -> ";
        for (auto j = 0; j < actual_path[i].size(); ++j)
        {
            std::cout << actual_path[i][j] << " -> ";
        }
        std::cout << i << std::endl;
    }
    return distances;
}