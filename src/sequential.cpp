#include "src/dijkstra.hpp"

#include <iostream>

std::vector<float> dijkstra_sequential(const Graph &graph,
                                       int source_vertex)
{
    auto number_of_vertexes = graph.vertex_array.size();

    // finalized_verticies -- true if vertex i is included in the shortest path tree
    //                      or the shortest distance from the source node to i is
    //                      finalized
    std::vector<bool>  finalized_verticies(number_of_vertexes, false);
    std::vector<float> shortest_path(number_of_vertexes, FLT_MAX);

    std::vector<std::vector<int>> actual_path(number_of_vertexes);

    // shortest_path of the source vertex from itself is always 0
    shortest_path[source_vertex] = 0.f;

    // --- Dijkstra iterations
    for (auto iter_count = 0; iter_count < number_of_vertexes - 1; ++iter_count)
    {
        int current_vertex = graph.MinDistances(shortest_path, finalized_verticies, source_vertex);

        finalized_verticies[current_vertex] = true;

        for (auto v = 0; v < number_of_vertexes; ++v)
        {
            if (!finalized_verticies[v] &&
                graph.weight_matrix[current_vertex * number_of_vertexes + v] &&
                shortest_path[current_vertex] != FLT_MAX &&
                shortest_path[current_vertex] + graph.weight_matrix[current_vertex * number_of_vertexes + v] < shortest_path[v])
            {
                shortest_path[v] = shortest_path[current_vertex] + graph.weight_matrix[current_vertex * number_of_vertexes + v];
                actual_path[current_vertex].push_back(v);
                std::cout << "shortest_path[" << v << "] = " << shortest_path[v] << std::endl;
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

        std::cout << source_vertex << " -> ";
        for (auto j = 0; j < actual_path[i].size(); ++j)
        {
            std::cout << actual_path[i][j] << " -> ";
        }
        std::cout << i << std::endl;
    }
    return shortest_path;
}