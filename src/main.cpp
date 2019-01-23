#include <iostream>

#include "src/dijkstra.hpp"

int main() {

    // --- Number of graph vertices
    int num_vertices = 8;

    // --- Number of edges per graph vertex
    int neighbors_per_vertex = 6;

    // --- Source vertex
    int sourceVertex = 0;


    std::cout << "Starting a program" << std::endl;
    // --- Allocate memory for arrays
    Graph graph(num_vertices, neighbors_per_vertex);

    // --- Displaying the adjacency list and constructing the adjacency matrix
    graph.PrintVertexData();
    graph.DisplayWeightMatrix();

    // --- Running sequential Dijkstra on the CPU
    auto shortest_distances_seq = dijkstra_sequential(graph, sourceVertex);
    std::cout << std::endl << "CPU results" << std::endl;
    for (int k = 0; k < num_vertices; k++)
    {
        std::cout << "From vertex " << sourceVertex << " to vertex " << k << " = " << shortest_distances_seq[k] << std::endl;
    }

    // --- Allocate space for the h_shortestDistancesGPU
    // float *h_shortestDistancesGPU = (float*)malloc(sizeof(float) * graph.numVertices);
    // dijkstraGPU(&graph, sourceVertex, h_shortestDistancesGPU);

    // printf("\nGPU results\n");
    // for (int k = 0; k < numVertices; k++) printf("From vertex %i to vertex %i = %f\n", sourceVertex, k, h_shortestDistancesGPU[k]);

    return 0;
}