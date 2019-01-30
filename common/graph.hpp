#pragma once

#include <vector>

class Graph
{
    int neighbors_per_vertex;

public:
    std::vector<int> vertex_array;
    std::vector<int> edge_array;
    std::vector<float> weight_array;
    std::vector<float> weight_matrix;

    Graph(int num_vertexes, int neighbors_per_vertex);

    inline int GetEdge(int vertex_num, int neighbor_idx) const;
    inline float GetWeight(int vertex_num, int neighbor_idx) const;

    void DisplayWeightMatrix() const;
    void PrintVertexData() const;

    int MinDistances(const std::vector<float>& shortest_path,
                     const std::vector<bool>& finalized_verticies,
                     int start_vertex) const;

    int MinDistancesOMP(const std::vector<float>& shortest_path,
                        const std::vector<bool>& finalized_verticies,
                        int start_vertex) const;
private:
    void generate_data(int num_vertexes, int neighbors_per_vertex);
};