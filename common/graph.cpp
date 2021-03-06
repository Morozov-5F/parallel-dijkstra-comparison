#include <iostream>
#include <cfloat>
#include <climits>
#include <omp.h>

#include "graph.hpp"


Graph::Graph(int num_vertexes, int neighbors_per_vertex) :
                                neighbors_per_vertex(neighbors_per_vertex),
                                vertex_array(num_vertexes),
                                edge_array(num_vertexes * neighbors_per_vertex),
                                weight_array(num_vertexes * neighbors_per_vertex),
                                weight_matrix(num_vertexes * num_vertexes)
{
    this->generate_data(num_vertexes, neighbors_per_vertex);
}

void Graph::generate_data(int num_vertexes, int neighbors_per_vertex)
{
    #pragma omp parallel for
    for (int i = 0; i < num_vertexes; ++i)
    {
        this->vertex_array[i] = i * neighbors_per_vertex;
    }

    for (int k = 0; k < num_vertexes; ++k)
    {
        std::vector<int> temp_array(neighbors_per_vertex, INT_MAX);
        #pragma omp parallel for shared(temp_array)
        for (int l = 0; l < neighbors_per_vertex; ++l)
        {
            bool skip_iteration = false;
            int temp = 0;
            while (!skip_iteration)
            {
                skip_iteration = true;

                temp = rand() % num_vertexes;
                for (auto t = 0ULL; t < temp_array.size(); ++t)
                {
                    if (temp == temp_array[t])
                    {
                        skip_iteration = false;
                        break;
                    }
                }
                if (temp == k)
                {
                    skip_iteration = false;
                }
                if (skip_iteration)
                {
                    temp_array[l] = temp;
                }
            }
            this->edge_array[k * neighbors_per_vertex + l] = temp;
            this->weight_array[k * neighbors_per_vertex + l] = (float)(rand() % 1000) / 1000.0f;
        }
    }
    // Generate weight matrix
    #pragma omp parallel for
    for (auto k = 0; k < num_vertexes; ++k)
    {
        weight_matrix[k * num_vertexes + k] = 0.f;
    }

    for (auto k = 0; k < num_vertexes; ++k)
    {
        #pragma omp parallel for shared(neighbors_per_vertex, k)
        for (int l = 0; l < neighbors_per_vertex; ++l)
        {
            auto edge = this->edge_array[this->vertex_array[k] + l];
            auto weight = this->weight_array[this->vertex_array[k] + l];
            weight_matrix[k * num_vertexes + edge] = weight;
        }
    }
}

inline int Graph::GetEdge(int vertex_num, int neighbor_idx) const
{
    return this->edge_array[this->vertex_array[vertex_num] + neighbor_idx];
}

inline float Graph::GetWeight(int vertex_num, int neighbor_idx) const
{
    return this->weight_array[this->vertex_array[vertex_num] + neighbor_idx];
}

void Graph::PrintVertexData() const
{
    auto num_vertices = this->vertex_array.size();

    for (auto k = 0ULL; k < num_vertices; ++k)
    {
        for (auto l = 0; l < this->neighbors_per_vertex; ++l)
        {
            auto edge = this->GetEdge(k, l);
            auto weight = this->GetWeight(k, l);

            std::cout << "Vertex nr. " << k << "; Edge nr. " << edge << "; Weight = " << weight << std::endl;
        }
    }

    for (auto k = 0ULL; k < num_vertices * this->neighbors_per_vertex; ++k)
    {
        std::cout << k << " " << this->edge_array[k] << " " << this->weight_array[k] << std::endl;
    }
}

void Graph::DisplayWeightMatrix() const
{
    auto num_vertices = this->vertex_array.size();

    std::cout << "Weight matrix" << std::endl;
    for (auto k = 0ULL; k < num_vertices; ++k)
    {
        for (auto l = 0ULL; l < num_vertices; ++l)
        {
            if (this->weight_matrix[k * num_vertices + l] < FLT_MAX)
            {
                std::cout.precision(3);
                std::cout << this->weight_matrix[k * num_vertices + l] << "\t";
            }
            else
            {
                std::cout << "--\t";
            }
        }
        std::cout << std::endl;
    }
}

int Graph::MinDistances(const std::vector<float>& shortest_path,
                        const std::vector<bool>& finalized_verticies,
                        int start_vertex) const
{
    auto minIndex = start_vertex;
    auto min = FLT_MAX;

    for (auto v = 0ULL; v < this->vertex_array.size(); v++)
    {
        if (finalized_verticies[v] == false && shortest_path[v] <= min)
        {
            min = shortest_path[v];
            minIndex = v;
        }
    }

    return minIndex;
}

int Graph::MinDistancesOMP(const std::vector<float>& shortest_path,
                           const std::vector<bool>& finalized_verticies,
                           int start_vertex) const
{
    int min_vertex = start_vertex;
    float min_dist = FLT_MAX;

    float thread_min_dist;
    int thread_min_vertex;

    #pragma omp parallel private(thread_min_dist, thread_min_vertex) shared(shortest_path, finalized_verticies, start_vertex, min_vertex, min_dist)
    {
        thread_min_dist = min_dist;
        thread_min_vertex = min_vertex;

        #pragma omp barrier

        #pragma omp for nowait
        for (auto v = 0ULL; v < this->vertex_array.size(); ++v)
        {
            if (finalized_verticies[v] == false && shortest_path[v] <= thread_min_dist)
            {
                thread_min_dist = shortest_path[v];
                thread_min_vertex = v;
            }
        }
        #pragma omp critical
        {
            if (thread_min_dist < min_dist)
            {
                min_dist = thread_min_dist;
                min_vertex = thread_min_vertex;
            }
        }
    }
    return min_vertex;
}
