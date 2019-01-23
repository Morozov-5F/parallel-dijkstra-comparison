#pragma once

#include <vector>
#include <cfloat>

#include "common/graph.hpp"

std::vector<float> dijkstra_sequential(const Graph &graph,
                                       int source_vertex);