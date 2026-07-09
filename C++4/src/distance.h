#pragma once

#include <unordered_map>
#include <vector>

#include "graph.h"

namespace bag {

class DistanceTable {
public:
    void reserve_rows(std::size_t count);
    void reserve_row(VertexId u, std::size_t count);
    [[nodiscard]] EdgeWeight get_or_inf(VertexId u, VertexId v) const;
    void set(VertexId u, VertexId v, EdgeWeight value);
    void erase_vertex(VertexId v);
    [[nodiscard]] std::vector<std::pair<VertexId, EdgeWeight>> distance_from(VertexId u) const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] std::size_t payload_bytes() const;
    [[nodiscard]] const std::unordered_map<VertexId, std::unordered_map<VertexId, EdgeWeight>>& rows() const;

private:
    std::unordered_map<VertexId, std::unordered_map<VertexId, EdgeWeight>> rows_;
};

using DistanceMap = std::unordered_map<VertexId, EdgeWeight>;

DistanceMap dijkstra(
    const Graph& graph,
    const std::vector<std::pair<VertexId, EdgeWeight>>& seeds,
    EdgeWeight max_radius = kInfWeight
);

DistanceTable all_pairs_shortest_paths(const Graph& graph);
DistanceTable update_all_pairs_after_undirected_edge_insertion(
    const DistanceTable& old_table,
    const Graph& updated_graph,
    VertexId u,
    VertexId v,
    EdgeWeight w
);

HalfWeight distal_point_distance(EdgeWeight lhs, EdgeWeight rhs, EdgeWeight edge_weight);

}  // namespace bag
