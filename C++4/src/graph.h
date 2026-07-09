#pragma once

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vertex.h"

namespace bag {

class Graph {
public:
    using NeighborMap = std::unordered_map<VertexId, EdgeWeight>;

    void reserve_vertices(std::size_t count);
    void reserve_neighbors(VertexId v, std::size_t count);
    bool insert(VertexId v);
    [[nodiscard]] bool contains(VertexId v) const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t edge_count() const;

    void add_directed_edge(VertexId u, VertexId v, EdgeWeight w);
    void connect(VertexId u, VertexId v, EdgeWeight w);
    void set_min_undirected_edge(VertexId u, VertexId v, EdgeWeight w);
    bool remove_directed_edge(VertexId u, VertexId v);
    bool disconnect(VertexId u, VertexId v);
    bool erase_isolated_vertex(VertexId v);

    [[nodiscard]] bool has_edge(VertexId u, VertexId v) const;
    [[nodiscard]] std::optional<EdgeWeight> get_weight(VertexId u, VertexId v) const;
    [[nodiscard]] const NeighborMap& neighbors(VertexId v) const;
    [[nodiscard]] const std::unordered_set<VertexId>& vertex_set() const;

    [[nodiscard]] std::vector<VertexId> vertices() const;
    [[nodiscard]] std::vector<VertexId> vertices_unsorted() const;
    [[nodiscard]] std::vector<std::pair<Edge, EdgeWeight>> directed_edges() const;
    [[nodiscard]] std::vector<std::pair<Edge, EdgeWeight>> undirected_edges() const;
    [[nodiscard]] std::vector<std::pair<Edge, EdgeWeight>> undirected_edges_unsorted() const;

private:
    std::unordered_set<VertexId> vertices_;
    std::unordered_map<VertexId, NeighborMap> adjacency_;
    std::size_t edge_count_{0};
};

}  // namespace bag
