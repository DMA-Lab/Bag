#include "graph.h"

#include <algorithm>

namespace bag {

namespace {

const Graph::NeighborMap& empty_neighbors() {
    static const Graph::NeighborMap empty;
    return empty;
}

}  // namespace

void Graph::reserve_vertices(std::size_t count) {
    vertices_.reserve(count);
    adjacency_.reserve(count);
}

void Graph::reserve_neighbors(VertexId v, std::size_t count) {
    adjacency_.try_emplace(v);
    adjacency_[v].reserve(count);
}

bool Graph::insert(VertexId v) {
    adjacency_.try_emplace(v);
    return vertices_.insert(v).second;
}

bool Graph::contains(VertexId v) const {
    return vertices_.contains(v);
}

std::size_t Graph::size() const {
    return vertices_.size();
}

std::size_t Graph::edge_count() const {
    return edge_count_;
}

void Graph::add_directed_edge(VertexId u, VertexId v, EdgeWeight w) {
    insert(u);
    insert(v);
    auto& row = adjacency_[u];
    const bool existed = row.contains(v);
    row[v] = w;
    if (!existed) {
        ++edge_count_;
    }
}

void Graph::connect(VertexId u, VertexId v, EdgeWeight w) {
    add_directed_edge(u, v, w);
    add_directed_edge(v, u, w);
}

void Graph::set_min_undirected_edge(VertexId u, VertexId v, EdgeWeight w) {
    const auto old = get_weight(u, v).value_or(kInfWeight);
    if (old == kInfWeight || w < old) {
        connect(u, v, w);
    }
}

bool Graph::remove_directed_edge(VertexId u, VertexId v) {
    const auto row_it = adjacency_.find(u);
    if (row_it == adjacency_.end()) {
        return false;
    }
    const auto erased = row_it->second.erase(v);
    if (erased != 0U) {
        --edge_count_;
    }
    return erased != 0U;
}

bool Graph::disconnect(VertexId u, VertexId v) {
    const bool removed_forward = remove_directed_edge(u, v);
    const bool removed_reverse = remove_directed_edge(v, u);
    return removed_forward || removed_reverse;
}

bool Graph::erase_isolated_vertex(VertexId v) {
    const auto row_it = adjacency_.find(v);
    if (row_it == adjacency_.end()) {
        return false;
    }
    if (!row_it->second.empty()) {
        return false;
    }
    for (const auto& [other, row] : adjacency_) {
        if (other == v) {
            continue;
        }
        if (row.contains(v)) {
            return false;
        }
    }
    adjacency_.erase(v);
    return vertices_.erase(v) != 0U;
}

bool Graph::has_edge(VertexId u, VertexId v) const {
    return get_weight(u, v).has_value() || get_weight(v, u).has_value();
}

std::optional<EdgeWeight> Graph::get_weight(VertexId u, VertexId v) const {
    const auto row_it = adjacency_.find(u);
    if (row_it == adjacency_.end()) {
        return std::nullopt;
    }
    const auto edge_it = row_it->second.find(v);
    if (edge_it == row_it->second.end()) {
        return std::nullopt;
    }
    return edge_it->second;
}

const Graph::NeighborMap& Graph::neighbors(VertexId v) const {
    const auto it = adjacency_.find(v);
    return (it == adjacency_.end()) ? empty_neighbors() : it->second;
}

const std::unordered_set<VertexId>& Graph::vertex_set() const {
    return vertices_;
}

std::vector<VertexId> Graph::vertices() const {
    auto result = vertices_unsorted();
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<VertexId> Graph::vertices_unsorted() const {
    return std::vector<VertexId>(vertices_.begin(), vertices_.end());
}

std::vector<std::pair<Edge, EdgeWeight>> Graph::directed_edges() const {
    std::vector<std::pair<Edge, EdgeWeight>> result;
    result.reserve(edge_count_);
    for (const auto& [u, row] : adjacency_) {
        for (const auto& [v, w] : row) {
            result.push_back({{u, v}, w});
        }
    }
    return result;
}

std::vector<std::pair<Edge, EdgeWeight>> Graph::undirected_edges() const {
    auto result = undirected_edges_unsorted();
    std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    return result;
}

std::vector<std::pair<Edge, EdgeWeight>> Graph::undirected_edges_unsorted() const {
    std::vector<std::pair<Edge, EdgeWeight>> result;
    std::unordered_set<std::uint64_t> seen;
    for (const auto& [u, row] : adjacency_) {
        for (const auto& [v, w] : row) {
            const auto [a, b] = ordered_edge(u, v);
            const auto key = pack_pair(a, b);
            if (seen.insert(key).second) {
                result.push_back({{a, b}, w});
            }
        }
    }
    return result;
}

}  // namespace bag
