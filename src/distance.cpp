#include "distance.h"

#include <functional>
#include <queue>

namespace bag {

void DistanceTable::reserve_rows(std::size_t count) {
    rows_.reserve(count);
}

void DistanceTable::reserve_row(VertexId u, std::size_t count) {
    rows_.try_emplace(u);
    rows_[u].reserve(count);
}

EdgeWeight DistanceTable::get_or_inf(VertexId u, VertexId v) const {
    if (u == v) {
        return 0;
    }
    const auto row_it = rows_.find(u);
    if (row_it == rows_.end()) {
        return kInfWeight;
    }
    const auto col_it = row_it->second.find(v);
    return (col_it == row_it->second.end()) ? kInfWeight : col_it->second;
}

void DistanceTable::set(VertexId u, VertexId v, EdgeWeight value) {
    rows_[u][v] = value;
}

void DistanceTable::erase_vertex(VertexId v) {
    rows_.erase(v);
    for (auto& [_, row] : rows_) {
        row.erase(v);
    }
}

std::vector<std::pair<VertexId, EdgeWeight>> DistanceTable::distance_from(VertexId u) const {
    std::vector<std::pair<VertexId, EdgeWeight>> result;
    const auto row_it = rows_.find(u);
    if (row_it == rows_.end()) {
        return result;
    }
    result.reserve(row_it->second.size());
    for (const auto& [v, d] : row_it->second) {
        if (v != u && d != kInfWeight) {
            result.push_back({v, d});
        }
    }
    return result;
}

bool DistanceTable::empty() const {
    return rows_.empty();
}

std::size_t DistanceTable::payload_bytes() const {
    std::size_t bytes = rows_.size() * sizeof(VertexId);
    for (const auto& [u, row] : rows_) {
        (void)u;
        bytes += row.size() * (sizeof(VertexId) + sizeof(EdgeWeight));
    }
    return bytes;
}

const std::unordered_map<VertexId, std::unordered_map<VertexId, EdgeWeight>>& DistanceTable::rows() const {
    return rows_;
}

DistanceMap dijkstra(
    const Graph& graph,
    const std::vector<std::pair<VertexId, EdgeWeight>>& seeds,
    EdgeWeight max_radius
) {
    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    DistanceMap dist;

    for (const auto& [seed, value] : seeds) {
        auto it = dist.find(seed);
        if (it == dist.end() || value < it->second) {
            dist[seed] = value;
            pq.push({value, seed});
        }
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();

        const auto best_it = dist.find(u);
        if (best_it == dist.end() || current != best_it->second) {
            continue;
        }
        if (current > max_radius) {
            continue;
        }

        for (const auto& [v, w] : graph.neighbors(u)) {
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next > max_radius) {
                continue;
            }
            auto it = dist.find(v);
            if (it == dist.end() || next < it->second) {
                dist[v] = next;
                pq.push({next, v});
            }
        }
    }

    return dist;
}

DistanceTable all_pairs_shortest_paths(const Graph& graph) {
    DistanceTable table;
    for (const auto src : graph.vertex_set()) {
        const auto dist = dijkstra(graph, {{src, 0}});
        table.set(src, src, 0);
        for (const auto& [dst, value] : dist) {
            table.set(src, dst, value);
        }
    }
    return table;
}

DistanceTable update_all_pairs_after_undirected_edge_insertion(
    const DistanceTable& old_table,
    const Graph& updated_graph,
    VertexId u,
    VertexId v,
    EdgeWeight w
) {
    DistanceTable updated = old_table;
    const auto vertices = updated_graph.vertices_unsorted();
    const auto n = vertices.size();
    std::vector<EdgeWeight> dist_to_u(n, kInfWeight);
    std::vector<EdgeWeight> dist_to_v(n, kInfWeight);

    for (std::size_t i = 0; i < n; ++i) {
        const auto x = vertices[i];
        dist_to_u[i] = old_table.get_or_inf(x, u);
        dist_to_v[i] = old_table.get_or_inf(x, v);
        updated.set(x, x, 0);
    }

    const auto relax_through_new_edge = [&](EdgeWeight lhs, EdgeWeight rhs) -> EdgeWeight {
        if (lhs == kInfWeight || rhs == kInfWeight) {
            return kInfWeight;
        }
        if (lhs > kInfWeight - w || rhs > kInfWeight - w - lhs) {
            return kInfWeight;
        }
        return static_cast<EdgeWeight>(lhs + w + rhs);
    };

    for (std::size_t i = 0; i < n; ++i) {
        const auto a = vertices[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            const auto b = vertices[j];
            auto best = old_table.get_or_inf(a, b);
            const auto via_uv = relax_through_new_edge(dist_to_u[i], dist_to_v[j]);
            const auto via_vu = relax_through_new_edge(dist_to_v[i], dist_to_u[j]);
            best = std::min(best, via_uv);
            best = std::min(best, via_vu);
            if (best != kInfWeight) {
                updated.set(a, b, best);
                updated.set(b, a, best);
            }
        }
    }

    const auto old_uv = old_table.get_or_inf(u, v);
    if (w < old_uv) {
        updated.set(u, v, w);
        updated.set(v, u, w);
    }
    return updated;
}

HalfWeight distal_point_distance(EdgeWeight lhs, EdgeWeight rhs, EdgeWeight edge_weight) {
    const std::uint64_t sum =
        static_cast<std::uint64_t>(lhs) + static_cast<std::uint64_t>(rhs) +
        static_cast<std::uint64_t>(edge_weight);
    return HalfWeight{
        static_cast<EdgeWeight>(sum / 2U),
        static_cast<bool>((sum % 2U) != 0U),
    };
}

}  // namespace bag
