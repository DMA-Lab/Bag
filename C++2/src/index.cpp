#include "index.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include "distance.h"

namespace bag {

namespace {

EdgeWeight edge_weight_any(const Graph& graph, const Edge& edge) {
    if (const auto forward = graph.get_weight(edge.first, edge.second); forward.has_value()) {
        return *forward;
    }
    if (const auto reverse = graph.get_weight(edge.second, edge.first); reverse.has_value()) {
        return *reverse;
    }
    throw std::runtime_error("edge not found");
}

std::optional<EdgeWeight> edge_weight_any_or_null(const Graph& graph, const Edge& edge) {
    if (const auto forward = graph.get_weight(edge.first, edge.second); forward.has_value()) {
        return *forward;
    }
    if (const auto reverse = graph.get_weight(edge.second, edge.first); reverse.has_value()) {
        return *reverse;
    }
    return std::nullopt;
}

DistanceMap local_query_distances(const Subgraph& sg, const QueryPoint& query, EdgeWeight max_radius) {
    const auto edge = ordered_edge(query.edge.first, query.edge.second);
    const auto weight = edge_weight_any(sg.graph, edge);
    if (query.offset >= weight) {
        throw std::runtime_error("query offset must be smaller than edge weight");
    }
    return dijkstra(
        sg.graph,
        {
            {edge.first, query.offset},
            {edge.second, static_cast<EdgeWeight>(weight - query.offset)},
        },
        max_radius
    );
}

EdgeWeight lookup_local_distance(const LocalDijkstraResult& result, VertexId vertex) {
    if (result.index == nullptr) {
        return kInfWeight;
    }
    const auto it = result.index->local_ids.find(vertex);
    if (it == result.index->local_ids.end()) {
        return kInfWeight;
    }
    const auto local_id = static_cast<std::size_t>(it->second);
    if (local_id >= result.dist.size()) {
        return kInfWeight;
    }
    return result.dist[local_id];
}

LocalDijkstraResult local_dijkstra_compact(
    const LocalSubgraphIndex& local_index,
    const std::vector<std::pair<VertexId, EdgeWeight>>& seeds,
    EdgeWeight max_radius
) {
    using QueueItem = std::pair<EdgeWeight, std::uint32_t>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;

    LocalDijkstraResult result;
    result.index = &local_index;
    result.dist.assign(local_index.adjacency.size(), kInfWeight);

    for (const auto& [seed, value] : seeds) {
        const auto it = local_index.local_ids.find(seed);
        if (it == local_index.local_ids.end()) {
            continue;
        }
        const auto local_id = static_cast<std::size_t>(it->second);
        if (value < result.dist[local_id]) {
            if (result.dist[local_id] == kInfWeight) {
                ++result.reached_count;
            }
            result.dist[local_id] = value;
            pq.push({value, it->second});
        }
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();

        const auto local_u = static_cast<std::size_t>(u);
        if (local_u >= result.dist.size() || current != result.dist[local_u]) {
            continue;
        }
        if (current > max_radius) {
            continue;
        }

        for (const auto& [v, w] : local_index.adjacency[local_u]) {
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next > max_radius) {
                continue;
            }
            const auto local_v = static_cast<std::size_t>(v);
            if (next < result.dist[local_v]) {
                if (result.dist[local_v] == kInfWeight) {
                    ++result.reached_count;
                }
                result.dist[local_v] = next;
                pq.push({next, v});
            }
        }
    }

    return result;
}

LocalDijkstraResult local_query_distances_compact(
    const Subgraph& sg,
    const LocalSubgraphIndex& local_index,
    const QueryPoint& query,
    EdgeWeight max_radius
) {
    const auto edge = ordered_edge(query.edge.first, query.edge.second);
    const auto weight = edge_weight_any(sg.graph, edge);
    if (query.offset >= weight) {
        throw std::runtime_error("query offset must be smaller than edge weight");
    }
    return local_dijkstra_compact(
        local_index,
        {
            {edge.first, query.offset},
            {edge.second, static_cast<EdgeWeight>(weight - query.offset)},
        },
        max_radius
    );
}

std::vector<std::pair<VertexId, EdgeWeight>> boundary_query_seeds(
    const Subgraph& initial_subgraph,
    const DistanceMap& local_dist
) {
    std::vector<std::pair<VertexId, EdgeWeight>> seeds;
    seeds.reserve(initial_subgraph.bound_vertices.size());
    for (const auto b : initial_subgraph.bound_vertices) {
        const auto it = local_dist.find(b);
        if (it != local_dist.end()) {
            seeds.push_back({b, it->second});
        }
    }
    return seeds;
}

std::vector<std::pair<VertexId, EdgeWeight>> boundary_query_seeds(
    const Subgraph& initial_subgraph,
    const LocalDijkstraResult& local_dist
) {
    std::vector<std::pair<VertexId, EdgeWeight>> seeds;
    seeds.reserve(initial_subgraph.bound_vertices.size());
    for (const auto b : initial_subgraph.bound_vertices) {
        const auto distance = lookup_local_distance(local_dist, b);
        if (distance != kInfWeight) {
            seeds.push_back({b, distance});
        }
    }
    return seeds;
}

DistanceMap boundary_query_distances(
    const Graph& skeleton,
    const Subgraph& initial_subgraph,
    const DistanceMap& local_dist,
    EdgeWeight max_radius
) {
    return dijkstra(skeleton, boundary_query_seeds(initial_subgraph, local_dist), max_radius);
}

struct DijkstraTrace {
    DistanceMap dist;
    std::vector<std::pair<VertexId, EdgeWeight>> settled;
    std::size_t relax_attempts{0};
    std::size_t successful_relaxes{0};
    std::size_t pq_pushes{0};
    std::size_t num_rows_truncated{0};
    std::size_t num_exits_skipped_by_truncation{0};
};

struct CliqueExitCandidate {
    VertexId border{kInvalidVertex};
    EdgeWeight local_distance{kInfWeight};
    EdgeWeight candidate_distance{kInfWeight};
};

struct MaterializedCliqueRow {
    std::vector<CliqueExitCandidate> exits;
    bool used_factorized{false};
    std::size_t hubs_used{0};
};

struct KnnFrontierItem {
    enum class Kind {
        Boundary,
        RowEmit,
    };

    EdgeWeight distance{kInfWeight};
    Kind kind{Kind::Boundary};
    VertexId vertex{kInvalidVertex};
    std::size_t row_id{0};
};

struct KnnFrontierGreater {
    bool operator()(const KnnFrontierItem& lhs, const KnnFrontierItem& rhs) const {
        if (lhs.distance != rhs.distance) {
            return lhs.distance > rhs.distance;
        }
        if (lhs.kind != rhs.kind) {
            return static_cast<int>(lhs.kind) > static_cast<int>(rhs.kind);
        }
        if (lhs.vertex != rhs.vertex) {
            return lhs.vertex > rhs.vertex;
        }
        return lhs.row_id > rhs.row_id;
    }
};

const std::vector<std::pair<VertexId, EdgeWeight>>& empty_sorted_row() {
    static const std::vector<std::pair<VertexId, EdgeWeight>> empty;
    return empty;
}

const std::vector<std::pair<VertexId, EdgeWeight>>& lookup_sorted_skeleton_row(
    VertexId border,
    const std::vector<std::int32_t>& row_index,
    const std::vector<std::vector<std::pair<VertexId, EdgeWeight>>>& rows
) {
    if (border >= row_index.size()) {
        return empty_sorted_row();
    }
    const auto idx = row_index[border];
    if (idx < 0) {
        return empty_sorted_row();
    }
    return rows[static_cast<std::size_t>(idx)];
}

const std::vector<std::pair<VertexId, EdgeWeight>>& lookup_subgraph_clique_row(
    VertexId entry_border,
    const std::unordered_map<VertexId, std::vector<std::pair<VertexId, EdgeWeight>>>& rows
) {
    const auto it = rows.find(entry_border);
    return (it == rows.end()) ? empty_sorted_row() : it->second;
}

std::vector<CliqueExitCandidate> materialize_clique_exit_candidates(
    const std::vector<std::pair<VertexId, EdgeWeight>>& local_row,
    EdgeWeight entry_distance
) {
    std::vector<CliqueExitCandidate> exits;
    exits.reserve(local_row.size());
    for (const auto& [other_b, local] : local_row) {
        if (entry_distance == kInfWeight || local == kInfWeight || entry_distance > kInfWeight - local) {
            continue;
        }
        exits.push_back(CliqueExitCandidate{
            other_b,
            local,
            static_cast<EdgeWeight>(entry_distance + local),
        });
    }
    return exits;
}

MaterializedCliqueRow materialize_factorized_exit_candidates(
    const FactorizedTransferSubgraphModel& model,
    VertexId entry_border,
    EdgeWeight entry_distance
) {
    MaterializedCliqueRow row;
    const auto it = model.entry_rows.find(entry_border);
    if (it == model.entry_rows.end()) {
        return row;
    }
    row.used_factorized = true;
    row.hubs_used = it->second.size();
    for (const auto& hub_row : it->second) {
        if (entry_distance == kInfWeight ||
            hub_row.entry_to_hub_distance == kInfWeight ||
            entry_distance > kInfWeight - hub_row.entry_to_hub_distance) {
            continue;
        }
        const auto base = static_cast<EdgeWeight>(entry_distance + hub_row.entry_to_hub_distance);
        for (const auto& [exit_border, hub_to_exit] : hub_row.exits) {
            if (hub_to_exit == kInfWeight || base > kInfWeight - hub_to_exit) {
                continue;
            }
            row.exits.push_back(CliqueExitCandidate{
                exit_border,
                static_cast<EdgeWeight>(hub_row.entry_to_hub_distance + hub_to_exit),
                static_cast<EdgeWeight>(base + hub_to_exit),
            });
        }
    }
    return row;
}

MaterializedCliqueRow materialize_subgraph_exit_candidates(
    VertexId entry_border,
    EdgeWeight entry_distance,
    SgId sg_id,
    const std::vector<std::unordered_map<VertexId, std::vector<std::pair<VertexId, EdgeWeight>>>>& explicit_rows,
    const std::vector<FactorizedTransferSubgraphModel>& factorized_models,
    const std::vector<std::uint8_t>& factorized_enabled,
    bool use_factorized_transfer
) {
    if (use_factorized_transfer &&
        sg_id < factorized_enabled.size() &&
        factorized_enabled[sg_id] != 0U) {
        auto row = materialize_factorized_exit_candidates(
            factorized_models[sg_id],
            entry_border,
            entry_distance
        );
        if (row.used_factorized) {
            return row;
        }
    }
    MaterializedCliqueRow row;
    row.exits = materialize_clique_exit_candidates(
        lookup_subgraph_clique_row(entry_border, explicit_rows[sg_id]),
        entry_distance
    );
    return row;
}

VertexId choose_representative_boundary(
    const Subgraph& sg,
    const std::vector<VertexId>& boundaries
) {
    if (boundaries.empty()) {
        return kInvalidVertex;
    }
    if (boundaries.size() == 1) {
        return boundaries.front();
    }

    VertexId best = boundaries.front();
    unsigned long long best_score = std::numeric_limits<unsigned long long>::max();
    for (const auto candidate : boundaries) {
        unsigned long long score = 0;
        bool disconnected = false;
        for (const auto other : boundaries) {
            if (candidate == other) {
                continue;
            }
            const auto distance = sg.distance.get_or_inf(candidate, other);
            if (distance == kInfWeight) {
                disconnected = true;
                break;
            }
            score += static_cast<unsigned long long>(distance);
        }
        if (disconnected) {
            continue;
        }
        if (score < best_score || (score == best_score && candidate < best)) {
            best = candidate;
            best_score = score;
        }
    }
    return best;
}

std::vector<SgId> gather_shortcut_candidate_subgraphs(
    const std::vector<std::vector<SgId>>& adjacency,
    SgId source,
    std::size_t leaf_hops
) {
    if (leaf_hops == 0 || source >= adjacency.size()) {
        return {};
    }

    const auto target_hop = leaf_hops > 1 ? leaf_hops : 1U;
    std::vector<int> depth(adjacency.size(), -1);
    std::queue<SgId> queue;
    queue.push(source);
    depth[source] = 0;

    std::vector<SgId> result;
    while (!queue.empty()) {
        const auto current = queue.front();
        queue.pop();
        const auto current_depth = static_cast<std::size_t>(depth[current]);
        if (current_depth >= leaf_hops) {
            continue;
        }
        for (const auto next : adjacency[current]) {
            if (depth[next] != -1) {
                continue;
            }
            depth[next] = static_cast<int>(current_depth + 1);
            queue.push(next);
            const auto next_depth = static_cast<std::size_t>(depth[next]);
            if (next_depth == target_hop) {
                result.push_back(next);
            }
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::size_t> build_parent_assignment(
    const std::vector<std::vector<SgId>>& adjacency,
    std::size_t max_children
) {
    const std::size_t n = adjacency.size();
    std::vector<std::size_t> parent_of(n, std::numeric_limits<std::size_t>::max());
    if (max_children == 0 || n == 0) {
        return parent_of;
    }

    std::size_t next_parent = 0;
    std::vector<SgId> queue;
    queue.reserve(max_children * 2U + 8U);

    for (SgId seed = 0; seed < n; ++seed) {
        if (parent_of[seed] != std::numeric_limits<std::size_t>::max()) {
            continue;
        }
        parent_of[seed] = next_parent;
        queue.clear();
        queue.push_back(seed);
        std::size_t head = 0;
        std::size_t assigned = 1;

        while (head < queue.size() && assigned < max_children) {
            const auto current = queue[head++];
            for (const auto neighbor : adjacency[current]) {
                if (parent_of[neighbor] != std::numeric_limits<std::size_t>::max()) {
                    continue;
                }
                parent_of[neighbor] = next_parent;
                queue.push_back(neighbor);
                ++assigned;
                if (assigned >= max_children) {
                    break;
                }
            }
        }

        ++next_parent;
    }

    return parent_of;
}

DijkstraTrace dijkstra_trace(
    const Graph& graph,
    const std::vector<std::pair<VertexId, EdgeWeight>>& seeds,
    EdgeWeight max_radius,
    const std::function<void(VertexId, EdgeWeight, std::size_t, const DistanceMap&)>& on_settle = {}
) {
    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    DijkstraTrace trace;

    for (const auto& [seed, value] : seeds) {
        auto it = trace.dist.find(seed);
        if (it == trace.dist.end() || value < it->second) {
            trace.dist[seed] = value;
            pq.push({value, seed});
        }
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();

        const auto best_it = trace.dist.find(u);
        if (best_it == trace.dist.end() || current != best_it->second) {
            continue;
        }
        if (current > max_radius) {
            continue;
        }

        trace.settled.push_back({u, current});
        if (on_settle) {
            on_settle(u, current, trace.settled.size(), trace.dist);
        }
        for (const auto& [v, w] : graph.neighbors(u)) {
            ++trace.relax_attempts;
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next > max_radius) {
                continue;
            }
            auto it = trace.dist.find(v);
            if (it == trace.dist.end() || next < it->second) {
                trace.dist[v] = next;
                ++trace.successful_relaxes;
                ++trace.pq_pushes;
                pq.push({next, v});
            }
        }
    }

    return trace;
}

DijkstraTrace dijkstra_trace_sorted_rows(
    const std::vector<std::int32_t>& sorted_row_index,
    const std::vector<std::vector<std::pair<VertexId, EdgeWeight>>>& sorted_rows,
    const std::vector<std::pair<VertexId, EdgeWeight>>& seeds,
    EdgeWeight max_radius,
    bool enable_row_truncation,
    const std::function<void(VertexId, EdgeWeight, std::size_t, const DistanceMap&)>& on_settle = {}
) {
    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    DijkstraTrace trace;

    for (const auto& [seed, value] : seeds) {
        auto it = trace.dist.find(seed);
        if (it == trace.dist.end() || value < it->second) {
            trace.dist[seed] = value;
            pq.push({value, seed});
        }
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();

        const auto best_it = trace.dist.find(u);
        if (best_it == trace.dist.end() || current != best_it->second) {
            continue;
        }
        if (current > max_radius) {
            continue;
        }

        trace.settled.push_back({u, current});
        if (on_settle) {
            on_settle(u, current, trace.settled.size(), trace.dist);
        }

        const auto& row = lookup_sorted_skeleton_row(u, sorted_row_index, sorted_rows);
        for (std::size_t i = 0; i < row.size(); ++i) {
            const auto [v, w] = row[i];
            ++trace.relax_attempts;
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next > max_radius) {
                if (enable_row_truncation) {
                    ++trace.num_rows_truncated;
                    trace.num_exits_skipped_by_truncation += row.size() - i;
                    break;
                }
                continue;
            }
            auto it = trace.dist.find(v);
            if (it == trace.dist.end() || next < it->second) {
                trace.dist[v] = next;
                ++trace.successful_relaxes;
                ++trace.pq_pushes;
                pq.push({next, v});
            }
        }
    }

    return trace;
}

DijkstraTrace dijkstra_trace_subgraph_rows(
    const std::vector<std::vector<SgId>>& inverted_index_fast,
    const std::vector<std::unordered_map<VertexId, std::vector<std::pair<VertexId, EdgeWeight>>>>& explicit_rows,
    const std::vector<FactorizedTransferSubgraphModel>& factorized_models,
    const std::vector<std::uint8_t>& factorized_enabled,
    const std::vector<std::pair<VertexId, EdgeWeight>>& seeds,
    EdgeWeight max_radius,
    bool use_factorized_transfer,
    std::size_t* factorized_rows_used = nullptr,
    std::size_t* factorized_hubs_used = nullptr,
    std::size_t* factorized_exits_emitted = nullptr,
    const std::function<void(VertexId, EdgeWeight, std::size_t, const DistanceMap&)>& on_settle = {}
) {
    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    DijkstraTrace trace;

    for (const auto& [seed, value] : seeds) {
        auto it = trace.dist.find(seed);
        if (it == trace.dist.end() || value < it->second) {
            trace.dist[seed] = value;
            pq.push({value, seed});
        }
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();

        const auto best_it = trace.dist.find(u);
        if (best_it == trace.dist.end() || current != best_it->second) {
            continue;
        }
        if (current > max_radius) {
            continue;
        }

        trace.settled.push_back({u, current});
        if (on_settle) {
            on_settle(u, current, trace.settled.size(), trace.dist);
        }

        if (u >= inverted_index_fast.size()) {
            continue;
        }
        for (const auto sg_id : inverted_index_fast[u]) {
            const auto row = materialize_subgraph_exit_candidates(
                u,
                current,
                sg_id,
                explicit_rows,
                factorized_models,
                factorized_enabled,
                use_factorized_transfer
            );
            if (row.used_factorized) {
                if (factorized_rows_used != nullptr) {
                    ++(*factorized_rows_used);
                }
                if (factorized_hubs_used != nullptr) {
                    *factorized_hubs_used += row.hubs_used;
                }
                if (factorized_exits_emitted != nullptr) {
                    *factorized_exits_emitted += row.exits.size();
                }
            }
            for (const auto& exit : row.exits) {
                ++trace.relax_attempts;
                const auto next = exit.candidate_distance;
                if (next > max_radius) {
                    continue;
                }
                auto [it, inserted] = trace.dist.emplace(exit.border, next);
                if (inserted || next < it->second) {
                    if (!inserted) {
                        it->second = next;
                    }
                    pq.push({next, exit.border});
                    ++trace.successful_relaxes;
                    ++trace.pq_pushes;
                }
            }
        }
    }

    return trace;
}

bool is_fully_covered(
    const Subgraph& sg,
    const DistanceMap& boundary_distances,
    EdgeWeight radius,
    FcRule rule
) {
    if (rule == FcRule::Disabled) {
        return false;
    }
    if (rule == FcRule::UpperBoundCandidate) {
        double ub = std::numeric_limits<double>::infinity();
        for (const auto& [b, rb] : sg.rb_map) {
            const auto it = boundary_distances.find(b);
            if (it == boundary_distances.end()) {
                continue;
            }
            ub = std::min(ub, static_cast<double>(it->second) + rb.to_double());
        }
        return ub <= static_cast<double>(radius);
    }

    bool all_covered = true;
    bool paper_guard = false;
    for (const auto b : sg.bound_vertices) {
        const auto it = boundary_distances.find(b);
        if (it == boundary_distances.end() || it->second > radius) {
            all_covered = false;
            break;
        }
        const auto rb_it = sg.rb_map.find(b);
        if (rb_it != sg.rb_map.end()) {
            const auto total = rb_it->second.to_double() + static_cast<double>(it->second);
            if (total <= static_cast<double>(radius)) {
                paper_guard = true;
            }
        }
    }
    if (!all_covered) {
        return false;
    }
    if (rule == FcRule::AllBordersVisited) {
        return true;
    }
    return paper_guard;
}

bool is_fully_covered(
    const Subgraph& sg,
    const std::vector<EdgeWeight>& boundary_distances,
    const std::vector<std::uint32_t>& boundary_stamps,
    std::uint32_t boundary_epoch,
    EdgeWeight radius,
    FcRule rule
) {
    if (rule == FcRule::Disabled) {
        return false;
    }
    const auto boundary_distance = [&](VertexId v) -> EdgeWeight {
        return (v < boundary_stamps.size() && boundary_stamps[v] == boundary_epoch)
            ? boundary_distances[v]
            : kInfWeight;
    };
    if (rule == FcRule::UpperBoundCandidate) {
        double ub = std::numeric_limits<double>::infinity();
        for (const auto& [b, rb] : sg.rb_map) {
            const auto distance = boundary_distance(b);
            if (distance == kInfWeight) {
                continue;
            }
            ub = std::min(ub, static_cast<double>(distance) + rb.to_double());
        }
        return ub <= static_cast<double>(radius);
    }

    bool all_covered = true;
    for (const auto b : sg.bound_vertices) {
        const auto distance = boundary_distance(b);
        if (distance == kInfWeight || distance > radius) {
            all_covered = false;
            if (rule == FcRule::AllBordersVisited) {
                return false;
            }
        }
    }
    if (rule == FcRule::AllBordersVisited) {
        return true;
    }
    if (!all_covered) {
        return false;
    }

    for (const auto b : sg.bound_vertices) {
        const auto distance = boundary_distance(b);
        if (distance <= radius) {
            const auto rb_it = sg.rb_map.find(b);
            if (rb_it != sg.rb_map.end()) {
                const auto rb = rb_it->second;
                if (distance <= radius && distance <= kInfWeight - static_cast<EdgeWeight>(rb.whole)) {
                    if (static_cast<double>(distance) + rb.to_double() <= static_cast<double>(radius)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

EdgeWeight object_distance_from_seed(
    const Subgraph& sg,
    const DistanceMap& seed_distances,
    const MovingObject& object
) {
    const auto left_it = seed_distances.find(object.edge.first);
    const auto right_it = seed_distances.find(object.edge.second);
    EdgeWeight best = kInfWeight;
    if (left_it != seed_distances.end() && left_it->second <= kInfWeight - object.offset) {
        best = std::min(best, static_cast<EdgeWeight>(left_it->second + object.offset));
    }
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    if (right_it != seed_distances.end() && right_it->second <= kInfWeight - right_cost) {
        best = std::min(best, static_cast<EdgeWeight>(right_it->second + right_cost));
    }
    return best;
}

EdgeWeight object_distance_from_vertex(
    const Subgraph& sg,
    VertexId query_vertex,
    const MovingObject& object
) {
    EdgeWeight best = kInfWeight;
    const auto left = sg.distance.get_or_inf(query_vertex, object.edge.first);
    if (left != kInfWeight && left <= kInfWeight - object.offset) {
        best = std::min(best, static_cast<EdgeWeight>(left + object.offset));
    }
    const auto right = sg.distance.get_or_inf(query_vertex, object.edge.second);
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    if (right != kInfWeight && right <= kInfWeight - right_cost) {
        best = std::min(best, static_cast<EdgeWeight>(right + right_cost));
    }
    return best;
}

EdgeWeight object_distance_from_boundary(
    const Subgraph& sg,
    VertexId boundary,
    EdgeWeight boundary_distance,
    const MovingObject& object
) {
    const auto left = sg.distance.get_or_inf(boundary, object.edge.first);
    const auto right = sg.distance.get_or_inf(boundary, object.edge.second);
    EdgeWeight best = kInfWeight;
    if (left != kInfWeight && boundary_distance <= kInfWeight - left &&
        boundary_distance + left <= kInfWeight - object.offset) {
        best = std::min(best, static_cast<EdgeWeight>(boundary_distance + left + object.offset));
    }
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    if (right != kInfWeight && boundary_distance <= kInfWeight - right &&
        boundary_distance + right <= kInfWeight - right_cost) {
        best = std::min(best, static_cast<EdgeWeight>(boundary_distance + right + right_cost));
    }
    return best;
}

EdgeWeight object_lower_bound_from_subgraph(
    const Subgraph& sg,
    EdgeWeight subgraph_lb,
    const MovingObject& object
) {
    if (subgraph_lb == kInfWeight) {
        return kInfWeight;
    }

    const auto lookup_md = [&](VertexId v) -> EdgeWeight {
        if (sg.bound_vertices.contains(v)) {
            return 0;
        }
        const auto it = sg.internal_to_nearest_border_dist.find(v);
        return (it == sg.internal_to_nearest_border_dist.end()) ? kInfWeight : it->second;
    };

    const auto md_left = lookup_md(object.edge.first);
    const auto md_right = lookup_md(object.edge.second);
    EdgeWeight suffix = kInfWeight;
    if (md_left != kInfWeight && md_left <= kInfWeight - object.offset) {
        suffix = std::min(suffix, static_cast<EdgeWeight>(md_left + object.offset));
    }
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    if (md_right != kInfWeight && md_right <= kInfWeight - right_cost) {
        suffix = std::min(suffix, static_cast<EdgeWeight>(md_right + right_cost));
    }
    if (suffix == kInfWeight || subgraph_lb > kInfWeight - suffix) {
        return kInfWeight;
    }
    return static_cast<EdgeWeight>(subgraph_lb + suffix);
}

EdgeWeight subgraph_lower_bound(const Subgraph& sg, const DistanceMap& skeleton_distances) {
    EdgeWeight best = kInfWeight;
    for (const auto b : sg.bound_vertices) {
        const auto it = skeleton_distances.find(b);
        if (it != skeleton_distances.end()) {
            best = std::min(best, it->second);
        }
    }
    return best;
}

EdgeWeight subgraph_lower_bound(const Subgraph& sg, const std::vector<EdgeWeight>& skeleton_distances) {
    EdgeWeight best = kInfWeight;
    for (const auto b : sg.bound_vertices) {
        if (b < skeleton_distances.size()) {
            best = std::min(best, skeleton_distances[b]);
        }
    }
    return best;
}

std::optional<VertexId> query_vertex_if_exact_vertex(const QueryPoint& query, const Graph& graph) {
    const auto edge = ordered_edge(query.edge.first, query.edge.second);
    const auto weight = edge_weight_any_or_null(graph, edge);
    if (!weight.has_value()) {
        return std::nullopt;
    }
    if (query.offset == 0) {
        return edge.first;
    }
    if (query.offset == *weight) {
        return edge.second;
    }
    return std::nullopt;
}

template <typename Heap>
void push_bounded_candidate(Heap& heap, std::size_t limit, const typename Heap::value_type& value) {
    if (limit == 0) {
        return;
    }
    if (heap.size() < limit) {
        heap.push(value);
        return;
    }
    if (value < heap.top()) {
        heap.pop();
        heap.push(value);
    }
}

bool fully_visited_edge(EdgeWeight du, EdgeWeight dv, EdgeWeight weight, EdgeWeight radius) {
    if (du <= radius && dv <= radius) {
        return true;
    }
    if (du == kInfWeight || dv == kInfWeight) {
        return false;
    }
    const auto lhs =
        static_cast<std::uint64_t>(du) + static_cast<std::uint64_t>(dv) +
        static_cast<std::uint64_t>(weight);
    const auto rhs = 2ULL * static_cast<std::uint64_t>(radius);
    return lhs <= rhs;
}

template <typename DistanceLookup, typename MarkSeen>
void collect_partial_subgraph_objects(
    const std::vector<PopulatedEdgeObjects>& populated_edges,
    DistanceLookup&& lookup_distance,
    EdgeWeight radius,
    const IndexedMovingObjectSet& objects,
    MarkSeen&& mark_seen,
    RangeQueryResult& result
) {
    for (const auto& populated : populated_edges) {
        const auto& edge = populated.edge;
        const auto weight = populated.edge_weight;
        const auto du = lookup_distance(edge.first);
        const auto dv = lookup_distance(edge.second);
        if (du == kInfWeight && dv == kInfWeight) {
            continue;
        }

        if (fully_visited_edge(du, dv, weight, radius)) {
            for (const auto obj_id : populated.object_ids) {
                if (mark_seen(obj_id)) {
                    result.object_ids.push_back(obj_id);
                    ++result.auto_included_objects;
                    ++result.partial_edge_auto_included_objects;
                }
            }
            continue;
        }

        for (const auto obj_id : populated.object_ids) {
            const auto& object = objects[obj_id];
            ++result.exact_checked_objects;

            EdgeWeight best = kInfWeight;
            if (du != kInfWeight && du <= kInfWeight - object.offset) {
                best = std::min(best, static_cast<EdgeWeight>(du + object.offset));
            }
            const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
            if (dv != kInfWeight && dv <= kInfWeight - right_cost) {
                best = std::min(best, static_cast<EdgeWeight>(dv + right_cost));
            }

            if (best <= radius && mark_seen(obj_id)) {
                result.object_ids.push_back(object.unique_id);
                ++result.exact_returned_objects;
            }
        }
    }
}

template <typename DistanceProvider>
KnnQueryResult compute_knn(
    const IndexedMovingObjectSet& objects,
    std::size_t k,
    DistanceProvider provider
) {
    std::vector<KnnItem> items;
    items.reserve(objects.size());
    for (const auto& object : objects.objects()) {
        const auto distance = provider(object);
        if (distance == kInfWeight) {
            continue;
        }
        items.push_back({object.unique_id, distance});
    }

    std::sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.distance != rhs.distance) {
            return lhs.distance < rhs.distance;
        }
        return lhs.id < rhs.id;
    });
    if (items.size() > k) {
        items.resize(k);
    }
    return KnnQueryResult{std::move(items)};
}

}  // namespace

SkeletonIndex SkeletonIndex::build(const Graph& global, std::vector<Subgraph> subgraphs) {
    SkeletonIndex index;
    index.global_ = &global;
    index.subgraphs_ = std::move(subgraphs);
    index.subgraph_boundaries_.resize(index.subgraphs_.size());
    index.subgraph_undirected_edges_.resize(index.subgraphs_.size());
    index.subgraph_clique_rows_.resize(index.subgraphs_.size());
    index.subgraph_local_indices_.resize(index.subgraphs_.size());

    for (auto& sg : index.subgraphs_) {
        for (const auto v : sg.graph.vertices()) {
            index.max_vertex_id_ = std::max(index.max_vertex_id_, v);
        }
        auto& boundaries = index.subgraph_boundaries_[sg.id];
        boundaries.assign(sg.bound_vertices.begin(), sg.bound_vertices.end());
        std::sort(boundaries.begin(), boundaries.end());
        index.subgraph_undirected_edges_[sg.id] = sg.graph.undirected_edges();
        auto& local_index = index.subgraph_local_indices_[sg.id];
        const auto vertices = sg.graph.vertices();
        local_index.local_ids.reserve(vertices.size() * 2U + 1U);
        local_index.adjacency.assign(vertices.size(), {});
        for (std::size_t local_id = 0; local_id < vertices.size(); ++local_id) {
            local_index.local_ids.emplace(vertices[local_id], static_cast<std::uint32_t>(local_id));
        }
        for (std::size_t local_id = 0; local_id < vertices.size(); ++local_id) {
            const auto vertex = vertices[local_id];
            const auto& neighbors = sg.graph.neighbors(vertex);
            auto& out = local_index.adjacency[local_id];
            out.reserve(neighbors.size());
            for (const auto& [neighbor, weight] : neighbors) {
                const auto neighbor_it = local_index.local_ids.find(neighbor);
                if (neighbor_it != local_index.local_ids.end()) {
                    out.push_back({neighbor_it->second, weight});
                }
            }
        }
        auto& clique_rows = index.subgraph_clique_rows_[sg.id];
        clique_rows.reserve(boundaries.size());
        for (const auto b : boundaries) {
            std::vector<std::pair<VertexId, EdgeWeight>> row;
            row.reserve(boundaries.size());
            for (const auto other : boundaries) {
                if (b == other) {
                    continue;
                }
                const auto distance = sg.distance.get_or_inf(b, other);
                if (distance != kInfWeight) {
                    row.push_back({other, distance});
                }
            }
            std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
                if (lhs.second != rhs.second) {
                    return lhs.second < rhs.second;
                }
                return lhs.first < rhs.first;
            });
            clique_rows.emplace(b, std::move(row));
        }
        for (const auto v : sg.internal_vertices) {
            index.vertex_to_subgraph_[v] = sg.id;
        }
        for (const auto& [edge, weight] : sg.graph.undirected_edges()) {
            (void)weight;
            index.edge_to_subgraph_[ordered_edge(edge.first, edge.second)] = sg.id;
        }

        for (const auto b : boundaries) {
            index.skeleton_.insert(b);
            index.inverted_map_[b].push_back(sg.id);
            for (const auto other : boundaries) {
                if (b == other) {
                    continue;
                }
                const auto distance = sg.distance.get_or_inf(b, other);
                if (distance != kInfWeight) {
                    index.skeleton_.set_min_undirected_edge(b, other, distance);
                }
            }
        }
    }

    index.inverted_index_fast_.resize(static_cast<std::size_t>(index.max_vertex_id_) + 1U);
    for (const auto& [vertex, ids] : index.inverted_map_) {
        index.inverted_index_fast_[vertex] = ids;
    }
    index.vertex_to_subgraph_fast_.assign(
        static_cast<std::size_t>(index.max_vertex_id_) + 1U,
        std::numeric_limits<SgId>::max()
    );
    for (const auto& [vertex, sg_id] : index.vertex_to_subgraph_) {
        index.vertex_to_subgraph_fast_[vertex] = sg_id;
    }

    std::vector<std::unordered_set<SgId>> adjacency_sets(index.subgraphs_.size());
    for (const auto& [border, ids] : index.inverted_map_) {
        (void)border;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            for (std::size_t j = i + 1; j < ids.size(); ++j) {
                adjacency_sets[ids[i]].insert(ids[j]);
                adjacency_sets[ids[j]].insert(ids[i]);
            }
        }
    }
    index.subgraph_adjacency_.resize(index.subgraphs_.size());
    for (std::size_t sg_id = 0; sg_id < adjacency_sets.size(); ++sg_id) {
        auto& out = index.subgraph_adjacency_[sg_id];
        out.assign(adjacency_sets[sg_id].begin(), adjacency_sets[sg_id].end());
        std::sort(out.begin(), out.end());
    }

    index.rebuild_sorted_skeleton_rows();

    return index;
}

void SkeletonIndex::configure_factorized_transfer(
    double max_arc_ratio,
    std::size_t min_border_count
) {
    factorized_transfer_models_.assign(subgraphs_.size(), FactorizedTransferSubgraphModel{});
    factorized_transfer_enabled_.assign(subgraphs_.size(), 0U);
    for (const auto& sg : subgraphs_) {
        if (sg.id >= factorized_transfer_models_.size()) {
            continue;
        }
        if (sg.bound_vertices.size() < min_border_count) {
            continue;
        }
        auto model = build_factorized_transfer_model(sg);
        if (!model.feasible || model.factorized_arc_ratio > max_arc_ratio) {
            factorized_transfer_models_[sg.id] = std::move(model);
            continue;
        }
        factorized_transfer_enabled_[sg.id] = 1U;
        factorized_transfer_models_[sg.id] = std::move(model);
    }
}

std::size_t SkeletonIndex::configure_gstar_shortcuts(
    std::size_t leaf_hops,
    std::size_t shortcuts_per_subgraph
) {
    gstar_shortcut_edges_added_ = 0;
    if (subgraphs_.empty() || leaf_hops == 0 || shortcuts_per_subgraph == 0) {
        return 0;
    }

    std::vector<VertexId> representative_boundaries(subgraphs_.size(), kInvalidVertex);
    for (const auto& sg : subgraphs_) {
        representative_boundaries[sg.id] =
            choose_representative_boundary(sg, subgraph_boundaries_[sg.id]);
    }

    std::unordered_set<Edge, PairHash> realized_shortcuts;
    realized_shortcuts.reserve(subgraphs_.size() * shortcuts_per_subgraph * 32U + 1U);
    std::unordered_set<Edge, PairHash> realized_subgraph_pairs;
    realized_subgraph_pairs.reserve(subgraphs_.size() * shortcuts_per_subgraph * 2U + 1U);

    std::vector<EdgeWeight> dist(static_cast<std::size_t>(max_vertex_id_) + 1U, kInfWeight);
    std::vector<std::uint32_t> stamp(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
    std::uint32_t epoch = 0U;
    using QueueItem = std::pair<EdgeWeight, VertexId>;

    for (const auto& sg : subgraphs_) {
        const auto source_rep = representative_boundaries[sg.id];
        if (source_rep == kInvalidVertex) {
            continue;
        }

        const auto target_sgs =
            gather_shortcut_candidate_subgraphs(subgraph_adjacency_, sg.id, leaf_hops);
        if (target_sgs.empty()) {
            continue;
        }

        std::unordered_map<VertexId, SgId> target_rep_to_sg;
        target_rep_to_sg.reserve(target_sgs.size() * 2U + 1U);
        for (const auto target_sg : target_sgs) {
            if (target_sg >= representative_boundaries.size()) {
                continue;
            }
            const auto target_rep = representative_boundaries[target_sg];
            if (target_rep == kInvalidVertex || target_rep == source_rep) {
                continue;
            }
            target_rep_to_sg.emplace(target_rep, target_sg);
        }
        if (target_rep_to_sg.empty()) {
            continue;
        }

        ++epoch;
        if (epoch == 0U) {
            std::fill(stamp.begin(), stamp.end(), 0U);
            epoch = 1U;
        }

        std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
        stamp[source_rep] = epoch;
        dist[source_rep] = 0;
        pq.push({0, source_rep});

        std::vector<SgId> selected_target_sgs;
        selected_target_sgs.reserve(shortcuts_per_subgraph);
        std::unordered_set<VertexId> settled_target_reps;
        settled_target_reps.reserve(target_rep_to_sg.size() * 2U + 1U);

        while (!pq.empty() && selected_target_sgs.size() < shortcuts_per_subgraph) {
            const auto [current, u] = pq.top();
            pq.pop();
            if (u >= stamp.size() || stamp[u] != epoch || current != dist[u]) {
                continue;
            }

            const auto rep_it = target_rep_to_sg.find(u);
            if (rep_it != target_rep_to_sg.end() && settled_target_reps.insert(u).second) {
                const auto pair_key = ordered_edge(
                    static_cast<VertexId>(sg.id),
                    static_cast<VertexId>(rep_it->second)
                );
                if (!realized_subgraph_pairs.contains(pair_key)) {
                    selected_target_sgs.push_back(rep_it->second);
                }
                if (selected_target_sgs.size() >= shortcuts_per_subgraph) {
                    break;
                }
            }

            for (const auto& [v, w] : skeleton_.neighbors(u)) {
                if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                    continue;
                }
                const auto next = static_cast<EdgeWeight>(current + w);
                if (v >= stamp.size()) {
                    continue;
                }
                if (stamp[v] != epoch || next < dist[v]) {
                    stamp[v] = epoch;
                    dist[v] = next;
                    pq.push({next, v});
                }
            }
        }

        for (const auto target_sg : selected_target_sgs) {
            const auto pair_key = ordered_edge(
                static_cast<VertexId>(sg.id),
                static_cast<VertexId>(target_sg)
            );
            if (!realized_subgraph_pairs.insert(pair_key).second) {
                continue;
            }

            const auto& source_boundaries = subgraph_boundaries_[sg.id];
            const auto& target_boundaries = subgraph_boundaries_[target_sg];
            if (source_boundaries.empty() || target_boundaries.empty()) {
                continue;
            }

            std::unordered_set<VertexId> target_boundary_set(
                target_boundaries.begin(),
                target_boundaries.end()
            );

            for (const auto source_b : source_boundaries) {
                ++epoch;
                if (epoch == 0U) {
                    std::fill(stamp.begin(), stamp.end(), 0U);
                    epoch = 1U;
                }

                std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> boundary_pq;
                stamp[source_b] = epoch;
                dist[source_b] = 0;
                boundary_pq.push({0, source_b});

                std::unordered_map<VertexId, EdgeWeight> target_distances;
                target_distances.reserve(target_boundaries.size() * 2U + 1U);

                while (!boundary_pq.empty() && target_distances.size() < target_boundaries.size()) {
                    const auto [current, u] = boundary_pq.top();
                    boundary_pq.pop();
                    if (u >= stamp.size() || stamp[u] != epoch || current != dist[u]) {
                        continue;
                    }

                    if (target_boundary_set.contains(u)) {
                        target_distances.emplace(u, current);
                        if (target_distances.size() >= target_boundaries.size()) {
                            break;
                        }
                    }

                    for (const auto& [v, w] : skeleton_.neighbors(u)) {
                        if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                            continue;
                        }
                        const auto next = static_cast<EdgeWeight>(current + w);
                        if (v >= stamp.size()) {
                            continue;
                        }
                        if (stamp[v] != epoch || next < dist[v]) {
                            stamp[v] = epoch;
                            dist[v] = next;
                            boundary_pq.push({next, v});
                        }
                    }
                }

                for (const auto target_b : target_boundaries) {
                    const auto dist_it = target_distances.find(target_b);
                    if (dist_it == target_distances.end()) {
                        continue;
                    }
                    const auto shortcut = ordered_edge(source_b, target_b);
                    const auto existing = skeleton_.get_weight(shortcut.first, shortcut.second);
                    if (!existing.has_value() || *existing > dist_it->second) {
                        skeleton_.set_min_undirected_edge(shortcut.first, shortcut.second, dist_it->second);
                    }
                    realized_shortcuts.insert(shortcut);
                }
            }
        }
    }

    gstar_shortcut_edges_added_ = realized_shortcuts.size();
    if (gstar_shortcut_edges_added_ != 0) {
        rebuild_sorted_skeleton_rows();
    }
    return gstar_shortcut_edges_added_;
}

const Graph& SkeletonIndex::skeleton() const {
    return skeleton_;
}

const std::vector<Subgraph>& SkeletonIndex::subgraphs() const {
    return subgraphs_;
}

const std::unordered_map<Edge, SgId, PairHash>& SkeletonIndex::edge_to_subgraph() const {
    return edge_to_subgraph_;
}

const std::unordered_map<VertexId, SgId>& SkeletonIndex::vertex_to_subgraph() const {
    return vertex_to_subgraph_;
}

std::size_t SkeletonIndex::gstar_shortcut_edges_added() const {
    return gstar_shortcut_edges_added_;
}

void SkeletonIndex::rebuild_sorted_skeleton_rows() {
    sorted_skeleton_row_index_.assign(
        static_cast<std::size_t>(max_vertex_id_) + 1U,
        -1
    );
    sorted_skeleton_row_storage_.clear();
    sorted_skeleton_row_storage_.reserve(skeleton_.size());
    for (const auto border : skeleton_.vertices()) {
        std::vector<std::pair<VertexId, EdgeWeight>> row;
        const auto& neighbors = skeleton_.neighbors(border);
        row.reserve(neighbors.size());
        for (const auto& [other, weight] : neighbors) {
            row.push_back({other, weight});
        }
        std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.second != rhs.second) {
                return lhs.second < rhs.second;
            }
            return lhs.first < rhs.first;
        });
        sorted_skeleton_row_index_[border] =
            static_cast<std::int32_t>(sorted_skeleton_row_storage_.size());
        sorted_skeleton_row_storage_.push_back(std::move(row));
    }
}

const Subgraph& SkeletonIndex::require_initial_subgraph(const Edge& edge) const {
    const auto normalized = ordered_edge(edge.first, edge.second);
    const auto it = edge_to_subgraph_.find(normalized);
    if (it == edge_to_subgraph_.end()) {
        throw std::runtime_error("query edge not found in any subgraph");
    }
    return subgraphs_.at(it->second);
}

RangeQueryResult SkeletonIndex::range_query(
    const QueryPoint& query,
    EdgeWeight radius,
    const IndexedMovingObjectSet& objects,
    FcRule rule,
    std::size_t query_id,
    std::vector<BorderExposureEvent>* exposure_events,
    std::vector<CliqueRowShadowRecord>* row_shadow_records,
    bool enable_row_truncation,
    bool use_factorized_transfer
) const {
    using Clock = std::chrono::steady_clock;
    const auto t_started = Clock::now();
    RangeQueryResult result;
    if (range_object_seen_stamp_.size() != objects.size()) {
        range_object_seen_stamp_.assign(objects.size(), 0U);
        range_object_seen_epoch_ = 0U;
    }
    if (range_object_seen_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(range_object_seen_stamp_.begin(), range_object_seen_stamp_.end(), 0U);
        range_object_seen_epoch_ = 0U;
    }
    ++range_object_seen_epoch_;
    const auto object_seen_epoch = range_object_seen_epoch_;
    const auto mark_seen_object = [&](ObjId obj_id) -> bool {
        const auto index = static_cast<std::size_t>(obj_id);
        if (index >= range_object_seen_stamp_.size()) {
            return false;
        }
        if (range_object_seen_stamp_[index] == object_seen_epoch) {
            return false;
        }
        range_object_seen_stamp_[index] = object_seen_epoch;
        return true;
    };
    std::unordered_map<std::pair<SgId, VertexId>, std::size_t, PairHash> exposure_index;
    struct PendingRowShadow {
        std::size_t index{0};
        std::vector<std::pair<VertexId, EdgeWeight>> pushed_exits;
    };
    const bool need_exposure_state = exposure_events != nullptr;
    const bool need_row_shadow = row_shadow_records != nullptr;
    const bool fast_dense_path = !need_exposure_state && !need_row_shadow && !use_factorized_transfer;
    std::vector<std::vector<VertexId>> sg_seed_borders =
        need_exposure_state ? std::vector<std::vector<VertexId>>(subgraphs_.size()) : std::vector<std::vector<VertexId>>{};
    std::vector<std::uint8_t> sg_fully_covered =
        need_exposure_state ? std::vector<std::uint8_t>(subgraphs_.size(), 0U) : std::vector<std::uint8_t>{};
    std::vector<std::size_t> sg_touch_events =
        need_exposure_state ? std::vector<std::size_t>(subgraphs_.size(), 0U) : std::vector<std::size_t>{};
    std::vector<std::uint8_t> shadow_seen_subgraphs =
        need_row_shadow ? std::vector<std::uint8_t>(subgraphs_.size(), 0U) : std::vector<std::uint8_t>{};
    std::vector<PendingRowShadow> pending_rows;
    std::size_t factorized_rows_used = 0;
    std::size_t factorized_hubs_used = 0;
    std::size_t factorized_exits_emitted = 0;

    const auto& initial = require_initial_subgraph(query.edge);
    const auto& initial_local_index = subgraph_local_indices_.at(initial.id);
    const auto initial_local = local_query_distances_compact(initial, initial_local_index, query, radius);
    const auto initial_seeds = boundary_query_seeds(initial, initial_local);
    const auto t_after_init = Clock::now();
    DijkstraTrace boundary_trace;
    DistanceMap boundary_distances;
    std::vector<VertexId> fast_touched_boundaries;
    std::vector<SgId> fast_touched_subgraphs;
    std::uint32_t fast_boundary_epoch = 0U;
    std::uint32_t fast_subgraph_epoch = 0U;
    if (use_factorized_transfer && exposure_events == nullptr && row_shadow_records == nullptr) {
        boundary_trace = dijkstra_trace_subgraph_rows(
            inverted_index_fast_,
            subgraph_clique_rows_,
            factorized_transfer_models_,
            factorized_transfer_enabled_,
            initial_seeds,
            radius,
            true,
            &factorized_rows_used,
            &factorized_hubs_used,
            &factorized_exits_emitted
        );
        boundary_distances = boundary_trace.dist;
    } else if (fast_dense_path) {
        using QueueItem = std::pair<EdgeWeight, VertexId>;
        std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
        if (skeleton_scratch_dist_.size() != static_cast<std::size_t>(max_vertex_id_) + 1U) {
            skeleton_scratch_dist_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
            skeleton_scratch_stamp_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
            skeleton_scratch_epoch_ = 0U;
        }
        if (skeleton_scratch_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(skeleton_scratch_stamp_.begin(), skeleton_scratch_stamp_.end(), 0U);
            skeleton_scratch_epoch_ = 0U;
        }
        ++skeleton_scratch_epoch_;
        fast_boundary_epoch = skeleton_scratch_epoch_;
        fast_touched_boundaries.reserve(initial_seeds.size() * 8U + 64U);
        const auto boundary_distance = [&](VertexId v) -> EdgeWeight {
            return (v < skeleton_scratch_stamp_.size() && skeleton_scratch_stamp_[v] == fast_boundary_epoch)
                ? skeleton_scratch_dist_[v]
                : kInfWeight;
        };
        const auto update_boundary = [&](VertexId v, EdgeWeight distance) {
            if (skeleton_scratch_stamp_[v] == fast_boundary_epoch) {
                if (distance >= skeleton_scratch_dist_[v]) {
                    return false;
                }
                skeleton_scratch_dist_[v] = distance;
            } else {
                skeleton_scratch_stamp_[v] = fast_boundary_epoch;
                skeleton_scratch_dist_[v] = distance;
                fast_touched_boundaries.push_back(v);
            }
            pq.push({distance, v});
            return true;
        };
        for (const auto& [seed, value] : initial_seeds) {
            update_boundary(seed, value);
        }
        while (!pq.empty()) {
            const auto [current, u] = pq.top();
            pq.pop();
            if (current != boundary_distance(u)) {
                continue;
            }
            if (current > radius) {
                continue;
            }
            const auto& row = lookup_sorted_skeleton_row(u, sorted_skeleton_row_index_, sorted_skeleton_row_storage_);
            for (std::size_t i = 0; i < row.size(); ++i) {
                const auto [v, w] = row[i];
                ++boundary_trace.relax_attempts;
                if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                    continue;
                }
                const auto next = static_cast<EdgeWeight>(current + w);
                if (next > radius) {
                    if (enable_row_truncation) {
                        ++boundary_trace.num_rows_truncated;
                        boundary_trace.num_exits_skipped_by_truncation += row.size() - i;
                        break;
                    }
                    continue;
                }
                if (update_boundary(v, next)) {
                    ++boundary_trace.successful_relaxes;
                    ++boundary_trace.pq_pushes;
                }
            }
        }
        if (range_subgraph_touch_count_scratch_.size() != subgraphs_.size()) {
            range_subgraph_touch_count_scratch_.assign(subgraphs_.size(), 0U);
            range_subgraph_touch_stamp_scratch_.assign(subgraphs_.size(), 0U);
            range_subgraph_touch_epoch_ = 0U;
        }
        if (range_subgraph_touch_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(range_subgraph_touch_stamp_scratch_.begin(), range_subgraph_touch_stamp_scratch_.end(), 0U);
            range_subgraph_touch_epoch_ = 0U;
        }
        ++range_subgraph_touch_epoch_;
        fast_subgraph_epoch = range_subgraph_touch_epoch_;
        fast_touched_subgraphs.reserve(fast_touched_boundaries.size() * 2U + 8U);
        for (const auto vertex : fast_touched_boundaries) {
            if (vertex >= inverted_index_fast_.size()) {
                continue;
            }
            for (const auto sg_id : inverted_index_fast_[vertex]) {
                if (range_subgraph_touch_stamp_scratch_[sg_id] == fast_subgraph_epoch) {
                    ++range_subgraph_touch_count_scratch_[sg_id];
                    ++result.num_redundant_subgraph_reentries;
                } else {
                    range_subgraph_touch_stamp_scratch_[sg_id] = fast_subgraph_epoch;
                    range_subgraph_touch_count_scratch_[sg_id] = 1U;
                    fast_touched_subgraphs.push_back(sg_id);
                }
            }
        }
    } else if (!use_factorized_transfer) {
        boundary_trace = dijkstra_trace_sorted_rows(
            sorted_skeleton_row_index_,
            sorted_skeleton_row_storage_,
            initial_seeds,
            radius,
            enable_row_truncation,
            [&](VertexId border, EdgeWeight distance, std::size_t visit_order, const DistanceMap& current_dist) {
                if (row_shadow_records == nullptr) {
                    return;
                }
                const auto inv_it = inverted_map_.find(border);
                if (inv_it == inverted_map_.end()) {
                    return;
                }
                for (const auto sg_id : inv_it->second) {
                    if (shadow_seen_subgraphs[sg_id] != 0U) {
                        continue;
                    }
                    shadow_seen_subgraphs[sg_id] = 1U;
                    std::vector<std::pair<VertexId, EdgeWeight>> exits;
                    exits = lookup_subgraph_clique_row(border, subgraph_clique_rows_[sg_id]);

                    CliqueRowShadowRecord record;
                    record.query_id = query_id;
                    record.query_type = QueryType::Range;
                    record.subgraph_id = sg_id;
                    record.entry_border = border;
                    record.entry_visit_order = visit_order;
                    record.entry_distance = distance;
                    record.threshold_at_entry = radius;
                    record.num_exits_total = exits.size();
                    record.useful_prefix_len = exits.size();

                    PendingRowShadow pending;
                    pending.index = row_shadow_records->size();

                    for (std::size_t i = 0; i < exits.size(); ++i) {
                        const auto [exit_border, local] = exits[i];
                        if (distance == kInfWeight || local == kInfWeight || distance > kInfWeight - local) {
                            ++record.num_exits_threshold_dead;
                            record.useful_prefix_len = std::min(record.useful_prefix_len, i);
                            continue;
                        }
                        const auto cand = static_cast<EdgeWeight>(distance + local);
                        const auto current_it = current_dist.find(exit_border);
                        const auto current_best =
                            (current_it == current_dist.end()) ? kInfWeight : current_it->second;
                        const bool dist_dominated = cand >= current_best;
                        const bool threshold_dead = cand > radius;
                        if (dist_dominated) {
                            ++record.num_exits_dist_dominated;
                        }
                        if (threshold_dead) {
                            ++record.num_exits_threshold_dead;
                            record.useful_prefix_len = std::min(record.useful_prefix_len, i);
                        }
                        if (!dist_dominated) {
                            ++record.num_pq_pushes_from_this_row;
                            if (i >= record.useful_prefix_len) {
                                ++record.num_pq_pushes_beyond_useful_prefix;
                            }
                            pending.pushed_exits.push_back({exit_border, cand});
                        }
                    }

                    row_shadow_records->push_back(record);
                    pending_rows.push_back(std::move(pending));
                }
            }
        );
        boundary_distances = boundary_trace.dist;
    } else {
        boundary_trace = dijkstra_trace_subgraph_rows(
            inverted_index_fast_,
            subgraph_clique_rows_,
            factorized_transfer_models_,
            factorized_transfer_enabled_,
            initial_seeds,
            radius,
            true,
            &factorized_rows_used,
            &factorized_hubs_used,
            &factorized_exits_emitted
        );
        boundary_distances = boundary_trace.dist;
    }
    const auto t_after_skeleton_trace = Clock::now();
    result.initial_local_vertices = initial_local.reached_count;
    result.boundary_vertices_reached = fast_dense_path ? fast_touched_boundaries.size() : boundary_distances.size();
    result.num_clique_relax_attempts = boundary_trace.relax_attempts;
    result.num_successful_clique_relaxes = boundary_trace.successful_relaxes;
    result.num_pq_pushes_from_clique = boundary_trace.pq_pushes;
    result.num_rows_truncated = boundary_trace.num_rows_truncated;
    result.num_exits_skipped_by_truncation = boundary_trace.num_exits_skipped_by_truncation;
    result.factorized_rows_used = factorized_rows_used;
    result.factorized_hubs_used = factorized_hubs_used;
    result.factorized_exits_emitted = factorized_exits_emitted;

    auto ensure_exposure = [&](SgId sg_id, VertexId border, std::size_t visit_order, EdgeWeight distance) {
        if (exposure_events == nullptr) {
            return;
        }
        const auto key = std::make_pair(sg_id, border);
        if (const auto it = exposure_index.find(key); it != exposure_index.end()) {
            auto& event = exposure_events->at(it->second);
            event.visit_order = std::min(event.visit_order, visit_order);
            event.distance_from_query = std::min(event.distance_from_query, distance);
            return;
        }
        exposure_index[key] = exposure_events->size();
        exposure_events->push_back(BorderExposureEvent{
            query_id,
            QueryType::Range,
            sg_id,
            border,
            visit_order,
            false,
            false,
            false,
            false,
            false,
            distance,
            0,
        });
    };

    if (exposure_events != nullptr) {
        for (std::size_t order = 0; order < boundary_trace.settled.size(); ++order) {
            const auto [border, distance] = boundary_trace.settled[order];
            const auto inv_it = inverted_map_.find(border);
            if (inv_it == inverted_map_.end()) {
                continue;
            }
            for (const auto sg_id : inv_it->second) {
                ensure_exposure(sg_id, border, order + 1U, distance);
            }
        }
    }

    std::vector<SgId> hit_subgraphs;
    if (fast_dense_path) {
        hit_subgraphs.reserve(fast_touched_subgraphs.size());
        for (const auto sg_id : fast_touched_subgraphs) {
            if (sg_id != initial.id) {
                hit_subgraphs.push_back(sg_id);
            }
        }
    } else {
        std::vector<std::size_t> hit_counts(subgraphs_.size(), 0U);
        hit_subgraphs.reserve(boundary_distances.size());
        for (const auto& [vertex, distance] : boundary_distances) {
            if (distance > radius) {
                continue;
            }
            const auto inv_it = inverted_map_.find(vertex);
            if (inv_it == inverted_map_.end()) {
                continue;
            }
            for (const auto sg_id : inv_it->second) {
                ++sg_touch_events[sg_id];
                if (sg_touch_events[sg_id] > 1U) {
                    ++result.num_redundant_subgraph_reentries;
                }
                if (sg_id != initial.id) {
                    if (hit_counts[sg_id] == 0U) {
                        hit_subgraphs.push_back(sg_id);
                    }
                    ++hit_counts[sg_id];
                }
            }
        }
    }

    const bool initial_fc = fast_dense_path
        ? is_fully_covered(initial, skeleton_scratch_dist_, skeleton_scratch_stamp_, fast_boundary_epoch, radius, rule)
        : is_fully_covered(initial, boundary_distances, radius, rule);
    if (need_exposure_state) {
        sg_fully_covered[initial.id] = initial_fc ? 1U : 0U;
        for (const auto& [b, distance] : initial_seeds) {
            (void)distance;
            sg_seed_borders[initial.id].push_back(b);
        }
    }
    if (initial_fc) {
        ++result.fc_subgraphs;
        for (const auto obj_id : objects.objects_in(initial.id)) {
            if (mark_seen_object(obj_id)) {
                result.object_ids.push_back(obj_id);
                ++result.auto_included_objects;
                ++result.br_fc_included_objects;
            }
        }
    } else {
        ++result.pc_subgraphs;
        collect_partial_subgraph_objects(
            objects.populated_edges_in(initial.id),
            [&](VertexId v) { return lookup_local_distance(initial_local, v); },
            radius,
            objects,
            mark_seen_object,
            result
        );
    }

    for (const auto sg_id : hit_subgraphs) {
        const auto hit_count = fast_dense_path
            ? range_subgraph_touch_count_scratch_[sg_id]
            : 1U;
        if (fast_dense_path && hit_count == 0) {
            continue;
        }
        const auto& sg = subgraphs_.at(sg_id);
        const bool fully_covered = fast_dense_path
            ? is_fully_covered(sg, skeleton_scratch_dist_, skeleton_scratch_stamp_, fast_boundary_epoch, radius, rule)
            : is_fully_covered(sg, boundary_distances, radius, rule);
        if (need_exposure_state) {
            sg_fully_covered[sg.id] = fully_covered ? 1U : 0U;
        }
        if (fully_covered) {
            ++result.fc_subgraphs;
            if (need_exposure_state) {
                for (const auto b : subgraph_boundaries_[sg.id]) {
                    const auto it = boundary_distances.find(b);
                    if (it != boundary_distances.end() && it->second <= radius) {
                        sg_seed_borders[sg.id].push_back(b);
                    }
                }
            }
            for (const auto obj_id : objects.objects_in(sg.id)) {
                if (mark_seen_object(obj_id)) {
                    result.object_ids.push_back(obj_id);
                    ++result.auto_included_objects;
                    ++result.br_fc_included_objects;
                }
            }
            continue;
        }

        ++result.pc_subgraphs;
        std::vector<std::pair<VertexId, EdgeWeight>> seeds;
        seeds.reserve(subgraph_boundaries_[sg.id].size());
        if (fast_dense_path) {
            for (const auto b : subgraph_boundaries_[sg.id]) {
                if (b < skeleton_scratch_stamp_.size() && skeleton_scratch_stamp_[b] == fast_boundary_epoch) {
                    const auto distance = skeleton_scratch_dist_[b];
                    if (distance <= radius) {
                        seeds.push_back({b, distance});
                    }
                }
            }
        } else {
            for (const auto b : subgraph_boundaries_[sg.id]) {
                const auto it = boundary_distances.find(b);
                if (it != boundary_distances.end() && it->second <= radius) {
                    seeds.push_back({b, it->second});
                    if (need_exposure_state) {
                        sg_seed_borders[sg.id].push_back(b);
                    }
                }
            }
        }
        if (seeds.empty()) {
            continue;
        }
        const auto local_dist = local_dijkstra_compact(subgraph_local_indices_[sg.id], seeds, radius);
        collect_partial_subgraph_objects(
            objects.populated_edges_in(sg.id),
            [&](VertexId v) { return lookup_local_distance(local_dist, v); },
            radius,
            objects,
            mark_seen_object,
            result
        );
    }

    result.touched_subgraphs = fast_dense_path ? fast_touched_subgraphs.size() : hit_subgraphs.size() + 1U;
    if (need_exposure_state) {
        std::unordered_map<SgId, std::size_t> overlap_by_subgraph;
        for (const auto obj_id : result.object_ids) {
            ++overlap_by_subgraph[objects.object_subgraph(obj_id)];
        }
        for (SgId sg_id = 0; sg_id < sg_seed_borders.size(); ++sg_id) {
            const auto& borders = sg_seed_borders[sg_id];
            if (borders.empty()) {
                continue;
            }
            const auto overlap = overlap_by_subgraph.contains(sg_id) ? overlap_by_subgraph.at(sg_id) : 0U;
            const bool triggered_full_cover = sg_fully_covered[sg_id] != 0U;
            for (const auto border : borders) {
                const auto key = std::make_pair(sg_id, border);
                const auto it = exposure_index.find(key);
                if (it == exposure_index.end()) {
                    continue;
                }
                auto& event = exposure_events->at(it->second);
                event.triggered_full_cover = triggered_full_cover;
                event.eventual_result_overlap = overlap;
                event.was_needed_for_final_answer = overlap != 0U;
                event.was_only_used_for_traversal = (overlap == 0U && !triggered_full_cover);
            }
        }
    }
    if (fast_dense_path) {
        for (const auto sg_id : fast_touched_subgraphs) {
            const auto touches = range_subgraph_touch_count_scratch_[sg_id];
            if (touches > 1U) {
                result.num_useful_subgraph_reentries += touches - 1U;
            }
        }
    } else {
        for (SgId sg_id = 0; sg_id < sg_touch_events.size(); ++sg_id) {
            const auto touches = sg_touch_events[sg_id];
            if (touches <= 1U) {
                continue;
            }
            const auto has_overlap = !sg_seed_borders[sg_id].empty();
            const auto is_fc = sg_fully_covered[sg_id] != 0U;
            if (has_overlap || is_fc) {
                result.num_useful_subgraph_reentries += touches - 1U;
            }
        }
    }
    if (need_row_shadow) {
        std::unordered_set<VertexId> settled_vertices;
        settled_vertices.reserve(boundary_trace.settled.size() * 2U + 1U);
        for (const auto& [vertex, distance] : boundary_trace.settled) {
            (void)distance;
            settled_vertices.insert(vertex);
        }
        for (const auto& pending : pending_rows) {
            auto& record = row_shadow_records->at(pending.index);
            for (const auto& [exit_border, cand] : pending.pushed_exits) {
                const auto it = boundary_distances.find(exit_border);
                if (it != boundary_distances.end() && it->second == cand && settled_vertices.contains(exit_border)) {
                    ++record.num_useful_pushes_from_this_row;
                    ++record.num_exits_useful;
                }
            }
        }
    }
    std::sort(result.object_ids.begin(), result.object_ids.end());
    const auto t_after_finalize = Clock::now();
    result.init_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_init - t_started).count();
    result.skeleton_trace_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t_after_skeleton_trace - t_after_init).count();
    result.subgraph_eval_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t_after_finalize - t_after_skeleton_trace).count();
    return result;
}

std::vector<ObjId> SkeletonIndex::exact_range_query(
    const QueryPoint& query,
    EdgeWeight radius,
    const IndexedMovingObjectSet& objects
) const {
    const auto seeds = std::vector<std::pair<VertexId, EdgeWeight>>{
        {query.edge.first, query.offset},
        {query.edge.second, static_cast<EdgeWeight>(edge_weight_any(*global_, query.edge) - query.offset)},
    };
    const auto dist = dijkstra(*global_, seeds, radius);
    std::vector<ObjId> result;
    for (const auto& object : objects.objects()) {
        EdgeWeight best = kInfWeight;
        if (const auto it = dist.find(object.edge.first); it != dist.end() &&
            it->second <= kInfWeight - object.offset) {
            best = std::min(best, static_cast<EdgeWeight>(it->second + object.offset));
        }
        const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
        if (const auto it = dist.find(object.edge.second); it != dist.end() &&
            it->second <= kInfWeight - right_cost) {
            best = std::min(best, static_cast<EdgeWeight>(it->second + right_cost));
        }
        if (best <= radius) {
            result.push_back(object.unique_id);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

KnnQueryResult SkeletonIndex::knn_query(
    const QueryPoint& query,
    std::size_t k,
    const IndexedMovingObjectSet& objects,
    std::size_t query_id,
    std::vector<BorderExposureEvent>* exposure_events,
    std::vector<CliqueRowShadowRecord>* row_shadow_records,
    bool streamed_clique,
    bool safe_coverage_shadow,
    std::size_t parent_shadow_size,
    bool use_factorized_transfer
) const {
    if (k == 0 || objects.size() == 0) {
        return KnnQueryResult{};
    }

    using Clock = std::chrono::steady_clock;
    const auto t_started = Clock::now();

    const auto maybe_query_vertex = query_vertex_if_exact_vertex(query, *global_);
    const Subgraph* initial_ptr = nullptr;
    std::optional<VertexId> query_vertex;
    if (maybe_query_vertex.has_value()) {
        if (*maybe_query_vertex < vertex_to_subgraph_fast_.size() &&
            vertex_to_subgraph_fast_[*maybe_query_vertex] != std::numeric_limits<SgId>::max()) {
            initial_ptr = &subgraphs_.at(vertex_to_subgraph_fast_[*maybe_query_vertex]);
            query_vertex = *maybe_query_vertex;
        }
    }
    if (initial_ptr == nullptr) {
        initial_ptr = &require_initial_subgraph(query.edge);
    }
    const auto& initial = *initial_ptr;

    std::optional<DistanceMap> initial_local;
    std::vector<std::pair<VertexId, EdgeWeight>> initial_boundary_seeds;
    if (query_vertex.has_value()) {
        initial_boundary_seeds.reserve(subgraph_boundaries_[initial.id].size());
        for (const auto b : subgraph_boundaries_[initial.id]) {
            const auto d = initial.distance.get_or_inf(*query_vertex, b);
            if (d != kInfWeight) {
                initial_boundary_seeds.push_back({b, d});
            }
        }
    } else {
        initial_local = local_query_distances(initial, query, kInfWeight);
        initial_boundary_seeds = boundary_query_seeds(initial, *initial_local);
    }
    using CandidateItem = std::tuple<EdgeWeight, ObjId, SgId>;
    using ExactItem = std::tuple<EdgeWeight, ObjId>;
    using CoverageItem = std::tuple<EdgeWeight, ObjId, SgId>;

    std::priority_queue<KnnFrontierItem, std::vector<KnnFrontierItem>, KnnFrontierGreater> frontier_heap;
    std::priority_queue<CandidateItem> threshold_heap;
    std::priority_queue<ExactItem> exact_heap;
    std::priority_queue<CoverageItem> safe_coverage_heap;
    std::vector<CandidateItem> final_candidates;
    if (skeleton_scratch_dist_.size() != static_cast<std::size_t>(max_vertex_id_) + 1U) {
        skeleton_scratch_dist_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_stamp_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    if (skeleton_scratch_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(skeleton_scratch_stamp_.begin(), skeleton_scratch_stamp_.end(), 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    ++skeleton_scratch_epoch_;
    const auto skeleton_epoch = skeleton_scratch_epoch_;
    std::vector<VertexId> touched_boundaries;
    touched_boundaries.reserve(initial_boundary_seeds.size() * 8U + 64U);
    std::vector<EdgeWeight> best_subgraph_lb(subgraphs_.size(), kInfWeight);
    std::vector<std::uint8_t> visited_subgraphs(subgraphs_.size(), 0U);
    std::vector<SgId> visited_subgraph_list;
    visited_subgraph_list.reserve(initial_boundary_seeds.size() * 4U + 16U);
    std::vector<std::size_t> sg_touch_events(subgraphs_.size(), 0U);
    std::unordered_map<std::pair<SgId, VertexId>, std::size_t, PairHash> exposure_index;
    std::unordered_map<SgId, std::size_t> trigger_event_index;
    std::vector<std::uint8_t> pruned_subgraphs(subgraphs_.size(), 0U);
    std::vector<std::uint8_t> tightened_subgraphs(subgraphs_.size(), 0U);
    std::size_t num_clique_relax_attempts = 0;
    std::size_t num_successful_clique_relaxes = 0;
    std::size_t num_pq_pushes_from_clique = 0;
    std::size_t num_redundant_subgraph_reentries = 0;
    std::size_t num_useful_subgraph_reentries = 0;
    std::vector<VertexId> settled_boundaries;
    struct PendingRowShadow {
        std::size_t index{0};
        std::vector<std::pair<VertexId, EdgeWeight>> pushed_exits;
    };
    struct StreamedCliqueRow {
        std::vector<std::pair<VertexId, EdgeWeight>> emissions;
        std::size_t next_index{0};
    };
    std::vector<PendingRowShadow> pending_rows;
    std::vector<StreamedCliqueRow> streamed_rows;
    streamed_rows.reserve(initial_boundary_seeds.size() + 16U);

    long long pq_us = 0;
    long long membership_us = 0;
    long long subgraph_bookkeeping_us = 0;
    long long clique_emit_us = 0;
    std::size_t boundary_pq_pushes = 0;
    std::size_t boundary_pq_pops = 0;
    std::size_t row_state_pushes = 0;
    std::size_t row_state_pops = 0;
    std::size_t streamed_rows_started = 0;
    std::size_t streamed_row_emissions = 0;
    std::size_t streamed_rows_stopped_by_threshold = 0;
    std::size_t streamed_exits_skipped_by_threshold = 0;
    std::size_t peak_frontier_queue_size = 0;
    std::size_t safe_coverage_candidate_count = 0;
    std::size_t first_k_candidates_boundary_visit_order = 0;
    EdgeWeight first_finite_safe_coverage_radius = kInfWeight;
    std::size_t first_finite_safe_coverage_boundary_visit_order = 0;
    EdgeWeight safe_coverage_radius = kInfWeight;
    std::size_t safe_coverage_updates = 0;
    EdgeWeight first_finite_tau = kInfWeight;
    std::size_t first_finite_tau_boundary_visit_order = 0;
    EdgeWeight last_observed_tau = kInfWeight;
    std::size_t tau_updates = 0;
    std::size_t would_stop_rows_by_safe_coverage = 0;
    std::size_t would_skip_exits_by_safe_coverage = 0;
    std::size_t would_save_pq_pushes_by_safe_coverage = 0;
    std::size_t factorized_rows_used = 0;
    std::size_t factorized_hubs_used = 0;
    std::size_t factorized_exits_emitted = 0;

    const auto boundary_distance = [&](VertexId v) -> EdgeWeight {
        return (v < skeleton_scratch_stamp_.size() && skeleton_scratch_stamp_[v] == skeleton_epoch)
            ? skeleton_scratch_dist_[v]
            : kInfWeight;
    };

    visited_subgraphs[initial.id] = 1U;
    visited_subgraph_list.push_back(initial.id);
    std::size_t visited_subgraph_count = 1;
    std::size_t candidates_considered = 0;
    std::size_t boundary_visit_order = 0;

    auto ensure_exposure = [&](SgId sg_id, VertexId border, EdgeWeight distance) -> std::size_t {
        if (exposure_events == nullptr) {
            return static_cast<std::size_t>(-1);
        }
        const auto key = std::make_pair(sg_id, border);
        if (const auto it = exposure_index.find(key); it != exposure_index.end()) {
            auto& event = exposure_events->at(it->second);
            event.distance_from_query = std::min(event.distance_from_query, distance);
            return it->second;
        }
        const auto index = exposure_events->size();
        exposure_index[key] = index;
        exposure_events->push_back(BorderExposureEvent{
            query_id,
            QueryType::Knn,
            sg_id,
            border,
            boundary_visit_order,
            false,
            false,
            false,
            false,
            false,
            distance,
            0,
        });
        return index;
    };

    const auto observe_tau = [&](EdgeWeight tau, std::size_t order) {
        if (tau == last_observed_tau) {
            return;
        }
        ++tau_updates;
        last_observed_tau = tau;
        if (tau != kInfWeight && first_finite_tau == kInfWeight) {
            first_finite_tau = tau;
            first_finite_tau_boundary_visit_order = order;
        }
    };

    const auto observe_safe_coverage_candidate = [&](EdgeWeight upper_bound, ObjId obj_id, SgId sg_id, std::size_t order) {
        if (!safe_coverage_shadow || upper_bound == kInfWeight) {
            return;
        }
        ++safe_coverage_candidate_count;
        push_bounded_candidate(safe_coverage_heap, k, CoverageItem{upper_bound, obj_id, sg_id});
        if (safe_coverage_heap.size() < k) {
            return;
        }
        if (first_k_candidates_boundary_visit_order == 0) {
            first_k_candidates_boundary_visit_order = order;
        }
        const auto new_radius = std::get<0>(safe_coverage_heap.top());
        if (new_radius == safe_coverage_radius) {
            return;
        }
        safe_coverage_radius = new_radius;
        ++safe_coverage_updates;
        if (first_finite_safe_coverage_radius == kInfWeight) {
            first_finite_safe_coverage_radius = new_radius;
            first_finite_safe_coverage_boundary_visit_order = order;
        }
    };

    for (const auto obj_id : objects.objects_in(initial.id)) {
        const auto& object = objects[obj_id];
        const auto exact = query_vertex.has_value()
            ? object_distance_from_vertex(initial, *query_vertex, object)
            : object_distance_from_seed(initial, *initial_local, object);
        if (exact != kInfWeight) {
            ++candidates_considered;
            push_bounded_candidate(threshold_heap, k, CandidateItem{exact, object.unique_id, initial.id});
            observe_safe_coverage_candidate(exact, object.unique_id, initial.id, 0);
        }
    }
    observe_tau((threshold_heap.size() >= k) ? std::get<0>(threshold_heap.top()) : kInfWeight, 0);

    const auto frontier_push = [&](const KnnFrontierItem& item) {
        const auto t0 = Clock::now();
        frontier_heap.push(item);
        pq_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
        if (item.kind == KnnFrontierItem::Kind::Boundary) {
            ++boundary_pq_pushes;
        } else {
            ++row_state_pushes;
        }
        peak_frontier_queue_size = std::max(peak_frontier_queue_size, frontier_heap.size());
    };

    const auto frontier_pop = [&]() {
        const auto t0 = Clock::now();
        const auto item = frontier_heap.top();
        frontier_heap.pop();
        pq_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
        if (item.kind == KnnFrontierItem::Kind::Boundary) {
            ++boundary_pq_pops;
        } else {
            ++row_state_pops;
        }
        return item;
    };

    const auto try_schedule_next_row_emission =
        [&](std::size_t row_id, bool count_start) -> bool {
            auto& row = streamed_rows[row_id];
            if (row.next_index >= row.emissions.size()) {
                return false;
            }
            const auto current_threshold =
                (threshold_heap.size() >= k) ? std::get<0>(threshold_heap.top()) : kInfWeight;
            const auto& [next_border, next_distance] = row.emissions[row.next_index];
            if (current_threshold != kInfWeight && next_distance > current_threshold) {
                ++streamed_rows_stopped_by_threshold;
                streamed_exits_skipped_by_threshold += row.emissions.size() - row.next_index;
                row.next_index = row.emissions.size();
                return false;
            }
            if (count_start) {
                ++streamed_rows_started;
            }
            ++row.next_index;
            frontier_push(KnnFrontierItem{
                next_distance,
                KnnFrontierItem::Kind::RowEmit,
                next_border,
                row_id,
            });
            return true;
        };

    const auto update_boundary_state = [&](VertexId b, EdgeWeight distance, bool from_clique) {
        const auto t0_membership = Clock::now();
        if (skeleton_scratch_stamp_[b] == skeleton_epoch) {
            auto& best_distance = skeleton_scratch_dist_[b];
            if (distance >= best_distance) {
                membership_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_membership).count();
                return false;
            }
            best_distance = distance;
        } else {
            skeleton_scratch_stamp_[b] = skeleton_epoch;
            skeleton_scratch_dist_[b] = distance;
            touched_boundaries.push_back(b);
        }
        membership_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_membership).count();

        frontier_push(KnnFrontierItem{distance, KnnFrontierItem::Kind::Boundary, b, 0});
        if (from_clique) {
            ++num_successful_clique_relaxes;
            ++num_pq_pushes_from_clique;
        }

        const auto t0_bookkeeping = Clock::now();
        if (b < inverted_index_fast_.size()) {
            for (const auto sg_id : inverted_index_fast_[b]) {
                auto& sg_best = best_subgraph_lb[sg_id];
                if (distance < sg_best) {
                    sg_best = distance;
                }
            }
        }
        subgraph_bookkeeping_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_bookkeeping).count();
        return true;
    };

    for (const auto& [b, distance] : initial_boundary_seeds) {
        update_boundary_state(b, distance, false);
    }
    const auto t_after_init = Clock::now();

    while (!frontier_heap.empty()) {
        const auto item = frontier_pop();
        const auto current_threshold = (threshold_heap.size() >= k) ? std::get<0>(threshold_heap.top()) : kInfWeight;
        if (current_threshold != kInfWeight && item.distance > current_threshold) {
            break;
        }

        if (item.kind == KnnFrontierItem::Kind::RowEmit) {
            ++streamed_row_emissions;
            const auto t0_emit = Clock::now();
            auto& row = streamed_rows[item.row_id];
            update_boundary_state(item.vertex, item.distance, true);
            try_schedule_next_row_emission(item.row_id, false);
            clique_emit_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_emit).count();
            continue;
        }

        const auto dist_vb = item.distance;
        const auto v_b = item.vertex;
        ++boundary_visit_order;

        const auto t0_membership = Clock::now();
        const bool stale = v_b >= skeleton_scratch_stamp_.size() || boundary_distance(v_b) != dist_vb;
        const bool missing_inverted = v_b >= inverted_index_fast_.size() || inverted_index_fast_[v_b].empty();
        membership_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_membership).count();
        if (stale) {
            continue;
        }

        settled_boundaries.push_back(v_b);
        if (missing_inverted) {
            continue;
        }

        for (const auto sg_id : inverted_index_fast_[v_b]) {
            EdgeWeight sg_lb = kInfWeight;
            std::size_t event_idx = static_cast<std::size_t>(-1);
            bool first_visit = false;
            {
                const auto t0 = Clock::now();
                ++sg_touch_events[sg_id];
                if (sg_touch_events[sg_id] > 1U) {
                    ++num_redundant_subgraph_reentries;
                }
                event_idx = ensure_exposure(sg_id, v_b, dist_vb);
                first_visit = visited_subgraphs[sg_id] == 0U;
                if (first_visit) {
                    visited_subgraphs[sg_id] = 1U;
                    visited_subgraph_list.push_back(sg_id);
                    ++visited_subgraph_count;
                    if (event_idx != static_cast<std::size_t>(-1)) {
                        trigger_event_index.try_emplace(sg_id, event_idx);
                    }
                }
                sg_lb = best_subgraph_lb[sg_id];
                membership_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
            }
            if (!first_visit) {
                continue;
            }
            const auto& sg = subgraphs_.at(sg_id);
            if (sg_lb == kInfWeight) {
                continue;
            }

            auto threshold = (threshold_heap.size() >= k) ? std::get<0>(threshold_heap.top()) : kInfWeight;
            const auto threshold_before = threshold;
            bool pruned = false;
            const bool fast_explicit_emit =
                !streamed_clique &&
                row_shadow_records == nullptr &&
                (!use_factorized_transfer ||
                 sg_id >= factorized_transfer_enabled_.size() ||
                 factorized_transfer_enabled_[sg_id] == 0U);
            MaterializedCliqueRow materialized_row;
            std::vector<CliqueExitCandidate> exits;
            if (!fast_explicit_emit) {
                const auto t0 = Clock::now();
                materialized_row = materialize_subgraph_exit_candidates(
                    v_b,
                    dist_vb,
                    sg_id,
                    subgraph_clique_rows_,
                    factorized_transfer_models_,
                    factorized_transfer_enabled_,
                    use_factorized_transfer
                );
                if (materialized_row.used_factorized) {
                    ++factorized_rows_used;
                    factorized_hubs_used += materialized_row.hubs_used;
                    factorized_exits_emitted += materialized_row.exits.size();
                }
                exits = std::move(materialized_row.exits);
                clique_emit_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
            }
            if (safe_coverage_shadow) {
                const auto t0 = Clock::now();
                for (const auto obj_id : objects.objects_in(sg_id)) {
                    const auto& object = objects[obj_id];
                    const auto upper_bound =
                        object_distance_from_boundary(sg, v_b, dist_vb, object);
                    observe_safe_coverage_candidate(upper_bound, object.unique_id, sg_id, boundary_visit_order);
                }
                subgraph_bookkeeping_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
            }
            if (row_shadow_records != nullptr) {
                CliqueRowShadowRecord record;
                record.query_id = query_id;
                record.query_type = QueryType::Knn;
                record.subgraph_id = sg_id;
                record.entry_border = v_b;
                record.entry_visit_order = boundary_visit_order;
                record.entry_distance = dist_vb;
                record.threshold_at_entry = threshold;
                record.num_exits_total = exits.size();
                record.useful_prefix_len = exits.size();

                PendingRowShadow pending;
                pending.index = row_shadow_records->size();

                for (std::size_t i = 0; i < exits.size(); ++i) {
                    const auto& exit = exits[i];
                    const auto exit_border = exit.border;
                    const auto cand = exit.candidate_distance;
                    const auto current_best = boundary_distance(exit_border);
                    const bool dist_dominated = cand >= current_best;
                    const bool threshold_dead = threshold != kInfWeight && cand >= threshold;
                    if (dist_dominated) {
                        ++record.num_exits_dist_dominated;
                    }
                    if (threshold_dead) {
                        ++record.num_exits_threshold_dead;
                        record.useful_prefix_len = std::min(record.useful_prefix_len, i);
                    }
                    if (!dist_dominated) {
                        ++record.num_pq_pushes_from_this_row;
                        if (i >= record.useful_prefix_len) {
                            ++record.num_pq_pushes_beyond_useful_prefix;
                        }
                        pending.pushed_exits.push_back({exit_border, cand});
                    }
                }

                row_shadow_records->push_back(record);
                pending_rows.push_back(std::move(pending));
            }
            const auto t0 = Clock::now();
            for (const auto obj_id : objects.objects_in_knn_order(sg_id)) {
                const auto& object = objects[obj_id];
                const auto suffix = objects.knn_suffix(obj_id);
                if (suffix == kInfWeight) {
                    continue;
                }
                ++candidates_considered;
                if (threshold != kInfWeight) {
                    if (sg_lb > threshold) {
                        pruned = true;
                        break;
                    }
                    const auto budget = static_cast<EdgeWeight>(threshold - sg_lb);
                    if (suffix > budget) {
                        pruned = true;
                        break;
                    }
                }
                const auto lower_bound = static_cast<EdgeWeight>(sg_lb + suffix);
                if (lower_bound != kInfWeight) {
                    push_bounded_candidate(threshold_heap, k, CandidateItem{lower_bound, object.unique_id, sg_id});
                    if (threshold_heap.size() >= k) {
                        threshold = std::get<0>(threshold_heap.top());
                    }
                }
            }
            subgraph_bookkeeping_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
            if (pruned) {
                pruned_subgraphs[sg_id] = 1U;
            }
            const auto threshold_after = (threshold_heap.size() >= k) ? std::get<0>(threshold_heap.top()) : kInfWeight;
            observe_tau(threshold_after, boundary_visit_order);
            if (threshold_after < threshold_before) {
                tightened_subgraphs[sg_id] = 1U;
            }

            if (safe_coverage_shadow && safe_coverage_radius != kInfWeight) {
                bool row_stopped = false;
                std::size_t row_skipped = 0;
                std::size_t row_saved_pushes = 0;
                for (std::size_t i = 0; i < exits.size(); ++i) {
                    const auto& exit = exits[i];
                    if (exit.candidate_distance <= safe_coverage_radius) {
                        continue;
                    }
                    row_stopped = true;
                    row_skipped = exits.size() - i;
                    for (std::size_t j = i; j < exits.size(); ++j) {
                        const auto current_best = boundary_distance(exits[j].border);
                        if (exits[j].candidate_distance < current_best) {
                            ++row_saved_pushes;
                        }
                    }
                    break;
                }
                if (row_stopped) {
                    ++would_stop_rows_by_safe_coverage;
                    would_skip_exits_by_safe_coverage += row_skipped;
                    would_save_pq_pushes_by_safe_coverage += row_saved_pushes;
                }
            }

            const auto t0_emit = Clock::now();
            if (fast_explicit_emit) {
                for (const auto& [exit_border, local] : lookup_subgraph_clique_row(v_b, subgraph_clique_rows_[sg_id])) {
                    if (dist_vb == kInfWeight || local == kInfWeight || dist_vb > kInfWeight - local) {
                        continue;
                    }
                    ++num_clique_relax_attempts;
                    update_boundary_state(exit_border, static_cast<EdgeWeight>(dist_vb + local), true);
                }
            } else if (streamed_clique) {
                if (!exits.empty()) {
                    StreamedCliqueRow row;
                    row.emissions.reserve(exits.size());
                    for (const auto& exit : exits) {
                        ++num_clique_relax_attempts;
                        row.emissions.push_back({exit.border, exit.candidate_distance});
                    }
                    const auto row_id = streamed_rows.size();
                    streamed_rows.push_back(std::move(row));
                    try_schedule_next_row_emission(row_id, true);
                }
            } else {
                for (const auto& exit : exits) {
                    ++num_clique_relax_attempts;
                    update_boundary_state(exit.border, exit.candidate_distance, true);
                }
            }
            clique_emit_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_emit).count();
        }
    }
    const auto t_after_explore = Clock::now();
    const auto final_tau_after_explore = (threshold_heap.size() >= k) ? std::get<0>(threshold_heap.top()) : kInfWeight;

    final_candidates.reserve(threshold_heap.size());
    while (!threshold_heap.empty()) {
        final_candidates.push_back(threshold_heap.top());
        threshold_heap.pop();
    }

    std::sort(final_candidates.begin(), final_candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (std::get<0>(lhs) != std::get<0>(rhs)) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        }
        return std::get<1>(lhs) < std::get<1>(rhs);
    });

    EdgeWeight kth_exact = kInfWeight;
    std::size_t exact_evaluated = 0;
    for (const auto& [lower_bound, obj_id, sg_id] : final_candidates) {
        if (exact_heap.size() >= k && lower_bound > kth_exact) {
            break;
        }
        ++exact_evaluated;

        EdgeWeight exact = kInfWeight;
        if (sg_id == initial.id) {
            const auto& object = objects[obj_id];
            exact = query_vertex.has_value()
                ? object_distance_from_vertex(initial, *query_vertex, object)
                : object_distance_from_seed(initial, *initial_local, object);
        } else {
            for (const auto& [b, border_cost] : objects.knn_border_costs(obj_id)) {
                const auto border_distance = boundary_distance(b);
                if (border_distance == kInfWeight || border_distance > kInfWeight - border_cost) {
                    continue;
                }
                exact = std::min(exact, static_cast<EdgeWeight>(border_distance + border_cost));
            }
        }
        if (exact == kInfWeight) {
            continue;
        }

        push_bounded_candidate(exact_heap, k, ExactItem{exact, obj_id});
        if (exact_heap.size() == k) {
            kth_exact = std::get<0>(exact_heap.top());
        }
    }

    std::vector<KnnItem> items;
    items.reserve(exact_heap.size());
    while (!exact_heap.empty()) {
        const auto [distance, obj_id] = exact_heap.top();
        exact_heap.pop();
        items.push_back({obj_id, distance});
    }
    std::sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.distance != rhs.distance) {
            return lhs.distance < rhs.distance;
        }
        return lhs.id < rhs.id;
    });
    if (exposure_events != nullptr) {
        std::unordered_map<SgId, std::size_t> overlap_by_subgraph;
        for (const auto& item : items) {
            ++overlap_by_subgraph[objects.object_subgraph(item.id)];
        }
        for (const auto& [key, idx] : exposure_index) {
            const auto sg_id = key.first;
            auto& event = exposure_events->at(idx);
            const auto overlap = overlap_by_subgraph.contains(sg_id) ? overlap_by_subgraph.at(sg_id) : 0U;
            event.eventual_result_overlap = overlap;
            event.was_needed_for_final_answer = overlap != 0U;
            event.was_only_used_for_traversal = overlap == 0U && tightened_subgraphs[sg_id] == 0U;
        }
        for (const auto& [sg_id, idx] : trigger_event_index) {
            auto& event = exposure_events->at(idx);
            event.tightened_upper_bound = tightened_subgraphs[sg_id] != 0U;
            event.pruned_subgraph = pruned_subgraphs[sg_id] != 0U;
        }
    }
    for (const auto sg_id : visited_subgraph_list) {
        const auto touches = sg_touch_events[sg_id];
        if (touches <= 1U) {
            continue;
        }
        if (tightened_subgraphs[sg_id] != 0U || pruned_subgraphs[sg_id] != 0U) {
            num_useful_subgraph_reentries += touches - 1U;
        }
    }
    if (row_shadow_records != nullptr) {
        std::unordered_set<VertexId> settled_vertices;
        settled_vertices.reserve(settled_boundaries.size() * 2U + 1U);
        for (const auto vertex : settled_boundaries) {
            settled_vertices.insert(vertex);
        }
        for (const auto& pending : pending_rows) {
            auto& record = row_shadow_records->at(pending.index);
            for (const auto& [exit_border, cand] : pending.pushed_exits) {
                if (boundary_distance(exit_border) == cand && settled_vertices.contains(exit_border)) {
                    ++record.num_useful_pushes_from_this_row;
                    ++record.num_exits_useful;
                }
            }
        }
    }
    const auto t_after_finalize = Clock::now();
    KnnQueryResult result;
    result.items = std::move(items);
    result.visited_boundaries = touched_boundaries.size();
    result.visited_subgraphs = visited_subgraph_count;
    result.candidates_considered = candidates_considered;
    result.final_candidates = final_candidates.size();
    result.exact_evaluated = exact_evaluated;
    result.vertex_fast_path = query_vertex.has_value();
    result.init_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_init - t_started).count();
    result.explore_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_explore - t_after_init).count();
    result.finalize_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_finalize - t_after_explore).count();
    result.pq_us = pq_us;
    result.membership_us = membership_us;
    result.subgraph_bookkeeping_us = subgraph_bookkeeping_us;
    result.clique_emit_us = clique_emit_us;
    result.num_clique_relax_attempts = num_clique_relax_attempts;
    result.num_successful_clique_relaxes = num_successful_clique_relaxes;
    result.num_pq_pushes_from_clique = num_pq_pushes_from_clique;
    result.num_redundant_subgraph_reentries = num_redundant_subgraph_reentries;
    result.num_useful_subgraph_reentries = num_useful_subgraph_reentries;
    result.boundary_pq_pushes = boundary_pq_pushes;
    result.boundary_pq_pops = boundary_pq_pops;
    result.row_state_pushes = row_state_pushes;
    result.row_state_pops = row_state_pops;
    result.streamed_rows_started = streamed_rows_started;
    result.streamed_row_emissions = streamed_row_emissions;
    result.streamed_rows_stopped_by_threshold = streamed_rows_stopped_by_threshold;
    result.streamed_exits_skipped_by_threshold = streamed_exits_skipped_by_threshold;
    result.peak_frontier_queue_size = peak_frontier_queue_size;
    result.safe_coverage_candidate_count = safe_coverage_candidate_count;
    result.first_k_candidates_boundary_visit_order = first_k_candidates_boundary_visit_order;
    result.first_finite_safe_coverage_radius = first_finite_safe_coverage_radius;
    result.first_finite_safe_coverage_boundary_visit_order = first_finite_safe_coverage_boundary_visit_order;
    result.final_safe_coverage_radius = safe_coverage_radius;
    result.safe_coverage_updates = safe_coverage_updates;
    result.first_finite_tau = first_finite_tau;
    result.first_finite_tau_boundary_visit_order = first_finite_tau_boundary_visit_order;
    result.final_tau = final_tau_after_explore;
    result.tau_updates = tau_updates;
    result.would_stop_rows_by_safe_coverage = would_stop_rows_by_safe_coverage;
    result.would_skip_exits_by_safe_coverage = would_skip_exits_by_safe_coverage;
    result.would_save_pq_pushes_by_safe_coverage = would_save_pq_pushes_by_safe_coverage;
    result.parent_region_size = parent_shadow_size;
    if (parent_shadow_size > 0 && !subgraph_adjacency_.empty()) {
        const auto parent_of = build_parent_assignment(subgraph_adjacency_, parent_shadow_size);
        std::size_t parent_count = 0;
        for (const auto parent_id : parent_of) {
            if (parent_id != std::numeric_limits<std::size_t>::max()) {
                parent_count = std::max(parent_count, parent_id + 1U);
            }
        }
        if (parent_count != 0) {
            std::vector<std::size_t> parent_group_sizes(parent_count, 0U);
            std::vector<std::size_t> children_descended(parent_count, 0U);
            for (std::size_t sg_id = 0; sg_id < parent_of.size(); ++sg_id) {
                if (parent_of[sg_id] != std::numeric_limits<std::size_t>::max()) {
                    ++parent_group_sizes[parent_of[sg_id]];
                }
            }
            for (const auto sg_id : visited_subgraph_list) {
                const auto parent_id = parent_of[sg_id];
                if (parent_id != std::numeric_limits<std::size_t>::max()) {
                    ++children_descended[parent_id];
                }
            }
            std::size_t touched_parents = 0;
            std::size_t total_children_descended = 0;
            std::size_t total_children_capacity = 0;
            std::size_t max_children_descended = 0;
            std::size_t single_child_parents = 0;
            for (std::size_t parent_id = 0; parent_id < parent_count; ++parent_id) {
                const auto descended = children_descended[parent_id];
                if (descended == 0) {
                    continue;
                }
                ++touched_parents;
                total_children_descended += descended;
                total_children_capacity += parent_group_sizes[parent_id];
                max_children_descended = std::max(max_children_descended, descended);
                if (descended == 1) {
                    ++single_child_parents;
                }
            }
            result.parent_regions_touched = touched_parents;
            result.max_children_descended_in_parent = max_children_descended;
            result.single_child_parent_regions = single_child_parents;
            result.avg_children_descended_per_parent =
                touched_parents == 0 ? 0.0 : static_cast<double>(total_children_descended) / static_cast<double>(touched_parents);
            result.parent_descent_ratio =
                total_children_capacity == 0 ? 0.0 : static_cast<double>(total_children_descended) / static_cast<double>(total_children_capacity);
        }
    }
    result.factorized_rows_used = factorized_rows_used;
    result.factorized_hubs_used = factorized_hubs_used;
    result.factorized_exits_emitted = factorized_exits_emitted;
    result.streamed_clique = streamed_clique;
    return result;
}

KnnQueryResult SkeletonIndex::knn_query_local_dijkstra(
    const QueryPoint& query,
    std::size_t k,
    const IndexedMovingObjectSet& objects,
    FcRule rule,
    std::size_t query_id
) const {
    (void)query_id;
    if (k == 0 || objects.size() == 0) {
        return KnnQueryResult{};
    }

    using Clock = std::chrono::steady_clock;
    const auto t_started = Clock::now();

    if (skeleton_scratch_dist_.size() != static_cast<std::size_t>(max_vertex_id_) + 1U) {
        skeleton_scratch_dist_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_stamp_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    if (skeleton_scratch_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(skeleton_scratch_stamp_.begin(), skeleton_scratch_stamp_.end(), 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    ++skeleton_scratch_epoch_;
    const auto skeleton_epoch = skeleton_scratch_epoch_;

    using OrderedObject = std::pair<EdgeWeight, ObjId>;
    std::set<OrderedObject> top_k;
    std::set<OrderedObject> overflow;
    std::vector<EdgeWeight> best_object_distance(objects.size(), kInfWeight);

    const auto rebalance_top_k = [&]() {
        while (top_k.size() < k && !overflow.empty()) {
            auto it = overflow.begin();
            top_k.insert(*it);
            overflow.erase(it);
        }
        while (top_k.size() > k) {
            auto it = std::prev(top_k.end());
            overflow.insert(*it);
            top_k.erase(it);
        }
        while (!top_k.empty() && !overflow.empty()) {
            auto worst_top = std::prev(top_k.end());
            auto best_overflow = overflow.begin();
            if (!(*best_overflow < *worst_top)) {
                break;
            }
            const auto move_to_overflow = *worst_top;
            const auto move_to_top = *best_overflow;
            top_k.erase(worst_top);
            overflow.erase(best_overflow);
            top_k.insert(move_to_top);
            overflow.insert(move_to_overflow);
        }
    };

    std::size_t candidates_considered = 0;
    std::size_t exact_evaluated = 0;
    const auto consider_object = [&](ObjId obj_id, EdgeWeight distance) {
        if (distance == kInfWeight) {
            return;
        }
        const auto index = static_cast<std::size_t>(obj_id);
        if (index >= best_object_distance.size() || distance >= best_object_distance[index]) {
            return;
        }
        const auto old = best_object_distance[index];
        if (old != kInfWeight) {
            const OrderedObject old_key{old, obj_id};
            if (top_k.erase(old_key) == 0U) {
                overflow.erase(old_key);
            }
        }
        best_object_distance[index] = distance;
        const OrderedObject key{distance, obj_id};
        if (top_k.size() < k) {
            top_k.insert(key);
        } else if (!top_k.empty() && key < *std::prev(top_k.end())) {
            top_k.insert(key);
        } else {
            overflow.insert(key);
        }
        rebalance_top_k();
    };
    const auto tau = [&]() -> EdgeWeight {
        return top_k.size() >= k ? std::prev(top_k.end())->first : kInfWeight;
    };

    const auto boundary_distance = [&](VertexId v) -> EdgeWeight {
        return (v < skeleton_scratch_stamp_.size() && skeleton_scratch_stamp_[v] == skeleton_epoch)
            ? skeleton_scratch_dist_[v]
            : kInfWeight;
    };

    const auto local_object_distance = [&](const LocalDijkstraResult& dist, const MovingObject& object) {
        const auto du = lookup_local_distance(dist, object.edge.first);
        const auto dv = lookup_local_distance(dist, object.edge.second);
        EdgeWeight best = kInfWeight;
        if (du != kInfWeight && du <= kInfWeight - object.offset) {
            best = std::min(best, static_cast<EdgeWeight>(du + object.offset));
        }
        const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
        if (dv != kInfWeight && dv <= kInfWeight - right_cost) {
            best = std::min(best, static_cast<EdgeWeight>(dv + right_cost));
        }
        return best;
    };

    const auto collect_from_local = [&](SgId sg_id, const LocalDijkstraResult& dist, EdgeWeight limit) {
        for (const auto& populated : objects.populated_edges_in(sg_id)) {
            for (const auto obj_id : populated.object_ids) {
                ++candidates_considered;
                ++exact_evaluated;
                const auto d = local_object_distance(dist, objects[obj_id]);
                if (d <= limit) {
                    consider_object(obj_id, d);
                }
            }
        }
    };

    const auto collect_from_boundaries = [&](const Subgraph& sg, EdgeWeight limit) {
        for (const auto obj_id : objects.objects_in(sg.id)) {
            ++candidates_considered;
            ++exact_evaluated;
            const auto& object = objects[obj_id];
            EdgeWeight best = kInfWeight;
            for (const auto b : subgraph_boundaries_[sg.id]) {
                const auto bd = boundary_distance(b);
                if (bd == kInfWeight || bd > limit) {
                    continue;
                }
                best = std::min(best, object_distance_from_boundary(sg, b, bd, object));
            }
            if (best <= limit) {
                consider_object(obj_id, best);
            }
        }
    };

    const auto& initial = require_initial_subgraph(query.edge);
    const auto& initial_local_index = subgraph_local_indices_.at(initial.id);
    const auto initial_local = local_query_distances_compact(initial, initial_local_index, query, kInfWeight);
    collect_from_local(initial.id, initial_local, kInfWeight);
    const auto initial_seeds = boundary_query_seeds(initial, initial_local);

    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    std::vector<VertexId> touched_boundaries;
    touched_boundaries.reserve(initial_seeds.size() * 8U + 64U);
    std::vector<std::uint8_t> visited_subgraphs(subgraphs_.size(), 0U);
    visited_subgraphs[initial.id] = 1U;
    std::size_t visited_subgraph_count = 1U;
    std::size_t num_clique_relax_attempts = 0;
    std::size_t num_successful_clique_relaxes = 0;
    std::size_t num_pq_pushes_from_clique = 0;
    std::size_t boundary_pq_pushes = 0;
    std::size_t boundary_pq_pops = 0;

    const auto update_boundary = [&](VertexId v, EdgeWeight distance, bool from_clique) {
        if (v >= skeleton_scratch_stamp_.size()) {
            return false;
        }
        if (skeleton_scratch_stamp_[v] == skeleton_epoch) {
            if (distance >= skeleton_scratch_dist_[v]) {
                return false;
            }
            skeleton_scratch_dist_[v] = distance;
        } else {
            skeleton_scratch_stamp_[v] = skeleton_epoch;
            skeleton_scratch_dist_[v] = distance;
            touched_boundaries.push_back(v);
        }
        pq.push({distance, v});
        ++boundary_pq_pushes;
        if (from_clique) {
            ++num_successful_clique_relaxes;
            ++num_pq_pushes_from_clique;
        }
        return true;
    };

    for (const auto& [b, distance] : initial_seeds) {
        update_boundary(b, distance, false);
    }

    const auto t_after_init = Clock::now();

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        const auto current_tau = tau();
        if (current_tau != kInfWeight && current > current_tau) {
            break;
        }
        pq.pop();
        ++boundary_pq_pops;
        if (current != boundary_distance(u)) {
            continue;
        }

        if (u < inverted_index_fast_.size()) {
            for (const auto sg_id : inverted_index_fast_[u]) {
                const auto& sg = subgraphs_.at(sg_id);
                if (visited_subgraphs[sg_id] == 0U) {
                    visited_subgraphs[sg_id] = 1U;
                    ++visited_subgraph_count;
                }

                std::vector<std::pair<VertexId, EdgeWeight>> seeds;
                seeds.reserve(subgraph_boundaries_[sg_id].size());
                for (const auto b : subgraph_boundaries_[sg_id]) {
                    const auto bd = boundary_distance(b);
                    if (bd != kInfWeight && bd <= tau()) {
                        seeds.push_back({b, bd});
                    }
                }
                if (seeds.empty()) {
                    continue;
                }

                const auto limit = tau();
                const bool fully_covered =
                    limit != kInfWeight &&
                    is_fully_covered(sg, skeleton_scratch_dist_, skeleton_scratch_stamp_, skeleton_epoch, limit, rule);
                if (fully_covered) {
                    collect_from_boundaries(sg, limit);
                } else {
                    const auto local_dist = local_dijkstra_compact(subgraph_local_indices_[sg_id], seeds, limit);
                    collect_from_local(sg_id, local_dist, limit);
                }
            }
        }

        const auto row_limit = tau();
        const auto& row = lookup_sorted_skeleton_row(u, sorted_skeleton_row_index_, sorted_skeleton_row_storage_);
        for (const auto& [v, w] : row) {
            ++num_clique_relax_attempts;
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (row_limit != kInfWeight && next > row_limit) {
                break;
            }
            update_boundary(v, next, true);
        }
    }

    const auto t_after_explore = Clock::now();

    std::vector<KnnItem> items;
    items.reserve(top_k.size());
    for (const auto& [distance, obj_id] : top_k) {
        items.push_back({obj_id, distance});
    }
    if (items.size() > k) {
        items.resize(k);
    }

    const auto t_after_finalize = Clock::now();
    KnnQueryResult result;
    result.items = std::move(items);
    result.visited_boundaries = touched_boundaries.size();
    result.visited_subgraphs = visited_subgraph_count;
    result.candidates_considered = candidates_considered;
    result.final_candidates = top_k.size() + overflow.size();
    result.exact_evaluated = exact_evaluated;
    result.init_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_init - t_started).count();
    result.explore_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_explore - t_after_init).count();
    result.finalize_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_finalize - t_after_explore).count();
    result.num_clique_relax_attempts = num_clique_relax_attempts;
    result.num_successful_clique_relaxes = num_successful_clique_relaxes;
    result.num_pq_pushes_from_clique = num_pq_pushes_from_clique;
    result.boundary_pq_pushes = boundary_pq_pushes;
    result.boundary_pq_pops = boundary_pq_pops;
    result.final_tau = tau();
    result.first_finite_tau = result.final_tau;
    return result;
}

KnnQueryResult SkeletonIndex::knn_query_global_dijkstra(
    const QueryPoint& query,
    std::size_t k,
    const IndexedMovingObjectSet& objects,
    std::size_t query_id
) const {
    (void)query_id;
    if (k == 0 || objects.size() == 0) {
        return KnnQueryResult{};
    }
    if (global_ == nullptr) {
        throw std::runtime_error("global graph is not available");
    }

    using Clock = std::chrono::steady_clock;
    const auto t_started = Clock::now();

    if (skeleton_scratch_dist_.size() != static_cast<std::size_t>(max_vertex_id_) + 1U) {
        skeleton_scratch_dist_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_stamp_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    if (skeleton_scratch_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(skeleton_scratch_stamp_.begin(), skeleton_scratch_stamp_.end(), 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    ++skeleton_scratch_epoch_;
    const auto epoch = skeleton_scratch_epoch_;

    using OrderedObject = std::pair<EdgeWeight, ObjId>;
    std::set<OrderedObject> top_k;
    std::set<OrderedObject> overflow;
    std::vector<EdgeWeight> best_object_distance(objects.size(), kInfWeight);

    const auto rebalance_top_k = [&]() {
        while (top_k.size() < k && !overflow.empty()) {
            auto it = overflow.begin();
            top_k.insert(*it);
            overflow.erase(it);
        }
        while (top_k.size() > k) {
            auto it = std::prev(top_k.end());
            overflow.insert(*it);
            top_k.erase(it);
        }
        while (!top_k.empty() && !overflow.empty()) {
            auto worst_top = std::prev(top_k.end());
            auto best_overflow = overflow.begin();
            if (!(*best_overflow < *worst_top)) {
                break;
            }
            const auto move_to_overflow = *worst_top;
            const auto move_to_top = *best_overflow;
            top_k.erase(worst_top);
            overflow.erase(best_overflow);
            top_k.insert(move_to_top);
            overflow.insert(move_to_overflow);
        }
    };

    std::size_t candidates_considered = 0;
    std::size_t exact_evaluated = 0;
    const auto consider_object = [&](ObjId obj_id, EdgeWeight distance) {
        if (distance == kInfWeight) {
            return;
        }
        const auto index = static_cast<std::size_t>(obj_id);
        if (index >= best_object_distance.size() || distance >= best_object_distance[index]) {
            return;
        }
        const auto old = best_object_distance[index];
        if (old != kInfWeight) {
            const OrderedObject old_key{old, obj_id};
            if (top_k.erase(old_key) == 0U) {
                overflow.erase(old_key);
            }
        }
        best_object_distance[index] = distance;
        const OrderedObject key{distance, obj_id};
        if (top_k.size() < k) {
            top_k.insert(key);
        } else if (!top_k.empty() && key < *std::prev(top_k.end())) {
            top_k.insert(key);
        } else {
            overflow.insert(key);
        }
        rebalance_top_k();
    };
    const auto tau = [&]() -> EdgeWeight {
        return top_k.size() >= k ? std::prev(top_k.end())->first : kInfWeight;
    };
    const auto vertex_distance = [&](VertexId vertex) -> EdgeWeight {
        return (vertex < skeleton_scratch_stamp_.size() && skeleton_scratch_stamp_[vertex] == epoch)
            ? skeleton_scratch_dist_[vertex]
            : kInfWeight;
    };

    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    std::vector<VertexId> touched_vertices;
    touched_vertices.reserve(1024U);
    std::size_t boundary_pq_pushes = 0;
    std::size_t boundary_pq_pops = 0;

    const auto push_vertex = [&](VertexId vertex, EdgeWeight distance) {
        if (vertex >= skeleton_scratch_stamp_.size()) {
            return false;
        }
        if (skeleton_scratch_stamp_[vertex] == epoch) {
            if (distance >= skeleton_scratch_dist_[vertex]) {
                return false;
            }
            skeleton_scratch_dist_[vertex] = distance;
        } else {
            skeleton_scratch_stamp_[vertex] = epoch;
            skeleton_scratch_dist_[vertex] = distance;
            touched_vertices.push_back(vertex);
        }
        pq.push({distance, vertex});
        ++boundary_pq_pushes;
        return true;
    };

    const auto query_edge = ordered_edge(query.edge.first, query.edge.second);
    const auto query_edge_weight = edge_weight_any(*global_, query_edge);
    if (query.offset >= query_edge_weight) {
        throw std::runtime_error("query offset must be smaller than edge weight");
    }
    push_vertex(query_edge.first, query.offset);
    push_vertex(query_edge.second, static_cast<EdgeWeight>(query_edge_weight - query.offset));
    const auto t_after_init = Clock::now();

    while (!pq.empty()) {
        const auto current_tau = tau();
        const auto [current, vertex] = pq.top();
        if (current_tau != kInfWeight && current > current_tau) {
            break;
        }
        pq.pop();
        ++boundary_pq_pops;
        if (current != vertex_distance(vertex)) {
            continue;
        }

        for (const auto obj_id : objects.objects_incident_to(vertex)) {
            const auto& object = objects[obj_id];
            EdgeWeight best = kInfWeight;
            if (object.edge.first == vertex && current <= kInfWeight - object.offset) {
                best = std::min(best, static_cast<EdgeWeight>(current + object.offset));
            }
            const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
            if (object.edge.second == vertex && current <= kInfWeight - right_cost) {
                best = std::min(best, static_cast<EdgeWeight>(current + right_cost));
            }
            if (best != kInfWeight) {
                ++candidates_considered;
                ++exact_evaluated;
                consider_object(obj_id, best);
            }
        }

        const auto limit = tau();
        for (const auto& [next_vertex, weight] : global_->neighbors(vertex)) {
            if (current == kInfWeight || weight == kInfWeight || current > kInfWeight - weight) {
                continue;
            }
            const auto next_distance = static_cast<EdgeWeight>(current + weight);
            if (limit != kInfWeight && next_distance > limit) {
                continue;
            }
            push_vertex(next_vertex, next_distance);
        }
    }

    const auto t_after_explore = Clock::now();

    std::vector<KnnItem> items;
    items.reserve(top_k.size());
    for (const auto& [distance, obj_id] : top_k) {
        items.push_back({obj_id, distance});
    }
    if (items.size() > k) {
        items.resize(k);
    }

    const auto t_after_finalize = Clock::now();
    KnnQueryResult result;
    result.items = std::move(items);
    result.visited_boundaries = touched_vertices.size();
    result.visited_subgraphs = 0;
    result.candidates_considered = candidates_considered;
    result.final_candidates = top_k.size() + overflow.size();
    result.exact_evaluated = exact_evaluated;
    result.init_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_init - t_started).count();
    result.explore_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_explore - t_after_init).count();
    result.finalize_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_finalize - t_after_explore).count();
    result.boundary_pq_pushes = boundary_pq_pushes;
    result.boundary_pq_pops = boundary_pq_pops;
    result.final_tau = tau();
    result.first_finite_tau = result.final_tau;
    return result;
}

KnnQueryResult SkeletonIndex::exact_knn_query(
    const QueryPoint& query,
    std::size_t k,
    const IndexedMovingObjectSet& objects
) const {
    const auto seeds = std::vector<std::pair<VertexId, EdgeWeight>>{
        {query.edge.first, query.offset},
        {query.edge.second, static_cast<EdgeWeight>(edge_weight_any(*global_, query.edge) - query.offset)},
    };
    const auto dist = dijkstra(*global_, seeds, kInfWeight);
    return compute_knn(objects, k, [&](const MovingObject& object) -> EdgeWeight {
        EdgeWeight best = kInfWeight;
        if (const auto it = dist.find(object.edge.first); it != dist.end() &&
            it->second <= kInfWeight - object.offset) {
            best = std::min(best, static_cast<EdgeWeight>(it->second + object.offset));
        }
        const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
        if (const auto it = dist.find(object.edge.second); it != dist.end() &&
            it->second <= kInfWeight - right_cost) {
            best = std::min(best, static_cast<EdgeWeight>(it->second + right_cost));
        }
        return best;
    });
}

FrontierBoundaryStats SkeletonIndex::frontier_boundary_stats(
    const QueryPoint& query,
    EdgeWeight radius
) const {
    const auto& initial = require_initial_subgraph(query.edge);
    const auto initial_local = local_query_distances(initial, query, radius);
    const auto initial_seeds = boundary_query_seeds(initial, initial_local);

    if (skeleton_scratch_dist_.size() != static_cast<std::size_t>(max_vertex_id_) + 1U) {
        skeleton_scratch_dist_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_stamp_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    if (skeleton_scratch_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(skeleton_scratch_stamp_.begin(), skeleton_scratch_stamp_.end(), 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    ++skeleton_scratch_epoch_;
    const auto scratch_epoch = skeleton_scratch_epoch_;

    const auto boundary_distance = [&](VertexId v) -> EdgeWeight {
        return (v < skeleton_scratch_stamp_.size() && skeleton_scratch_stamp_[v] == scratch_epoch)
            ? skeleton_scratch_dist_[v]
            : kInfWeight;
    };

    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    std::vector<VertexId> touched_boundaries;
    touched_boundaries.reserve(initial_seeds.size() * 8U + 64U);

    const auto update_boundary = [&](VertexId v, EdgeWeight distance) {
        if (skeleton_scratch_stamp_[v] == scratch_epoch) {
            if (distance >= skeleton_scratch_dist_[v]) {
                return false;
            }
            skeleton_scratch_dist_[v] = distance;
        } else {
            skeleton_scratch_stamp_[v] = scratch_epoch;
            skeleton_scratch_dist_[v] = distance;
            touched_boundaries.push_back(v);
        }
        pq.push({distance, v});
        return true;
    };

    for (const auto& [seed, value] : initial_seeds) {
        update_boundary(seed, value);
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();
        if (current != boundary_distance(u)) {
            continue;
        }
        if (current > radius) {
            continue;
        }
        const auto& row = lookup_sorted_skeleton_row(u, sorted_skeleton_row_index_, sorted_skeleton_row_storage_);
        for (const auto& [v, w] : row) {
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next > radius) {
                continue;
            }
            update_boundary(v, next);
        }
    }

    std::size_t frontier_boundaries = 0;
    std::size_t outward_edges = 0;
    for (const auto b : touched_boundaries) {
        const auto current = boundary_distance(b);
        if (current == kInfWeight || current > radius) {
            continue;
        }
        bool is_frontier = false;
        const auto& row = lookup_sorted_skeleton_row(b, sorted_skeleton_row_index_, sorted_skeleton_row_storage_);
        for (const auto& [v, w] : row) {
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next <= radius) {
                continue;
            }
            if (boundary_distance(v) == kInfWeight) {
                is_frontier = true;
                ++outward_edges;
            }
        }
        if (is_frontier) {
            ++frontier_boundaries;
        }
    }

    FrontierBoundaryStats result;
    result.radius = radius;
    result.reached_boundaries = touched_boundaries.size();
    result.frontier_boundaries = frontier_boundaries;
    result.outward_edges = outward_edges;
    return result;
}

}  // namespace bag
