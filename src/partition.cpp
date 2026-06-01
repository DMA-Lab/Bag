#include "partition.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <string_view>

#include "partition_cache.h"

namespace bag {

namespace {

template <typename Fn>
long long calc_execution_time_in_us(Fn&& fn) {
    const auto started = std::chrono::steady_clock::now();
    fn();
    const auto finished = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(finished - started).count();
}

void clear_query_derived_state(Subgraph& sg) {
    sg.rb_map.clear();
    sg.internal_to_nearest_border_dist.clear();
}

bool compute_br_metrics_phase1_no_distal(Subgraph& sg);
void fill_internal_to_nearest_border(Subgraph& sg);

std::size_t count_new_members(
    const std::unordered_set<VertexId>& before,
    const std::unordered_set<VertexId>& after
) {
    std::size_t count = 0;
    for (const auto v : after) {
        if (!before.contains(v)) {
            ++count;
        }
    }
    return count;
}

std::string json_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 8U);
    for (const auto ch : value) {
        switch (ch) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped += ch;
            break;
        }
    }
    return escaped;
}

void discard_rebuildable_state(Subgraph& sg) {
    sg.distance = DistanceTable{};
    clear_query_derived_state(sg);
}

Subgraph clone_subgraph_for_phase1(const Subgraph& source) {
    Subgraph candidate;
    candidate.id = source.id;
    candidate.seed_vertex = source.seed_vertex;
    candidate.graph = source.graph;
    candidate.distance = source.distance;
    candidate.bound_vertices = source.bound_vertices;
    candidate.internal_vertices = source.internal_vertices;
    candidate.insertion_order = source.insertion_order;
    return candidate;
}

Subgraph clone_subgraph_for_phase2(const Subgraph& source) {
    Subgraph candidate;
    candidate.id = source.id;
    candidate.seed_vertex = source.seed_vertex;
    candidate.graph = source.graph;
    candidate.bound_vertices = source.bound_vertices;
    candidate.internal_vertices = source.internal_vertices;
    candidate.insertion_order = source.insertion_order;
    return candidate;
}

struct VertexMembershipSnapshot {
    VertexId v{kInvalidVertex};
    bool valid{false};
    bool was_boundary{false};
    bool was_internal{false};
};

VertexMembershipSnapshot capture_membership_snapshot(const Subgraph& sg, VertexId v) {
    if (v == kInvalidVertex || !sg.contains(v)) {
        return {};
    }
    return VertexMembershipSnapshot{
        v,
        true,
        sg.bound_vertices.contains(v),
        sg.internal_vertices.contains(v),
    };
}

void restore_membership_snapshot(Subgraph& sg, const VertexMembershipSnapshot& snapshot) {
    if (!snapshot.valid || !sg.contains(snapshot.v)) {
        return;
    }
    if (snapshot.was_boundary) {
        sg.bound_vertices.insert(snapshot.v);
    } else {
        sg.bound_vertices.erase(snapshot.v);
    }
    if (snapshot.was_internal) {
        sg.internal_vertices.insert(snapshot.v);
    } else {
        sg.internal_vertices.erase(snapshot.v);
    }
}

struct Phase1LeafRollback {
    VertexId attach{kInvalidVertex};
    VertexId new_vertex{kInvalidVertex};
    std::size_t insertion_order_size{0};
    std::array<VertexMembershipSnapshot, 4> snapshots{};
};

Phase1LeafRollback make_phase1_leaf_rollback(
    const Subgraph& sg,
    VertexId seed,
    VertexId attach,
    VertexId new_vertex,
    bool paper_strict_mode
) {
    const auto previous_newest =
        sg.insertion_order.empty() ? kInvalidVertex : sg.insertion_order.back();
    Phase1LeafRollback rollback;
    rollback.attach = attach;
    rollback.new_vertex = new_vertex;
    rollback.insertion_order_size = sg.insertion_order.size();
    rollback.snapshots = {
        capture_membership_snapshot(sg, seed),
        capture_membership_snapshot(sg, attach),
        capture_membership_snapshot(sg, previous_newest),
        capture_membership_snapshot(sg, paper_strict_mode ? new_vertex : previous_newest),
    };
    return rollback;
}

void rollback_phase1_leaf_inplace(
    Subgraph& sg,
    const Phase1LeafRollback& rollback,
    bool eager_nearest_border_fill
) {
    sg.graph.disconnect(rollback.attach, rollback.new_vertex);
    sg.graph.erase_isolated_vertex(rollback.new_vertex);
    sg.distance.erase_vertex(rollback.new_vertex);
    while (sg.insertion_order.size() > rollback.insertion_order_size) {
        sg.insertion_order.pop_back();
    }
    sg.bound_vertices.erase(rollback.new_vertex);
    sg.internal_vertices.erase(rollback.new_vertex);
    for (const auto& snapshot : rollback.snapshots) {
        restore_membership_snapshot(sg, snapshot);
    }
    clear_query_derived_state(sg);
    (void)compute_br_metrics_phase1_no_distal(sg);
    if (eager_nearest_border_fill) {
        fill_internal_to_nearest_border(sg);
    }
}

void audit_phase1_local_invariants(
    const Subgraph& sg,
    VertexId seed,
    VertexId newest,
    bool paper_strict_mode,
    bool eager_nearest_border_fill,
    std::string_view context
) {
    if (!sg.contains(seed)) {
        throw std::runtime_error(std::string(context) + ": seed missing from subgraph");
    }
    if (!sg.bound_vertices.contains(seed)) {
        throw std::runtime_error(std::string(context) + ": seed is not boundary");
    }
    if (paper_strict_mode && newest != kInvalidVertex) {
        if (!sg.contains(newest)) {
            throw std::runtime_error(std::string(context) + ": newest missing from subgraph");
        }
        if (!sg.bound_vertices.contains(newest)) {
            throw std::runtime_error(std::string(context) + ": newest is not boundary under paper-strict mode");
        }
    }

    for (const auto v : sg.graph.vertex_set()) {
        const bool is_boundary = sg.bound_vertices.contains(v);
        const bool is_internal = sg.internal_vertices.contains(v);
        if (is_boundary && is_internal) {
            throw std::runtime_error(std::string(context) + ": vertex appears in both boundary and internal sets");
        }
        if (!is_boundary && !is_internal) {
            throw std::runtime_error(std::string(context) + ": vertex is missing from both boundary and internal sets");
        }
    }

    auto shadow = clone_subgraph_for_phase1(sg);
    clear_query_derived_state(shadow);
    if (!compute_br_metrics_phase1_no_distal(shadow)) {
        throw std::runtime_error(std::string(context) + ": phase-1 BR invariant failed");
    }
    if (shadow.rb_map.size() != sg.bound_vertices.size()) {
        throw std::runtime_error(std::string(context) + ": recomputed rb_map size does not match boundary size");
    }
    if (sg.rb_map.size() != shadow.rb_map.size()) {
        throw std::runtime_error(std::string(context) + ": rb_map size does not match recomputed phase-1 BR cache");
    }
    for (const auto& [boundary, expected_rb] : shadow.rb_map) {
        const auto it = sg.rb_map.find(boundary);
        if (it == sg.rb_map.end()) {
            throw std::runtime_error(std::string(context) + ": rb_map missing boundary entry");
        }
        if (!(it->second == expected_rb)) {
            throw std::runtime_error(std::string(context) + ": rb_map entry mismatch");
        }
    }

    if (!eager_nearest_border_fill) {
        return;
    }

    for (const auto v : sg.internal_vertices) {
        const auto it = sg.internal_to_nearest_border_dist.find(v);
        if (it == sg.internal_to_nearest_border_dist.end()) {
            throw std::runtime_error(std::string(context) + ": nearest-border cache missing internal vertex");
        }
        EdgeWeight expected = kInfWeight;
        for (const auto b : sg.bound_vertices) {
            expected = std::min(expected, sg.distance.get_or_inf(v, b));
        }
        if (it->second != expected) {
            throw std::runtime_error(std::string(context) + ": nearest-border cache mismatch");
        }
    }
}

void audit_phase1_rollback_cleanup(
    const Subgraph& sg,
    const Phase1LeafRollback& rollback,
    bool paper_strict_mode,
    bool eager_nearest_border_fill
) {
    if (sg.contains(rollback.new_vertex)) {
        throw std::runtime_error("phase1_rollback: rolled-back vertex still present in graph");
    }
    if (sg.bound_vertices.contains(rollback.new_vertex) || sg.internal_vertices.contains(rollback.new_vertex)) {
        throw std::runtime_error("phase1_rollback: rolled-back vertex still present in membership sets");
    }
    if (sg.insertion_order.size() != rollback.insertion_order_size) {
        throw std::runtime_error("phase1_rollback: insertion order size mismatch");
    }
    if (!sg.distance.distance_from(rollback.new_vertex).empty()) {
        throw std::runtime_error("phase1_rollback: rolled-back vertex still has distance row");
    }
    for (const auto v : sg.graph.vertex_set()) {
        if (sg.distance.get_or_inf(v, rollback.new_vertex) != kInfWeight) {
            throw std::runtime_error("phase1_rollback: rolled-back vertex still reachable in distance table");
        }
    }
    const auto newest =
        sg.insertion_order.empty() ? sg.seed_vertex : sg.insertion_order.back();
    audit_phase1_local_invariants(
        sg,
        sg.seed_vertex,
        newest,
        paper_strict_mode,
        eager_nearest_border_fill,
        "phase1_rollback"
    );
}

EdgeWeight edge_weight_any(const Graph& graph, VertexId u, VertexId v) {
    if (const auto forward = graph.get_weight(u, v); forward.has_value()) {
        return *forward;
    }
    if (const auto reverse = graph.get_weight(v, u); reverse.has_value()) {
        return *reverse;
    }
    throw std::runtime_error("edge not found");
}

void fill_internal_to_nearest_border(Subgraph& sg) {
    sg.internal_to_nearest_border_dist.clear();
    for (const auto v : sg.internal_vertices) {
        EdgeWeight best = kInfWeight;
        for (const auto b : sg.bound_vertices) {
            best = std::min(best, sg.distance.get_or_inf(v, b));
        }
        sg.internal_to_nearest_border_dist[v] = best;
    }
}

EdgeWeight nearest_border_distance_full(const Subgraph& sg, VertexId v) {
    EdgeWeight best = kInfWeight;
    for (const auto b : sg.bound_vertices) {
        best = std::min(best, sg.distance.get_or_inf(v, b));
    }
    return best;
}

double estimate_full_cover_power(const Subgraph& sg) {
    double avg_rb = 0.0;
    if (!sg.rb_map.empty()) {
        for (const auto& [_, rb] : sg.rb_map) {
            avg_rb += rb.to_double();
        }
        avg_rb /= static_cast<double>(sg.rb_map.size());
    }
    return static_cast<double>(sg.internal_vertices.size()) * 2.0 + avg_rb;
}

void refresh_internal_vertices(
    Subgraph& sg,
    const Graph& global,
    VertexId seed,
    VertexId newest,
    bool paper_strict_mode
) {
    for (const auto v : sg.graph.vertex_set()) {
        // Rust VFIP keeps the seed as a boundary vertex during phase 1.
        // "Paper strict" additionally pins the newest inserted vertex.
        if (v == seed || (paper_strict_mode && v == newest)) {
            sg.bound_vertices.insert(v);
            sg.internal_vertices.erase(v);
            continue;
        }
        // Rust phase 1 keeps degree-1 non-seed vertices on the boundary even
        // when they are locally internalizable.
        if (sg.check_internal_vertex(global, v) && global.neighbors(v).size() != 1U) {
            sg.turn_to_internal(v);
        } else {
            sg.internal_vertices.erase(v);
            sg.bound_vertices.insert(v);
        }
    }
}

void refresh_internal_vertices_phase1_affected_only(
    Subgraph& sg,
    const Graph& global,
    VertexId seed,
    VertexId newest,
    VertexId attach,
    VertexId previous_newest,
    bool paper_strict_mode
) {
    auto force_boundary = [&](VertexId v) {
        if (v == kInvalidVertex || !sg.contains(v)) {
            return;
        }
        sg.internal_vertices.erase(v);
        sg.bound_vertices.insert(v);
    };

    auto refresh_one = [&](VertexId v) {
        if (v == kInvalidVertex || !sg.contains(v)) {
            return;
        }
        if (v == seed || (paper_strict_mode && v == newest)) {
            force_boundary(v);
            return;
        }
        if (sg.check_internal_vertex(global, v) && global.neighbors(v).size() != 1U) {
            sg.turn_to_internal(v);
        } else {
            force_boundary(v);
        }
    };

    force_boundary(seed);
    if (paper_strict_mode) {
        force_boundary(newest);
    }

    refresh_one(attach);
    refresh_one(previous_newest);
    if (!paper_strict_mode) {
        refresh_one(newest);
    }
}

bool compute_br_metrics(Subgraph& sg) {
    sg.rb_map.clear();
    if (sg.bound_vertices.empty()) {
        return false;
    }

    const auto edges = sg.graph.undirected_edges_unsorted();
    for (const auto b : sg.bound_vertices) {
        HalfWeight rb{0, false};
        for (const auto other : sg.bound_vertices) {
            rb = std::max(rb, HalfWeight{sg.distance.get_or_inf(b, other), false});
        }

        HalfWeight ib{0, false};
        for (const auto v : sg.internal_vertices) {
            ib = std::max(ib, HalfWeight{sg.distance.get_or_inf(b, v), false});
        }
        for (const auto& [edge, w] : edges) {
            const auto lhs = sg.distance.get_or_inf(b, edge.first);
            const auto rhs = sg.distance.get_or_inf(b, edge.second);
            if (lhs == kInfWeight || rhs == kInfWeight) {
                continue;
            }
            ib = std::max(ib, distal_point_distance(lhs, rhs, w));
        }

        if (rb < ib) {
            return false;
        }
        sg.rb_map[b] = rb;
    }
    return true;
}

bool compute_br_metrics_phase1_no_distal(Subgraph& sg) {
    sg.rb_map.clear();
    if (sg.bound_vertices.empty()) {
        return false;
    }

    for (const auto b : sg.bound_vertices) {
        HalfWeight rb{0, false};
        for (const auto other : sg.bound_vertices) {
            rb = std::max(rb, HalfWeight{sg.distance.get_or_inf(b, other), false});
        }

        HalfWeight ib{0, false};
        for (const auto v : sg.internal_vertices) {
            ib = std::max(ib, HalfWeight{sg.distance.get_or_inf(b, v), false});
        }

        if (rb < ib) {
            return false;
        }
        sg.rb_map[b] = rb;
    }
    return true;
}

bool finalize_candidate(
    Subgraph& candidate,
    const Graph& global,
    VertexId seed,
    VertexId newest,
    bool paper_strict_mode,
    bool eager_nearest_border_fill,
    PartitionRuntimeStats* stats
) {
    const auto t_refresh = std::chrono::steady_clock::now();
    refresh_internal_vertices(candidate, global, seed, newest, paper_strict_mode);
    if (stats != nullptr) {
        stats->phase2_refresh_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_refresh
        ).count();
    }

    const auto t_apsp = std::chrono::steady_clock::now();
    candidate.distance = all_pairs_shortest_paths(candidate.graph);
    if (stats != nullptr) {
        stats->phase2_apsp_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_apsp
        ).count();
    }

    const auto t_br = std::chrono::steady_clock::now();
    if (!compute_br_metrics(candidate)) {
        if (stats != nullptr) {
            stats->phase2_br_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_br
            ).count();
        }
        return false;
    }
    if (stats != nullptr) {
        stats->phase2_br_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_br
        ).count();
    }
    if (eager_nearest_border_fill) {
        const auto t_nearest = std::chrono::steady_clock::now();
        fill_internal_to_nearest_border(candidate);
        if (stats != nullptr) {
            stats->phase2_nearest_border_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_nearest
            ).count();
        }
    }
    return true;
}

bool finalize_candidate_with_incremental_distance_update(
    Subgraph& candidate,
    const DistanceTable& base_distance,
    VertexId inserted_u,
    VertexId inserted_v,
    EdgeWeight inserted_weight,
    const Graph& global,
    VertexId seed,
    VertexId newest,
    bool paper_strict_mode,
    bool eager_nearest_border_fill,
    PartitionRuntimeStats* stats
) {
    const auto t_refresh = std::chrono::steady_clock::now();
    refresh_internal_vertices(candidate, global, seed, newest, paper_strict_mode);
    if (stats != nullptr) {
        stats->phase2_refresh_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_refresh
        ).count();
    }

    const auto t_update = std::chrono::steady_clock::now();
    candidate.distance = update_all_pairs_after_undirected_edge_insertion(
        base_distance,
        candidate.graph,
        inserted_u,
        inserted_v,
        inserted_weight
    );
    if (stats != nullptr) {
        stats->phase2_distance_update_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_update
        ).count();
    }

    const auto t_br = std::chrono::steady_clock::now();
    if (!compute_br_metrics(candidate)) {
        if (stats != nullptr) {
            stats->phase2_br_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_br
            ).count();
        }
        return false;
    }
    if (stats != nullptr) {
        stats->phase2_br_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_br
        ).count();
    }

    if (eager_nearest_border_fill) {
        const auto t_nearest = std::chrono::steady_clock::now();
        fill_internal_to_nearest_border(candidate);
        if (stats != nullptr) {
            stats->phase2_nearest_border_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_nearest
            ).count();
        }
    }
    return true;
}

void extend_distance_for_new_leaf(
    Subgraph& candidate,
    VertexId attach,
    VertexId new_vertex,
    EdgeWeight weight
) {
    candidate.distance.set(new_vertex, new_vertex, 0);
    candidate.distance.set(attach, new_vertex, weight);
    candidate.distance.set(new_vertex, attach, weight);

    for (const auto v : candidate.graph.vertex_set()) {
        if (v == new_vertex || v == attach) {
            continue;
        }
        const auto base = candidate.distance.get_or_inf(v, attach);
        if (base == kInfWeight || base > kInfWeight - weight) {
            continue;
        }
        const auto extended = static_cast<EdgeWeight>(base + weight);
        candidate.distance.set(v, new_vertex, extended);
        candidate.distance.set(new_vertex, v, extended);
    }
}

bool finalize_phase1_candidate_incremental(
    Subgraph& candidate,
    const Graph& global,
    VertexId seed,
    VertexId newest,
    VertexId attach,
    EdgeWeight edge_weight,
    bool paper_strict_mode,
    bool eager_nearest_border_fill,
    PartitionRuntimeStats* stats,
    bool skip_phase1_br_check
) {
    clear_query_derived_state(candidate);
    const auto previous_newest =
        candidate.insertion_order.size() >= 2U
            ? candidate.insertion_order[candidate.insertion_order.size() - 2U]
            : kInvalidVertex;
    const auto t_refresh = std::chrono::steady_clock::now();
    refresh_internal_vertices_phase1_affected_only(
        candidate,
        global,
        seed,
        newest,
        attach,
        previous_newest,
        paper_strict_mode
    );
    if (stats != nullptr) {
        stats->phase1_refresh_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_refresh
        ).count();
    }
    const auto t_extend = std::chrono::steady_clock::now();
    extend_distance_for_new_leaf(candidate, attach, newest, edge_weight);
    if (stats != nullptr) {
        stats->phase1_extend_distance_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_extend
        ).count();
    }
    if (!skip_phase1_br_check) {
        const auto t_br = std::chrono::steady_clock::now();
        if (!compute_br_metrics_phase1_no_distal(candidate)) {
            if (stats != nullptr) {
                stats->phase1_br_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_br
                ).count();
            }
            return false;
        }
        if (stats != nullptr) {
            stats->phase1_br_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_br
            ).count();
        }
    }
    if (eager_nearest_border_fill) {
        const auto t_nearest = std::chrono::steady_clock::now();
        fill_internal_to_nearest_border(candidate);
        if (stats != nullptr) {
            stats->phase1_nearest_border_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_nearest
            ).count();
        }
    }
    return true;
}

void minimize_borders(Subgraph& sg, const Graph& global, bool eager_nearest_border_fill) {
    if (sg.bound_vertices.size() <= 2) {
        return;
    }

    std::vector<VertexId> order;
    std::unordered_set<VertexId> added;

    auto push_if_new = [&](VertexId v) {
        if (v != kInvalidVertex && sg.bound_vertices.contains(v) && added.insert(v).second) {
            order.push_back(v);
        }
    };

    push_if_new(sg.seed_vertex);
    if (!sg.insertion_order.empty()) {
        push_if_new(sg.insertion_order.back());
    }
    for (auto it = sg.insertion_order.rbegin(); it != sg.insertion_order.rend(); ++it) {
        push_if_new(*it);
    }
    for (const auto v : sg.bound_vertices) {
        push_if_new(v);
    }

    for (const auto candidate_vertex : order) {
        if (!sg.bound_vertices.contains(candidate_vertex)) {
            continue;
        }
        if (sg.bound_vertices.size() <= 2) {
            break;
        }
        if (!sg.check_internal_vertex(global, candidate_vertex)) {
            continue;
        }

        Subgraph candidate = sg;
        candidate.turn_to_internal(candidate_vertex);
        if (!compute_br_metrics(candidate)) {
            continue;
        }
        if (eager_nearest_border_fill) {
            fill_internal_to_nearest_border(candidate);
        }
        sg = std::move(candidate);
    }
}

struct ShortcutRepartitionOptions {
    std::size_t small_upper_bound{3};
    std::size_t k_neighbors{200};
    EdgeWeight radius_limit{6000};
    std::size_t max_tiny_subgraphs{100000};
    bool eager_nearest_border_fill{true};
};

struct ShortcutRepartitionResult {
    std::vector<Subgraph> subgraphs;
    ShortcutRepartitionStats stats;
};

std::unordered_set<Edge, PairHash> subgraph_edge_set(const Subgraph& sg) {
    std::unordered_set<Edge, PairHash> edges;
    for (const auto& [edge, weight] : sg.graph.undirected_edges()) {
        (void)weight;
        edges.insert(edge);
    }
    return edges;
}

Graph build_graph_from_edge_set(const Graph& global, const std::unordered_set<Edge, PairHash>& edges) {
    Graph graph;
    for (const auto& edge : edges) {
        graph.insert(edge.first);
        graph.insert(edge.second);
        graph.connect(edge.first, edge.second, edge_weight_any(global, edge.first, edge.second));
    }
    return graph;
}

std::unordered_map<VertexId, std::vector<std::size_t>> build_small_boundary_index(
    const std::vector<const Subgraph*>& small_subs
) {
    std::unordered_map<VertexId, std::vector<std::size_t>> index;
    for (std::size_t i = 0; i < small_subs.size(); ++i) {
        for (const auto b : small_subs[i]->bound_vertices) {
            index[b].push_back(i);
        }
    }
    return index;
}

std::vector<std::pair<std::size_t, EdgeWeight>> knn_small_neighbors(
    std::size_t sid,
    const std::vector<const Subgraph*>& small_subs,
    const Graph& global,
    const std::unordered_map<VertexId, std::vector<std::size_t>>& boundary_to_small,
    std::size_t k,
    EdgeWeight radius_limit
) {
    const auto* source = small_subs[sid];
    if (source->bound_vertices.empty()) {
        return {};
    }

    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    DistanceMap dist;
    dist.reserve(source->bound_vertices.size() * 2 + 16);
    for (const auto b : source->bound_vertices) {
        dist[b] = 0;
        pq.push({0, b});
    }

    std::unordered_map<std::size_t, EdgeWeight> best;
    while (!pq.empty()) {
        const auto [d, v] = pq.top();
        pq.pop();
        const auto it = dist.find(v);
        if (it == dist.end() || d != it->second) {
            continue;
        }
        if (radius_limit > 0 && d > radius_limit) {
            break;
        }

        if (const auto boundary_it = boundary_to_small.find(v); boundary_it != boundary_to_small.end()) {
            for (const auto other_id : boundary_it->second) {
                if (other_id == sid) {
                    continue;
                }
                auto best_it = best.find(other_id);
                if (best_it == best.end() || d < best_it->second) {
                    best[other_id] = d;
                }
            }
        }

        if (best.size() >= k && !best.empty()) {
            EdgeWeight worst = 0;
            for (const auto& [_, value] : best) {
                worst = std::max(worst, value);
            }
            if (d > worst) {
                break;
            }
        }

        for (const auto& [nv, w] : global.neighbors(v)) {
            if (d > kInfWeight - w) {
                continue;
            }
            const auto nd = static_cast<EdgeWeight>(d + w);
            auto best_it = dist.find(nv);
            if (best_it == dist.end() || nd < best_it->second) {
                dist[nv] = nd;
                pq.push({nd, nv});
            }
        }
    }

    std::vector<std::pair<std::size_t, EdgeWeight>> items(best.begin(), best.end());
    std::sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
            return lhs.second < rhs.second;
        }
        return lhs.first < rhs.first;
    });
    if (items.size() > k) {
        items.resize(k);
    }
    return items;
}

class DSU {
public:
    explicit DSU(const std::vector<std::size_t>& sizes)
        : parent_(sizes.size()),
          size_vertices_(sizes) {
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            parent_[i] = i;
        }
    }

    std::size_t find(std::size_t x) {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    bool unite_if_fit(std::size_t a, std::size_t b, std::size_t theta) {
        auto x = find(a);
        auto y = find(b);
        if (x == y) {
            return false;
        }
        const auto new_size = size_vertices_[x] + size_vertices_[y];
        if (new_size > theta) {
            return false;
        }
        if (size_vertices_[x] < size_vertices_[y]) {
            std::swap(x, y);
        }
        parent_[y] = x;
        size_vertices_[x] = new_size;
        return true;
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> size_vertices_;
};

ShortcutRepartitionResult shortcut_local_repartition_tiny_only(
    const Graph& global,
    std::size_t theta,
    std::vector<Subgraph> subgraphs,
    const ShortcutRepartitionOptions& options
) {
    ShortcutRepartitionResult result;
    std::vector<Subgraph> keep;
    std::vector<std::pair<std::size_t, Subgraph>> tiny;
    keep.reserve(subgraphs.size());
    for (std::size_t i = 0; i < subgraphs.size(); ++i) {
        auto& sg = subgraphs[i];
        if (sg.graph.size() <= options.small_upper_bound) {
            tiny.push_back({i, std::move(sg)});
        } else {
            keep.push_back(std::move(sg));
        }
    }
    result.stats.tiny_subgraphs = tiny.size();
    if (tiny.empty()) {
        for (std::size_t i = 0; i < keep.size(); ++i) {
            keep[i].id = i;
        }
        result.subgraphs = std::move(keep);
        return result;
    }

    if (options.max_tiny_subgraphs > 0 && tiny.size() > options.max_tiny_subgraphs) {
        result.stats.skipped_by_tiny_budget = true;
        for (auto& [_, sg] : tiny) {
            keep.push_back(std::move(sg));
        }
        for (std::size_t i = 0; i < keep.size(); ++i) {
            keep[i].id = i;
        }
        result.subgraphs = std::move(keep);
        return result;
    }

    std::vector<const Subgraph*> tiny_refs;
    tiny_refs.reserve(tiny.size());
    std::vector<std::size_t> sizes;
    sizes.reserve(tiny.size());
    for (const auto& [_, sg] : tiny) {
        tiny_refs.push_back(&sg);
        sizes.push_back(sg.graph.size());
    }

    const auto boundary_to_small = build_small_boundary_index(tiny_refs);
    std::vector<std::tuple<std::size_t, std::size_t, EdgeWeight>> edges;
    for (std::size_t sid = 0; sid < tiny_refs.size(); ++sid) {
        const auto neighbors = knn_small_neighbors(
            sid,
            tiny_refs,
            global,
            boundary_to_small,
            options.k_neighbors,
            options.radius_limit
        );
        for (const auto& [oid, d] : neighbors) {
            edges.push_back({sid, oid, d});
        }
    }

    std::sort(edges.begin(), edges.end(), [](const auto& lhs, const auto& rhs) {
        if (std::get<2>(lhs) != std::get<2>(rhs)) {
            return std::get<2>(lhs) < std::get<2>(rhs);
        }
        if (std::get<0>(lhs) != std::get<0>(rhs)) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        }
        return std::get<1>(lhs) < std::get<1>(rhs);
    });

    std::vector<std::tuple<std::size_t, std::size_t, EdgeWeight>> undirected;
    std::unordered_set<std::uint64_t> seen_pairs;
    undirected.reserve(edges.size());
    for (const auto& [a0, b0, d] : edges) {
        const auto a = std::min(a0, b0);
        const auto b = std::max(a0, b0);
        if (a == b) {
            continue;
        }
        const auto key = (static_cast<std::uint64_t>(a) << 32U) | static_cast<std::uint64_t>(b);
        if (seen_pairs.insert(key).second) {
            undirected.push_back({a, b, d});
        }
    }

    DSU dsu(sizes);
    for (const auto& [u, v, d] : undirected) {
        (void)d;
        dsu.unite_if_fit(u, v, theta);
    }

    std::unordered_map<std::size_t, std::vector<std::size_t>> components;
    for (std::size_t i = 0; i < tiny_refs.size(); ++i) {
        components[dsu.find(i)].push_back(i);
    }

    std::vector<bool> taken_tiny(tiny_refs.size(), false);
    std::vector<Subgraph> merged;
    for (auto& [_, members] : components) {
        if (members.size() <= 1) {
            continue;
        }

        std::unordered_set<Edge, PairHash> edge_union;
        for (const auto sid : members) {
            const auto edges_in_sg = subgraph_edge_set(*tiny_refs[sid]);
            edge_union.insert(edges_in_sg.begin(), edges_in_sg.end());
        }
        if (edge_union.empty()) {
            continue;
        }

        Subgraph sg;
        sg.graph = build_graph_from_edge_set(global, edge_union);
        for (const auto v : sg.graph.vertex_set()) {
            sg.bound_vertices.insert(v);
        }
        for (const auto v : sg.graph.vertex_set()) {
            if (sg.check_internal_vertex(global, v)) {
                sg.turn_to_internal(v);
            }
        }
        sg.distance = all_pairs_shortest_paths(sg.graph);
        if (!compute_br_metrics(sg)) {
            continue;
        }
        if (options.eager_nearest_border_fill) {
            fill_internal_to_nearest_border(sg);
        }
        merged.push_back(std::move(sg));
        ++result.stats.merged_components;

        for (const auto sid : members) {
            taken_tiny[sid] = true;
        }
    }

    for (std::size_t i = 0; i < tiny.size(); ++i) {
        if (!taken_tiny[i]) {
            keep.push_back(std::move(tiny[i].second));
        }
    }
    for (auto& sg : merged) {
        keep.push_back(std::move(sg));
    }
    for (std::size_t i = 0; i < keep.size(); ++i) {
        keep[i].id = i;
    }
    result.stats.merged_output_subgraphs = merged.size();
    result.subgraphs = std::move(keep);
    return result;
}

}  // namespace

bool Subgraph::contains(VertexId v) const {
    return graph.contains(v);
}

std::size_t Subgraph::size() const {
    return graph.size();
}

std::vector<VertexId> Subgraph::vertices() const {
    return graph.vertices();
}

bool Subgraph::check_internal_vertex(const Graph& global, VertexId v) const {
    const auto& local_neighbors = graph.neighbors(v);
    for (const auto& [adj, _] : global.neighbors(v)) {
        if (!graph.contains(adj) || !local_neighbors.contains(adj)) {
            return false;
        }
    }
    return true;
}

void Subgraph::turn_to_internal(VertexId v) {
    bound_vertices.erase(v);
    internal_vertices.insert(v);
}

VfipPartition::VfipPartition(const Graph& global, PartitionOptions options)
    : global_(global), options_(std::move(options)) {
    if (!options_.growth_trace_output.empty()) {
        std::filesystem::create_directories(options_.growth_trace_output.parent_path());
        growth_trace_out_.open(options_.growth_trace_output, std::ios::trunc);
        if (!growth_trace_out_) {
            throw std::runtime_error("failed to open vip growth trace output");
        }
    }
}

void VfipPartition::trace_growth_step(
    int phase_id,
    SgId subgraph_id,
    const std::string& chosen_vertex_or_edge,
    std::size_t before_num_vertices,
    std::size_t after_num_vertices,
    std::size_t before_num_borders,
    std::size_t after_num_borders,
    std::size_t num_vertices_internalized_this_step,
    std::size_t num_vertices_became_border_this_step,
    const std::string& rollback_or_commit
) {
    if (!growth_trace_out_) {
        return;
    }
    const auto before_clique = (before_num_borders * (before_num_borders - 1U)) / 2U;
    const auto after_clique = (after_num_borders * (after_num_borders - 1U)) / 2U;
    growth_trace_out_
        << "{"
        << "\"step_id\":" << next_trace_step_++ << ","
        << "\"phase_id\":" << phase_id << ","
        << "\"subgraph_id\":" << subgraph_id << ","
        << "\"chosen_vertex_or_edge\":\"" << json_escape(chosen_vertex_or_edge) << "\","
        << "\"before_num_vertices\":" << before_num_vertices << ","
        << "\"after_num_vertices\":" << after_num_vertices << ","
        << "\"before_num_borders\":" << before_num_borders << ","
        << "\"after_num_borders\":" << after_num_borders << ","
        << "\"delta_borders\":" << static_cast<long long>(after_num_borders) - static_cast<long long>(before_num_borders) << ","
        << "\"delta_border_clique_edges\":" << static_cast<long long>(after_clique) - static_cast<long long>(before_clique) << ","
        << "\"num_vertices_internalized_this_step\":" << num_vertices_internalized_this_step << ","
        << "\"num_vertices_became_border_this_step\":" << num_vertices_became_border_this_step << ","
        << "\"rollback_or_commit\":\"" << rollback_or_commit << "\""
        << "}\n";
}

bool VfipPartition::has_unallocated_adjacent_edge(VertexId source) const {
    for (const auto& [adj, _] : global_.neighbors(source)) {
        const auto edge = ordered_edge(source, adj);
        if (!added_edges_.contains(pack_pair(edge.first, edge.second))) {
            return true;
        }
    }
    return false;
}

std::vector<std::pair<VertexId, EdgeWeight>> VfipPartition::unallocated_adjacent_edges(VertexId source) const {
    std::vector<std::pair<VertexId, EdgeWeight>> result;
    for (const auto& [adj, weight] : global_.neighbors(source)) {
        const auto edge = ordered_edge(source, adj);
        if (!added_edges_.contains(pack_pair(edge.first, edge.second))) {
            result.push_back({adj, weight});
        }
    }
    return result;
}

Subgraph VfipPartition::expand_next(VertexId seed, SgId provisional_subgraph_id) {
    using HeapItem = std::tuple<EdgeWeight, VertexId, VertexId>;
    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<>> heap;
    std::unordered_map<VertexId, EdgeWeight> best_seed_distance;
    const bool trace_enabled = static_cast<bool>(growth_trace_out_);

    Subgraph subgraph;
    subgraph.seed_vertex = seed;
    subgraph.graph.insert(seed);
    subgraph.bound_vertices.insert(seed);
    subgraph.insertion_order.push_back(seed);
    best_seed_distance[seed] = 0;

    for (const auto& [adj, weight] : unallocated_adjacent_edges(seed)) {
        heap.push({weight, seed, adj});
        best_seed_distance[adj] = weight;
    }

    while (!heap.empty() && subgraph.size() < options_.theta) {
        const auto [cost, from, to] = heap.top();
        heap.pop();
        if (subgraph.contains(to)) {
            continue;
        }
        const auto best_it = best_seed_distance.find(to);
        if (best_it != best_seed_distance.end() && cost > best_it->second) {
            continue;
        }

        const auto before_vertices = subgraph.graph.size();
        const auto before_borders = subgraph.bound_vertices.size();
        const auto old_cover_power =
            (options_.adaptive_z && before_vertices >= 2U) ? estimate_full_cover_power(subgraph) : 0.0;
        std::unordered_set<VertexId> before_internal;
        std::unordered_set<VertexId> before_bound;
        if (trace_enabled) {
            before_internal = subgraph.internal_vertices;
            before_bound = subgraph.bound_vertices;
        }
        const auto weight = edge_weight_any(global_, from, to);

        if (options_.phase1_inplace) {
            const auto rollback = make_phase1_leaf_rollback(subgraph, seed, from, to, options_.paper_strict_mode);
            subgraph.seed_vertex = seed;
            subgraph.graph.insert(to);
            subgraph.graph.connect(from, to, weight);
            subgraph.bound_vertices.insert(to);
            subgraph.insertion_order.push_back(to);
            if (!subgraph.internal_vertices.contains(from)) {
                subgraph.bound_vertices.insert(from);
            }

            ++stats_.phase1_attempts;
            bool phase1_ok = false;
            stats_.phase1_finalize_us += calc_execution_time_in_us([&] {
                phase1_ok = finalize_phase1_candidate_incremental(
                    subgraph,
                    global_,
                    seed,
                    to,
                    from,
                    weight,
                    options_.paper_strict_mode,
                    !options_.defer_nearest_border_fill,
                    &stats_,
                    options_.skip_phase1_br_check && !options_.adaptive_z
                );
            });
            if (!phase1_ok) {
                ++stats_.phase1_rejects;
                if (trace_enabled) {
                    trace_growth_step(
                        1,
                        provisional_subgraph_id,
                        "vertex:" + std::to_string(to) + "@from:" + std::to_string(from),
                        before_vertices,
                        subgraph.graph.size(),
                        before_borders,
                        subgraph.bound_vertices.size(),
                        count_new_members(before_internal, subgraph.internal_vertices),
                        count_new_members(before_bound, subgraph.bound_vertices),
                        "rollback"
                    );
                }
                rollback_phase1_leaf_inplace(
                    subgraph,
                    rollback,
                    !options_.defer_nearest_border_fill
                );
                if (options_.phase1_local_audit) {
                    audit_phase1_rollback_cleanup(
                        subgraph,
                        rollback,
                        options_.paper_strict_mode,
                        !options_.defer_nearest_border_fill
                    );
                }
                continue;
            }
            ++stats_.phase1_commits;

            if (options_.adaptive_z && before_vertices >= 2U) {
                const double gain = estimate_full_cover_power(subgraph) - old_cover_power;
                const double old_cost = static_cast<double>(before_borders * before_borders);
                const double new_cost =
                    static_cast<double>(subgraph.bound_vertices.size() * subgraph.bound_vertices.size());
                const double cost = new_cost - old_cost;
                if (gain <= 0.0 || cost > gain * options_.adaptive_alpha) {
                    if (trace_enabled) {
                        trace_growth_step(
                            1,
                            provisional_subgraph_id,
                            "vertex:" + std::to_string(to) + "@from:" + std::to_string(from),
                            before_vertices,
                            subgraph.graph.size(),
                            before_borders,
                            subgraph.bound_vertices.size(),
                            count_new_members(before_internal, subgraph.internal_vertices),
                            count_new_members(before_bound, subgraph.bound_vertices),
                            "rollback"
                        );
                    }
                    rollback_phase1_leaf_inplace(
                        subgraph,
                        rollback,
                        !options_.defer_nearest_border_fill
                    );
                    if (options_.phase1_local_audit) {
                        audit_phase1_rollback_cleanup(
                            subgraph,
                            rollback,
                            options_.paper_strict_mode,
                            !options_.defer_nearest_border_fill
                        );
                    }
                    break;
                }
            }

            if (trace_enabled) {
                trace_growth_step(
                    1,
                    provisional_subgraph_id,
                    "vertex:" + std::to_string(to) + "@from:" + std::to_string(from),
                    before_vertices,
                    subgraph.graph.size(),
                    before_borders,
                    subgraph.bound_vertices.size(),
                    count_new_members(before_internal, subgraph.internal_vertices),
                    count_new_members(before_bound, subgraph.bound_vertices),
                    "commit"
                );
            }
        } else {
            const auto phase1_clone_started = std::chrono::steady_clock::now();
            Subgraph candidate = clone_subgraph_for_phase1(subgraph);
            stats_.phase1_clone_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - phase1_clone_started
            ).count();
            candidate.seed_vertex = seed;
            candidate.graph.insert(to);
            candidate.graph.connect(from, to, weight);
            candidate.bound_vertices.insert(to);
            candidate.insertion_order.push_back(to);
            if (!candidate.internal_vertices.contains(from)) {
                candidate.bound_vertices.insert(from);
            }

            ++stats_.phase1_attempts;
            bool phase1_ok = false;
            stats_.phase1_finalize_us += calc_execution_time_in_us([&] {
                phase1_ok = finalize_phase1_candidate_incremental(
                    candidate,
                    global_,
                    seed,
                    to,
                    from,
                    weight,
                    options_.paper_strict_mode,
                    !options_.defer_nearest_border_fill,
                    &stats_,
                    options_.skip_phase1_br_check && !options_.adaptive_z
                );
            });
            if (!phase1_ok) {
                ++stats_.phase1_rejects;
                if (trace_enabled) {
                    trace_growth_step(
                        1,
                        provisional_subgraph_id,
                        "vertex:" + std::to_string(to) + "@from:" + std::to_string(from),
                        before_vertices,
                        candidate.graph.size(),
                        before_borders,
                        candidate.bound_vertices.size(),
                        count_new_members(before_internal, candidate.internal_vertices),
                        count_new_members(before_bound, candidate.bound_vertices),
                        "rollback"
                    );
                }
                continue;
            }
            ++stats_.phase1_commits;

            if (options_.adaptive_z && before_vertices >= 2U) {
                const double gain = estimate_full_cover_power(candidate) - old_cover_power;
                const double old_cost = static_cast<double>(before_borders * before_borders);
                const double new_cost =
                    static_cast<double>(candidate.bound_vertices.size() * candidate.bound_vertices.size());
                const double cost = new_cost - old_cost;
                if (gain <= 0.0 || cost > gain * options_.adaptive_alpha) {
                    if (trace_enabled) {
                        trace_growth_step(
                            1,
                            provisional_subgraph_id,
                            "vertex:" + std::to_string(to) + "@from:" + std::to_string(from),
                            before_vertices,
                            candidate.graph.size(),
                            before_borders,
                            candidate.bound_vertices.size(),
                            count_new_members(before_internal, candidate.internal_vertices),
                            count_new_members(before_bound, candidate.bound_vertices),
                            "rollback"
                        );
                    }
                    break;
                }
            }

            if (trace_enabled) {
                trace_growth_step(
                    1,
                    provisional_subgraph_id,
                    "vertex:" + std::to_string(to) + "@from:" + std::to_string(from),
                    before_vertices,
                    candidate.graph.size(),
                    before_borders,
                    candidate.bound_vertices.size(),
                    count_new_members(before_internal, candidate.internal_vertices),
                    count_new_members(before_bound, candidate.bound_vertices),
                    "commit"
                );
            }
            subgraph = std::move(candidate);
        }
        const auto edge = ordered_edge(from, to);
        added_edges_.insert(pack_pair(edge.first, edge.second));

        const auto seed_to_to = subgraph.distance.get_or_inf(seed, to);
        for (const auto& [adj, weight_out] : unallocated_adjacent_edges(to)) {
            if (subgraph.contains(adj)) {
                continue;
            }
            if (seed_to_to == kInfWeight || seed_to_to > kInfWeight - weight_out) {
                continue;
            }
            const auto next_cost = static_cast<EdgeWeight>(seed_to_to + weight_out);
            auto it = best_seed_distance.find(adj);
            if (it == best_seed_distance.end() || next_cost < it->second) {
                best_seed_distance[adj] = next_cost;
                heap.push({next_cost, to, adj});
            }
        }
    }

    std::vector<std::pair<Edge, EdgeWeight>> potential_edges_sorted;
    for (const auto b : subgraph.bound_vertices) {
        for (const auto& [dst, weight] : unallocated_adjacent_edges(b)) {
            if (subgraph.contains(dst) && !subgraph.graph.has_edge(b, dst)) {
                potential_edges_sorted.push_back({ordered_edge(b, dst), weight});
            }
        }
    }
    std::sort(potential_edges_sorted.begin(), potential_edges_sorted.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
            return lhs.second < rhs.second;
        }
        return lhs.first < rhs.first;
    });
    potential_edges_sorted.erase(
        std::unique(
            potential_edges_sorted.begin(),
            potential_edges_sorted.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; }
        ),
        potential_edges_sorted.end()
    );

    std::sort(potential_edges_sorted.begin(), potential_edges_sorted.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
            return lhs.second > rhs.second;
        }
        return lhs.first > rhs.first;
    });
    std::deque<std::pair<Edge, EdgeWeight>> potential_edges(
        potential_edges_sorted.begin(),
        potential_edges_sorted.end()
    );

    if (options_.skip_phase1_br_check && !options_.adaptive_z) {
        if (!compute_br_metrics_phase1_no_distal(subgraph)) {
            throw std::runtime_error("phase-1 BR validation failed after deferred checking");
        }
        if (options_.phase1_local_audit) {
            const auto newest =
                subgraph.insertion_order.empty() ? seed : subgraph.insertion_order.back();
            audit_phase1_local_invariants(
                subgraph,
                seed,
                newest,
                options_.paper_strict_mode,
                !options_.defer_nearest_border_fill,
                "phase1_post_deferred_br"
            );
        }
    }

    const auto queue_length = potential_edges.size();
    std::size_t failed_insertion_count = 0;
    while (!potential_edges.empty()) {
        const auto [edge, weight] = potential_edges.back();
        potential_edges.pop_back();
        if ((failed_insertion_count > 0 && potential_edges.empty()) || failed_insertion_count >= queue_length) {
            break;
        }
        if (subgraph.graph.has_edge(edge.first, edge.second)) {
            continue;
        }
        if (added_edges_.contains(pack_pair(edge.first, edge.second))) {
            continue;
        }

        const auto before_vertices = subgraph.graph.size();
        const auto before_borders = subgraph.bound_vertices.size();
        std::unordered_set<VertexId> before_internal;
        std::unordered_set<VertexId> before_bound;
        if (trace_enabled) {
            before_internal = subgraph.internal_vertices;
            before_bound = subgraph.bound_vertices;
        }
        const auto phase2_clone_started = std::chrono::steady_clock::now();
        Subgraph candidate = clone_subgraph_for_phase2(subgraph);
        stats_.phase2_clone_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - phase2_clone_started
        ).count();
        candidate.seed_vertex = seed;
        candidate.graph.connect(edge.first, edge.second, weight);
        if (!candidate.internal_vertices.contains(edge.first)) {
            candidate.bound_vertices.insert(edge.first);
        }
        if (!candidate.internal_vertices.contains(edge.second)) {
            candidate.bound_vertices.insert(edge.second);
        }

        ++stats_.phase2_attempts;
        bool phase2_ok = false;
        stats_.phase2_finalize_us += calc_execution_time_in_us([&] {
            if (options_.phase2_incremental_distance_update) {
                phase2_ok = finalize_candidate_with_incremental_distance_update(
                    candidate,
                    subgraph.distance,
                    edge.first,
                    edge.second,
                    weight,
                    global_,
                    seed,
                    seed,
                    options_.paper_strict_mode,
                    !options_.defer_nearest_border_fill,
                    &stats_
                );
            } else {
                phase2_ok = finalize_candidate(
                    candidate,
                    global_,
                    seed,
                    seed,
                    options_.paper_strict_mode,
                    !options_.defer_nearest_border_fill,
                    &stats_
                );
            }
        });
        if (!phase2_ok) {
            ++stats_.phase2_rejects;
            if (trace_enabled) {
                trace_growth_step(
                    2,
                    provisional_subgraph_id,
                    "edge:(" + std::to_string(edge.first) + "," + std::to_string(edge.second) + ")",
                    before_vertices,
                    candidate.graph.size(),
                    before_borders,
                    candidate.bound_vertices.size(),
                    count_new_members(before_internal, candidate.internal_vertices),
                    count_new_members(before_bound, candidate.bound_vertices),
                    "rollback"
                );
            }
            ++failed_insertion_count;
            potential_edges.push_front({edge, weight});
            continue;
        }
        ++stats_.phase2_commits;

        if (trace_enabled) {
            trace_growth_step(
                2,
                provisional_subgraph_id,
                "edge:(" + std::to_string(edge.first) + "," + std::to_string(edge.second) + ")",
                before_vertices,
                candidate.graph.size(),
                before_borders,
                candidate.bound_vertices.size(),
                count_new_members(before_internal, candidate.internal_vertices),
                count_new_members(before_bound, candidate.bound_vertices),
                "commit"
            );
        }
        subgraph = std::move(candidate);
        added_edges_.insert(pack_pair(edge.first, edge.second));
        failed_insertion_count = 0;
    }

    if (options_.border_minimization) {
        minimize_borders(subgraph, global_, !options_.defer_nearest_border_fill);
    }
    if (!options_.defer_nearest_border_fill) {
        fill_internal_to_nearest_border(subgraph);
    }
    return subgraph;
}

std::vector<Subgraph> VfipPartition::run() {
    stats_ = {};
    checkpoint_state_ = {};
    added_edges_.clear();
    std::set<VertexId> seed_vertices;
    std::unordered_set<VertexId> no_progress_seeds;
    const auto partition_started = std::chrono::steady_clock::now();
    checkpoint_state_.enabled =
        !options_.checkpoint_path.empty() &&
        (options_.checkpoint_resume || options_.checkpoint_write || options_.checkpoint_every_subgraphs != 0U);
    checkpoint_state_.resume_requested = options_.checkpoint_resume;
    auto add_seed_if_needed = [&](VertexId v) {
        if (!no_progress_seeds.contains(v) && has_unallocated_adjacent_edge(v)) {
            seed_vertices.insert(v);
        }
    };
    auto remove_unavailable_seeds = [&]() {
        for (auto it = seed_vertices.begin(); it != seed_vertices.end();) {
            if (no_progress_seeds.contains(*it) || !has_unallocated_adjacent_edge(*it)) {
                it = seed_vertices.erase(it);
            } else {
                ++it;
            }
        }
    };

    std::vector<Subgraph> result;
    if (checkpoint_state_.enabled && options_.checkpoint_resume) {
        PartitionCheckpointEntry checkpoint_entry;
        bool loaded = false;
        checkpoint_state_.load_us = calc_execution_time_in_us([&] {
            loaded = load_partition_checkpoint(options_.checkpoint_path, checkpoint_entry);
        });
        if (!loaded) {
            if (options_.checkpoint_resume_required) {
                throw std::runtime_error(
                    "partition checkpoint miss for required resume mode: " +
                    options_.checkpoint_path.string()
                );
            }
        } else {
            if (!options_.checkpoint_key.empty() && checkpoint_entry.key != options_.checkpoint_key) {
                throw std::runtime_error(
                    "partition checkpoint key mismatch: " + options_.checkpoint_path.string()
                );
            }
            stats_ = checkpoint_entry.stats;
            for (const auto v : checkpoint_entry.pending_seed_vertices) {
                seed_vertices.insert(v);
            }
            for (const auto v : checkpoint_entry.no_progress_seeds) {
                no_progress_seeds.insert(v);
            }
            result = std::move(checkpoint_entry.subgraphs);
            checkpoint_state_.resume_hit = true;
            checkpoint_state_.resumed_subgraphs = result.size();
            for (const auto& sg : result) {
                for (const auto& [edge, _] : sg.graph.undirected_edges_unsorted()) {
                    const auto ordered = ordered_edge(edge.first, edge.second);
                    added_edges_.insert(pack_pair(ordered.first, ordered.second));
                }
            }
        }
    }

    if (!checkpoint_state_.resume_hit && has_unallocated_adjacent_edge(options_.partition_seed)) {
        seed_vertices.insert(options_.partition_seed);
    }

    auto save_checkpoint = [&](const std::vector<Subgraph>& finished_subgraphs) {
        if (!checkpoint_state_.enabled || !options_.checkpoint_write) {
            return;
        }
        const std::vector<VertexId> pending_seed_vertices(seed_vertices.begin(), seed_vertices.end());
        const std::vector<VertexId> no_progress_seed_vertices(
            no_progress_seeds.begin(),
            no_progress_seeds.end()
        );
        checkpoint_state_.save_us += calc_execution_time_in_us([&] {
            save_partition_checkpoint(
                options_.checkpoint_path,
                options_.checkpoint_key,
                stats_,
                pending_seed_vertices,
                no_progress_seed_vertices,
                finished_subgraphs
            );
        });
        ++checkpoint_state_.save_count;
        checkpoint_state_.last_saved_subgraphs = finished_subgraphs.size();
    };

    const auto prior_core_partition_us = stats_.core_partition_us;
    stats_.core_partition_us = prior_core_partition_us + calc_execution_time_in_us([&] {
        while (true) {
            remove_unavailable_seeds();
            if (seed_vertices.empty()) {
                for (const auto vertex : global_.vertex_set()) {
                    if (!no_progress_seeds.contains(vertex) && has_unallocated_adjacent_edge(vertex)) {
                        seed_vertices.insert(vertex);
                        break;
                    }
                }
                if (seed_vertices.empty()) {
                    break;
                }
            }

            const auto seed = *seed_vertices.begin();
            seed_vertices.erase(seed);

            const auto edges_before = added_edges_.size();
            auto sg = expand_next(seed, result.size());
            if (added_edges_.size() == edges_before) {
                // Prevent infinite reselection of a seed that cannot allocate any
                // remaining edge under the current semantics. Neighboring vertices
                // can still become future seeds and cover those edges from the
                // opposite side.
                no_progress_seeds.insert(seed);
                for (const auto& [adj, _] : global_.neighbors(seed)) {
                    add_seed_if_needed(adj);
                }
                continue;
            }
            for (const auto b : sg.bound_vertices) {
                add_seed_if_needed(b);
            }
            sg.id = result.size();
            result.push_back(std::move(sg));
            if (checkpoint_state_.enabled &&
                options_.checkpoint_write &&
                options_.checkpoint_every_subgraphs != 0U &&
                (result.size() % options_.checkpoint_every_subgraphs) == 0U) {
                save_checkpoint(result);
            }
            if (options_.checkpoint_stop_after_subgraphs != 0U &&
                result.size() >= options_.checkpoint_stop_after_subgraphs) {
                if (!checkpoint_state_.enabled || checkpoint_state_.last_saved_subgraphs != result.size()) {
                    const std::vector<VertexId> pending_seed_vertices(seed_vertices.begin(), seed_vertices.end());
                    const std::vector<VertexId> no_progress_seed_vertices(
                        no_progress_seeds.begin(),
                        no_progress_seeds.end()
                    );
                    if (checkpoint_state_.enabled) {
                        checkpoint_state_.save_us += calc_execution_time_in_us([&] {
                            save_partition_checkpoint(
                                options_.checkpoint_path,
                                options_.checkpoint_key,
                                stats_,
                                pending_seed_vertices,
                                no_progress_seed_vertices,
                                result
                            );
                        });
                        ++checkpoint_state_.save_count;
                        checkpoint_state_.last_saved_subgraphs = result.size();
                    }
                }
                throw std::runtime_error(
                    "partition stopped after requested subgraph checkpoint: " +
                    std::to_string(result.size())
                );
            }
            if (options_.progress_log &&
                options_.progress_log_every_subgraphs != 0U &&
                (result.size() % options_.progress_log_every_subgraphs) == 0U) {
                const auto elapsed_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - partition_started)
                        .count();
                std::cerr
                    << "[partition-progress] subgraphs=" << result.size()
                    << " added_edges=" << added_edges_.size()
                    << " seed_candidates=" << seed_vertices.size()
                    << " no_progress_seeds=" << no_progress_seeds.size()
                    << " phase1_attempts=" << stats_.phase1_attempts
                    << " phase1_clone_us=" << stats_.phase1_clone_us
                    << " phase2_attempts=" << stats_.phase2_attempts
                    << " phase2_clone_us=" << stats_.phase2_clone_us
                    << " elapsed_us=" << elapsed_us
                    << std::endl;
            }
        }
    });

    if (checkpoint_state_.enabled &&
        options_.checkpoint_write &&
        !result.empty() &&
        checkpoint_state_.last_saved_subgraphs != result.size()) {
        save_checkpoint(result);
    }

    const auto global_edge_count = global_.undirected_edges().size();
    if (added_edges_.size() != global_edge_count) {
        throw std::runtime_error("partition did not cover all global edges");
    }

    stats_.subgraphs_before_shortcut = result.size();
    if (options_.shortcut_repartition) {
        if (options_.progress_log) {
            std::cerr
                << "[partition-progress] entering_shortcut"
                << " subgraphs_before_shortcut=" << stats_.subgraphs_before_shortcut
                << std::endl;
        }
        ShortcutRepartitionResult shortcut_result;
        stats_.shortcut_repartition_us = calc_execution_time_in_us([&] {
            shortcut_result = shortcut_local_repartition_tiny_only(
                global_,
                options_.theta,
                std::move(result),
                ShortcutRepartitionOptions{
                    options_.shortcut_small_upper_bound,
                    options_.shortcut_k_neighbors,
                    options_.shortcut_radius_limit,
                    options_.shortcut_max_tiny_subgraphs,
                    !options_.defer_nearest_border_fill,
                }
            );
        });
        result = std::move(shortcut_result.subgraphs);
        stats_.shortcut_stats = shortcut_result.stats;
    }
    stats_.subgraphs_after_shortcut = result.size();
    if (options_.defer_nearest_border_fill) {
        for (auto& sg : result) {
            fill_internal_to_nearest_border(sg);
        }
    }
    if (options_.progress_log) {
        const auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - partition_started)
                .count();
        std::cerr
            << "[partition-progress] completed"
            << " subgraphs_after_shortcut=" << stats_.subgraphs_after_shortcut
            << " core_partition_us=" << stats_.core_partition_us
            << " shortcut_repartition_us=" << stats_.shortcut_repartition_us
            << " phase1_attempts=" << stats_.phase1_attempts
            << " phase1_clone_us=" << stats_.phase1_clone_us
            << " phase1_finalize_us=" << stats_.phase1_finalize_us
            << " phase2_attempts=" << stats_.phase2_attempts
            << " phase2_clone_us=" << stats_.phase2_clone_us
            << " phase2_finalize_us=" << stats_.phase2_finalize_us
            << " elapsed_us=" << elapsed_us
            << std::endl;
    }

    return result;
}

const PartitionRuntimeStats& VfipPartition::stats() const {
    return stats_;
}

const PartitionCheckpointState& VfipPartition::checkpoint_state() const {
    return checkpoint_state_;
}

}  // namespace bag
