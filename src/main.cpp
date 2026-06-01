#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>
#include <unordered_set>

#include "diagnostics.h"
#include "factorized_transfer.h"
#include "index.h"
#include "loader.h"
#include "object.h"
#include "partition.h"
#include "partition_cache.h"
#include "update_repro.h"
#include "utils.h"

namespace {

using bag::EdgeWeight;
using bag::IndexedMovingObjectSet;
using bag::MovingObjectSet;
using bag::QueryPoint;
using bag::SkeletonIndex;
using bag::VfipPartition;

struct PartitionSummary {
    std::size_t subgraphs{0};
    std::size_t graph_vertices{0};
    std::size_t graph_edges{0};
    std::size_t skeleton_vertices{0};
    std::size_t skeleton_edges{0};
    std::size_t total_boundary_vertices{0};
    std::size_t total_internal_vertices{0};
    std::size_t max_boundary_vertices{0};
    double avg_boundary_per_subgraph{0.0};
    double skeleton_avg_degree{0.0};
    bool br_property_ok{true};
    std::size_t br_violating_subgraphs{0};
    std::size_t br_violating_boundaries{0};
};

struct PartitionAuditSummary {
    PartitionSummary summary;
    std::size_t covered_graph_edges{0};
    std::size_t uncovered_graph_edges{0};
    std::size_t duplicated_graph_edges{0};
    std::size_t seed_missing_subgraphs{0};
    std::size_t seed_not_boundary_subgraphs{0};
    std::size_t membership_overlap_vertices{0};
    std::size_t membership_unassigned_vertices{0};
    std::size_t nearest_border_missing_entries{0};
    std::size_t nearest_border_mismatch_entries{0};
    std::size_t rb_map_missing_entries{0};
    std::size_t rb_map_mismatch_entries{0};
};

bool partition_audit_ok(const PartitionAuditSummary& audit) {
    return audit.summary.br_property_ok &&
           audit.uncovered_graph_edges == 0 &&
           audit.duplicated_graph_edges == 0 &&
           audit.seed_not_boundary_subgraphs == 0 &&
           audit.membership_overlap_vertices == 0 &&
           audit.membership_unassigned_vertices == 0 &&
           audit.nearest_border_missing_entries == 0 &&
           audit.nearest_border_mismatch_entries == 0 &&
           audit.rb_map_missing_entries == 0 &&
           audit.rb_map_mismatch_entries == 0;
}

struct HierarchyLiteSummary {
    std::size_t group_size{0};
    std::size_t parent_regions{0};
    std::size_t avg_children_per_parent_num{0};
    double avg_children_per_parent{0.0};
    std::size_t max_children_per_parent{0};
    std::size_t upper_unique_boundary_vertices{0};
    std::size_t upper_total_boundary_vertices{0};
    std::size_t upper_parent_graph_edges{0};
    std::size_t upper_estimated_clique_edges{0};
    double upper_unique_boundary_ratio_vs_l1{0.0};
    double upper_total_boundary_ratio_vs_l1_total{0.0};
    double upper_estimated_clique_ratio_vs_l1{0.0};
};

struct IndexMemoryReport {
    std::size_t max_vertex_id{0};
    std::size_t skeleton_graph_bytes{0};
    std::size_t sorted_skeleton_rows_bytes{0};
    std::size_t sorted_skeleton_row_index_bytes{0};
    std::size_t vertex_to_subgraph_bytes{0};
    std::size_t inverted_index_bytes{0};
    std::size_t edge_to_subgraph_bytes{0};
    std::size_t subgraph_graph_bytes{0};
    std::size_t subgraph_distance_bytes{0};
    std::size_t boundary_bytes{0};
    std::size_t rb_bytes{0};
    std::size_t internal_to_border_bytes{0};
    std::size_t clique_rows_bytes{0};
    std::size_t local_index_bytes{0};
    std::size_t subgraph_adjacency_bytes{0};
    std::size_t total_bytes{0};
};

struct ObjectMemoryReport {
    std::size_t object_vector_bytes{0};
    std::size_t subgraph_object_lists_bytes{0};
    std::size_t edge_bucket_bytes{0};
    std::size_t knn_sorted_order_bytes{0};
    std::size_t knn_suffix_bytes{0};
    std::size_t knn_border_cost_bytes{0};
    std::size_t total_bytes{0};
};

std::size_t graph_payload_bytes(const bag::Graph& graph) {
    const auto vertices = graph.vertices();
    std::size_t bytes = vertices.size() * sizeof(bag::VertexId);  // vertex set payload
    bytes += vertices.size() * sizeof(bag::VertexId);             // adjacency row keys
    for (const auto v : vertices) {
        bytes += graph.neighbors(v).size() * (sizeof(bag::VertexId) + sizeof(bag::EdgeWeight));
    }
    return bytes;
}

std::unordered_map<bag::VertexId, std::vector<bag::SgId>> build_border_to_subgraphs(
    const std::vector<bag::Subgraph>& subgraphs
) {
    std::unordered_map<bag::VertexId, std::vector<bag::SgId>> border_to_sgs;
    border_to_sgs.reserve(subgraphs.size() * 4U + 1U);
    for (const auto& sg : subgraphs) {
        for (const auto b : sg.bound_vertices) {
            border_to_sgs[b].push_back(sg.id);
        }
    }
    return border_to_sgs;
}

IndexMemoryReport estimate_index_memory_report(
    const bag::Graph& graph,
    const bag::SkeletonIndex& index
) {
    IndexMemoryReport report;
    const auto& subgraphs = index.subgraphs();
    const auto& clique_rows = index.subgraph_clique_rows();
    const auto skeleton_vertices = index.skeleton().vertices();
    for (const auto v : graph.vertices()) {
        report.max_vertex_id = std::max<std::size_t>(report.max_vertex_id, v);
    }

    report.skeleton_graph_bytes = graph_payload_bytes(index.skeleton());
    report.sorted_skeleton_row_index_bytes = (report.max_vertex_id + 1U) * sizeof(std::int32_t);
    for (const auto border : skeleton_vertices) {
        report.sorted_skeleton_rows_bytes +=
            index.skeleton().neighbors(border).size() * (sizeof(bag::VertexId) + sizeof(bag::EdgeWeight));
    }
    report.vertex_to_subgraph_bytes = (report.max_vertex_id + 1U) * sizeof(bag::SgId);

    const auto border_to_sgs = build_border_to_subgraphs(subgraphs);
    report.inverted_index_bytes = (report.max_vertex_id + 1U) * sizeof(std::vector<bag::SgId>);
    for (const auto& [border, ids] : border_to_sgs) {
        (void)border;
        report.inverted_index_bytes += ids.size() * sizeof(bag::SgId);
    }

    report.edge_to_subgraph_bytes =
        index.edge_to_subgraph().size() * (sizeof(bag::Edge) + sizeof(bag::SgId));

    std::vector<std::unordered_set<bag::SgId>> adjacency_sets(subgraphs.size());
    for (const auto& [_, ids] : border_to_sgs) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            for (std::size_t j = i + 1; j < ids.size(); ++j) {
                adjacency_sets[ids[i]].insert(ids[j]);
                adjacency_sets[ids[j]].insert(ids[i]);
            }
        }
    }
    for (const auto& nbrs : adjacency_sets) {
        report.subgraph_adjacency_bytes += nbrs.size() * sizeof(bag::SgId);
    }

    for (const auto& sg : subgraphs) {
        report.subgraph_graph_bytes += graph_payload_bytes(sg.graph);
        report.subgraph_distance_bytes += sg.distance.payload_bytes();
        report.boundary_bytes += sg.bound_vertices.size() * sizeof(bag::VertexId);
        report.rb_bytes += sg.rb_map.size() * (sizeof(bag::VertexId) + sizeof(bag::HalfWeight));
        report.internal_to_border_bytes +=
            sg.internal_to_nearest_border_dist.size() * (sizeof(bag::VertexId) + sizeof(bag::EdgeWeight));

        const auto vertices = sg.graph.vertices();
        report.local_index_bytes += vertices.size() * (sizeof(bag::VertexId) + sizeof(std::uint32_t));
        for (const auto v : vertices) {
            report.local_index_bytes +=
                sg.graph.neighbors(v).size() * (sizeof(std::uint32_t) + sizeof(bag::EdgeWeight));
        }

        if (sg.id < clique_rows.size()) {
            for (const auto& [boundary, row] : clique_rows[sg.id]) {
                (void)boundary;
                report.clique_rows_bytes += row.size() * (sizeof(bag::VertexId) + sizeof(bag::EdgeWeight));
            }
        }
    }

    report.total_bytes =
        report.skeleton_graph_bytes +
        report.sorted_skeleton_rows_bytes +
        report.sorted_skeleton_row_index_bytes +
        report.vertex_to_subgraph_bytes +
        report.inverted_index_bytes +
        report.edge_to_subgraph_bytes +
        report.subgraph_graph_bytes +
        report.subgraph_distance_bytes +
        report.boundary_bytes +
        report.rb_bytes +
        report.internal_to_border_bytes +
        report.clique_rows_bytes +
        report.local_index_bytes +
        report.subgraph_adjacency_bytes;
    return report;
}

ObjectMemoryReport estimate_object_memory_report(
    const bag::SkeletonIndex& index,
    const bag::IndexedMovingObjectSet& objects,
    bool count_knn
) {
    ObjectMemoryReport report;
    report.object_vector_bytes = objects.objects().size() * sizeof(bag::MovingObject);
    for (std::size_t sg_id = 0; sg_id < index.subgraphs().size(); ++sg_id) {
        report.subgraph_object_lists_bytes += objects.objects_in(sg_id).size() * sizeof(bag::ObjId);
        for (const auto& bucket : objects.populated_edges_in(sg_id)) {
            report.edge_bucket_bytes += sizeof(bucket.edge) + sizeof(bucket.edge_weight);
            report.edge_bucket_bytes += bucket.object_ids.size() * sizeof(bag::ObjId);
        }
        if (count_knn) {
            report.knn_sorted_order_bytes += objects.objects_in_knn_order(sg_id).size() * sizeof(bag::ObjId);
        }
    }
    if (count_knn) {
        report.knn_suffix_bytes = objects.objects().size() * sizeof(bag::EdgeWeight);
        for (std::size_t id = 0; id < objects.objects().size(); ++id) {
            report.knn_border_cost_bytes +=
                objects.knn_border_costs(static_cast<bag::ObjId>(id)).size() *
                (sizeof(bag::VertexId) + sizeof(bag::EdgeWeight));
        }
    }
    report.total_bytes =
        report.object_vector_bytes +
        report.subgraph_object_lists_bytes +
        report.edge_bucket_bytes +
        report.knn_sorted_order_bytes +
        report.knn_suffix_bytes +
        report.knn_border_cost_bytes;
    return report;
}

bag::Graph build_skeleton_graph_from_subgraphs(const std::vector<bag::Subgraph>& subgraphs) {
    bag::Graph skeleton;
    for (const auto& sg : subgraphs) {
        std::vector<bag::VertexId> boundaries(sg.bound_vertices.begin(), sg.bound_vertices.end());
        std::sort(boundaries.begin(), boundaries.end());
        for (const auto b : boundaries) {
            skeleton.insert(b);
        }
        for (std::size_t i = 0; i < boundaries.size(); ++i) {
            for (std::size_t j = i + 1; j < boundaries.size(); ++j) {
                const auto distance = sg.distance.get_or_inf(boundaries[i], boundaries[j]);
                if (distance != bag::kInfWeight) {
                    skeleton.set_min_undirected_edge(boundaries[i], boundaries[j], distance);
                }
            }
        }
    }
    return skeleton;
}

std::vector<std::size_t> parse_size_list(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::vector<std::size_t>& fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    std::vector<std::size_t> result;
    std::stringstream ss(it->second);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            result.push_back(static_cast<std::size_t>(std::stoull(token)));
        }
    }
    return result.empty() ? fallback : result;
}

std::vector<std::string> parse_string_list(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::vector<std::string>& fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    std::vector<std::string> result;
    std::stringstream ss(it->second);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            result.push_back(token);
        }
    }
    return result.empty() ? fallback : result;
}

std::vector<std::vector<bag::SgId>> build_subgraph_adjacency(const std::vector<bag::Subgraph>& subgraphs) {
    std::unordered_map<bag::VertexId, std::vector<bag::SgId>> border_to_sgs;
    border_to_sgs.reserve(subgraphs.size() * 4U + 1U);
    for (const auto& sg : subgraphs) {
        for (const auto b : sg.bound_vertices) {
            border_to_sgs[b].push_back(sg.id);
        }
    }

    std::vector<std::unordered_set<bag::SgId>> adjacency_sets(subgraphs.size());
    for (const auto& [_, ids] : border_to_sgs) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            for (std::size_t j = i + 1; j < ids.size(); ++j) {
                adjacency_sets[ids[i]].insert(ids[j]);
                adjacency_sets[ids[j]].insert(ids[i]);
            }
        }
    }

    std::vector<std::vector<bag::SgId>> adjacency(subgraphs.size());
    for (std::size_t i = 0; i < adjacency_sets.size(); ++i) {
        adjacency[i].assign(adjacency_sets[i].begin(), adjacency_sets[i].end());
        std::sort(adjacency[i].begin(), adjacency[i].end());
    }
    return adjacency;
}

HierarchyLiteSummary analyze_hierarchy_lite(
    const std::vector<bag::Subgraph>& subgraphs,
    const PartitionSummary& level1_summary,
    std::size_t group_size
) {
    HierarchyLiteSummary summary;
    summary.group_size = group_size;
    if (subgraphs.empty() || group_size == 0) {
        return summary;
    }

    const auto adjacency = build_subgraph_adjacency(subgraphs);
    std::vector<int> parent_of(subgraphs.size(), -1);
    std::vector<std::vector<bag::SgId>> groups;
    groups.reserve((subgraphs.size() + group_size - 1U) / group_size);

    for (std::size_t seed = 0; seed < subgraphs.size(); ++seed) {
        if (parent_of[seed] != -1) {
            continue;
        }
        const int parent_id = static_cast<int>(groups.size());
        groups.push_back({});
        std::queue<bag::SgId> queue;
        queue.push(static_cast<bag::SgId>(seed));
        parent_of[seed] = parent_id;

        while (!queue.empty() && groups[parent_id].size() < group_size) {
            const auto current = queue.front();
            queue.pop();
            groups[parent_id].push_back(current);
            for (const auto neighbor : adjacency[current]) {
                if (parent_of[neighbor] == -1 &&
                    groups[parent_id].size() + queue.size() < group_size) {
                    parent_of[neighbor] = parent_id;
                    queue.push(neighbor);
                }
            }
        }
    }

    for (std::size_t i = 0; i < parent_of.size(); ++i) {
        if (parent_of[i] == -1) {
            const int parent_id = static_cast<int>(groups.size());
            groups.push_back({static_cast<bag::SgId>(i)});
            parent_of[i] = parent_id;
        }
    }

    summary.parent_regions = groups.size();
    std::size_t children_sum = 0;
    for (const auto& group : groups) {
        children_sum += group.size();
        summary.max_children_per_parent = std::max(summary.max_children_per_parent, group.size());
    }
    summary.avg_children_per_parent_num = children_sum;
    summary.avg_children_per_parent =
        groups.empty() ? 0.0 : static_cast<double>(children_sum) / static_cast<double>(groups.size());

    std::unordered_map<bag::VertexId, std::vector<bag::SgId>> border_to_sgs;
    border_to_sgs.reserve(subgraphs.size() * 4U + 1U);
    for (const auto& sg : subgraphs) {
        for (const auto b : sg.bound_vertices) {
            border_to_sgs[b].push_back(sg.id);
        }
    }

    std::vector<std::unordered_set<bag::VertexId>> parent_exposed_boundaries(groups.size());
    for (const auto& [border, ids] : border_to_sgs) {
        std::unordered_set<int> parents;
        for (const auto sg_id : ids) {
            parents.insert(parent_of[sg_id]);
        }
        if (parents.size() <= 1U) {
            continue;
        }
        ++summary.upper_unique_boundary_vertices;
        summary.upper_total_boundary_vertices += parents.size();
        for (const auto parent_id : parents) {
            parent_exposed_boundaries[static_cast<std::size_t>(parent_id)].insert(border);
        }
    }

    std::unordered_set<std::uint64_t> parent_edges;
    for (std::size_t u = 0; u < adjacency.size(); ++u) {
        for (const auto v : adjacency[u]) {
            const auto pu = static_cast<std::size_t>(parent_of[u]);
            const auto pv = static_cast<std::size_t>(parent_of[v]);
            if (pu == pv) {
                continue;
            }
            const auto edge = bag::ordered_edge(
                static_cast<bag::VertexId>(pu),
                static_cast<bag::VertexId>(pv)
            );
            parent_edges.insert(bag::pack_pair(edge.first, edge.second));
        }
    }
    summary.upper_parent_graph_edges = parent_edges.size();

    for (const auto& exposed : parent_exposed_boundaries) {
        const auto b = exposed.size();
        if (b >= 2U) {
            summary.upper_estimated_clique_edges += (b * (b - 1U)) / 2U;
        }
    }

    summary.upper_unique_boundary_ratio_vs_l1 =
        level1_summary.skeleton_vertices == 0
            ? 0.0
            : static_cast<double>(summary.upper_unique_boundary_vertices) /
                  static_cast<double>(level1_summary.skeleton_vertices);
    summary.upper_total_boundary_ratio_vs_l1_total =
        level1_summary.total_boundary_vertices == 0
            ? 0.0
            : static_cast<double>(summary.upper_total_boundary_vertices) /
                  static_cast<double>(level1_summary.total_boundary_vertices);

    std::size_t level1_clique_edges = 0;
    for (const auto& sg : subgraphs) {
        const auto b = sg.bound_vertices.size();
        if (b >= 2U) {
            level1_clique_edges += (b * (b - 1U)) / 2U;
        }
    }
    summary.upper_estimated_clique_ratio_vs_l1 =
        level1_clique_edges == 0
            ? 0.0
            : static_cast<double>(summary.upper_estimated_clique_edges) /
                  static_cast<double>(level1_clique_edges);

    return summary;
}

PartitionSummary summarize_partition(
    const bag::Graph& graph,
    const bag::Graph& skeleton,
    const std::vector<bag::Subgraph>& subgraphs
) {
    PartitionSummary summary;
    summary.subgraphs = subgraphs.size();
    summary.graph_vertices = graph.size();
    summary.graph_edges = graph.edge_count();
    summary.skeleton_vertices = skeleton.size();
    summary.skeleton_edges = skeleton.undirected_edges().size();

    for (const auto& sg : subgraphs) {
        summary.total_boundary_vertices += sg.bound_vertices.size();
        summary.total_internal_vertices += sg.internal_vertices.size();
        summary.max_boundary_vertices = std::max(summary.max_boundary_vertices, sg.bound_vertices.size());

        bool subgraph_ok = true;
        const auto sg_edges = sg.graph.undirected_edges();
        for (const auto b : sg.bound_vertices) {
            bag::HalfWeight rb{0, false};
            for (const auto other : sg.bound_vertices) {
                rb = std::max(rb, bag::HalfWeight{sg.distance.get_or_inf(b, other), false});
            }

            bag::HalfWeight ib{0, false};
            for (const auto v : sg.internal_vertices) {
                ib = std::max(ib, bag::HalfWeight{sg.distance.get_or_inf(b, v), false});
            }
            for (const auto& [edge, w] : sg_edges) {
                const auto lhs = sg.distance.get_or_inf(b, edge.first);
                const auto rhs = sg.distance.get_or_inf(b, edge.second);
                if (lhs == bag::kInfWeight || rhs == bag::kInfWeight) {
                    continue;
                }
                ib = std::max(ib, bag::distal_point_distance(lhs, rhs, w));
            }

            if (rb < ib) {
                ++summary.br_violating_boundaries;
                subgraph_ok = false;
            }
        }
        if (!subgraph_ok) {
            ++summary.br_violating_subgraphs;
        }
    }

    summary.avg_boundary_per_subgraph =
        subgraphs.empty()
            ? 0.0
            : static_cast<double>(summary.total_boundary_vertices) / static_cast<double>(subgraphs.size());
    summary.skeleton_avg_degree =
        summary.skeleton_vertices == 0
            ? 0.0
            : (2.0 * static_cast<double>(summary.skeleton_edges)) / static_cast<double>(summary.skeleton_vertices);
    summary.br_property_ok = summary.br_violating_boundaries == 0;
    return summary;
}

PartitionAuditSummary audit_partition(
    const bag::Graph& graph,
    const bag::Graph& skeleton,
    const std::vector<bag::Subgraph>& subgraphs
) {
    PartitionAuditSummary audit;
    audit.summary = summarize_partition(graph, skeleton, subgraphs);

    std::unordered_map<std::uint64_t, std::uint32_t> edge_coverage_counts;
    edge_coverage_counts.reserve(graph.edge_count() * 2U + 1U);

    for (const auto& sg : subgraphs) {
        if (!sg.contains(sg.seed_vertex)) {
            ++audit.seed_missing_subgraphs;
        } else if (!sg.bound_vertices.contains(sg.seed_vertex)) {
            ++audit.seed_not_boundary_subgraphs;
        }

        for (const auto v : sg.graph.vertex_set()) {
            const bool is_boundary = sg.bound_vertices.contains(v);
            const bool is_internal = sg.internal_vertices.contains(v);
            if (is_boundary && is_internal) {
                ++audit.membership_overlap_vertices;
            }
            if (!is_boundary && !is_internal) {
                ++audit.membership_unassigned_vertices;
            }
            if (!is_internal) {
                continue;
            }

            const auto it = sg.internal_to_nearest_border_dist.find(v);
            if (it == sg.internal_to_nearest_border_dist.end()) {
                ++audit.nearest_border_missing_entries;
                continue;
            }

            bag::EdgeWeight expected = bag::kInfWeight;
            for (const auto b : sg.bound_vertices) {
                expected = std::min(expected, sg.distance.get_or_inf(v, b));
            }
            if (it->second != expected) {
                ++audit.nearest_border_mismatch_entries;
            }
        }

        const auto sg_edges = sg.graph.undirected_edges_unsorted();
        for (const auto& [edge, weight] : sg_edges) {
            (void)weight;
            const auto ordered = bag::ordered_edge(edge.first, edge.second);
            ++edge_coverage_counts[bag::pack_pair(ordered.first, ordered.second)];
        }

        for (const auto b : sg.bound_vertices) {
            bag::HalfWeight expected_rb{0, false};
            for (const auto other : sg.bound_vertices) {
                expected_rb = std::max(
                    expected_rb,
                    bag::HalfWeight{sg.distance.get_or_inf(b, other), false}
                );
            }

            bag::HalfWeight expected_ib{0, false};
            for (const auto v : sg.internal_vertices) {
                expected_ib = std::max(
                    expected_ib,
                    bag::HalfWeight{sg.distance.get_or_inf(b, v), false}
                );
            }
            for (const auto& [edge, w] : sg_edges) {
                const auto lhs = sg.distance.get_or_inf(b, edge.first);
                const auto rhs = sg.distance.get_or_inf(b, edge.second);
                if (lhs == bag::kInfWeight || rhs == bag::kInfWeight) {
                    continue;
                }
                expected_ib = std::max(expected_ib, bag::distal_point_distance(lhs, rhs, w));
            }

            const auto rb_it = sg.rb_map.find(b);
            if (rb_it == sg.rb_map.end()) {
                ++audit.rb_map_missing_entries;
            } else if (!(rb_it->second == expected_rb)) {
                ++audit.rb_map_mismatch_entries;
            }
        }
    }

    for (const auto& [edge, weight] : graph.undirected_edges_unsorted()) {
        (void)weight;
        const auto ordered = bag::ordered_edge(edge.first, edge.second);
        const auto key = bag::pack_pair(ordered.first, ordered.second);
        const auto it = edge_coverage_counts.find(key);
        if (it == edge_coverage_counts.end()) {
            ++audit.uncovered_graph_edges;
            continue;
        }
        ++audit.covered_graph_edges;
        if (it->second > 1U) {
            ++audit.duplicated_graph_edges;
        }
    }

    return audit;
}

struct PartitionMaterializationResult {
    long long partition_us{0};
    bag::PartitionRuntimeStats partition_runtime_stats;
    bag::PartitionCheckpointState partition_checkpoint_state;
    std::vector<bag::Subgraph> subgraphs;
    bag::PartitionCacheState cache_state;
};

struct IndexCacheState {
    bool enabled{false};
    bool hit{false};
    bag::PartitionCacheMode mode{bag::PartitionCacheMode::Off};
    std::string key;
    std::filesystem::path path;
    long long load_us{0};
    long long save_us{0};
    long long compute_us{0};
};

struct IndexMaterializationResult {
    long long index_us{0};
    bag::SkeletonIndex index;
    IndexCacheState cache_state;
};

bool deep_audit_partition(
    const bag::Graph& graph,
    const std::vector<bag::Subgraph>& subgraphs,
    PartitionAuditSummary* audit_out = nullptr
) {
    const auto skeleton_graph = build_skeleton_graph_from_subgraphs(subgraphs);
    const auto audit = audit_partition(graph, skeleton_graph, subgraphs);
    if (audit_out != nullptr) {
        *audit_out = audit;
    }
    return partition_audit_ok(audit);
}

PartitionMaterializationResult materialize_partition(
    const std::string& graph_path,
    const bag::Graph& graph,
    const bag::PartitionOptions& partition_options,
    const bag::PartitionCacheConfig& cache_config,
    bool deep_audit
) {
    PartitionMaterializationResult result;
    result.cache_state.enabled = cache_config.mode != bag::PartitionCacheMode::Off;
    result.cache_state.mode = cache_config.mode;
    result.cache_state.deep_audit_requested = deep_audit;

    if (result.cache_state.enabled) {
        result.cache_state.key = bag::make_partition_cache_key(graph_path, graph, partition_options);
        result.cache_state.path = bag::partition_cache_file_path(cache_config, result.cache_state.key);
    }

    const auto compute_partition = [&] {
        result.partition_us = bag::calc_execution_time_in_us([&] {
            bag::VfipPartition partitioner(graph, partition_options);
            result.subgraphs = partitioner.run();
            result.partition_runtime_stats = partitioner.stats();
            result.partition_checkpoint_state = partitioner.checkpoint_state();
        });
        result.cache_state.compute_us = result.partition_us;
        if (deep_audit) {
            result.cache_state.deep_audit_us = bag::calc_execution_time_in_us([&] {
                result.cache_state.deep_audit_ok = deep_audit_partition(graph, result.subgraphs);
            });
            if (!result.cache_state.deep_audit_ok) {
                throw std::runtime_error("computed partition failed deep audit");
            }
        }
    };

    if (result.cache_state.enabled && bag::partition_cache_should_try_read(cache_config.mode)) {
        bag::PartitionCacheEntry entry;
        bool loaded = false;
        try {
            result.cache_state.load_us = bag::calc_execution_time_in_us([&] {
                loaded = bag::load_partition_cache(
                    result.cache_state.path,
                    entry,
                    cache_config.skip_distance_on_read
                );
            });
        } catch (const std::exception&) {
            if (cache_config.mode == bag::PartitionCacheMode::Read) {
                throw;
            }
            loaded = false;
            result.cache_state.load_us = 0;
        }

        if (loaded) {
            if (entry.key != result.cache_state.key) {
                if (cache_config.mode == bag::PartitionCacheMode::Read) {
                    throw std::runtime_error("partition cache key mismatch for " + result.cache_state.path.string());
                }
            } else {
                result.partition_runtime_stats = entry.stats;
                result.subgraphs = std::move(entry.subgraphs);
                result.partition_us = result.cache_state.load_us;
                result.cache_state.hit = true;
                if (deep_audit) {
                    result.cache_state.deep_audit_us = bag::calc_execution_time_in_us([&] {
                        result.cache_state.deep_audit_ok = deep_audit_partition(graph, result.subgraphs);
                    });
                    if (!result.cache_state.deep_audit_ok) {
                        if (cache_config.mode == bag::PartitionCacheMode::Read) {
                            throw std::runtime_error(
                                "partition cache deep audit failed for " + result.cache_state.path.string()
                            );
                        }
                        result.cache_state.hit = false;
                        result.subgraphs.clear();
                        result.partition_runtime_stats = {};
                    }
                }
                if (result.cache_state.hit) {
                    return result;
                }
            }
        } else if (cache_config.mode == bag::PartitionCacheMode::Read) {
            throw std::runtime_error(
                "partition cache miss for required read mode: " + result.cache_state.path.string()
            );
        }
    }

    compute_partition();

    if (result.cache_state.enabled && bag::partition_cache_should_write(cache_config.mode)) {
        result.cache_state.save_us = bag::calc_execution_time_in_us([&] {
            bag::save_partition_cache(
                result.cache_state.path,
                result.cache_state.key,
                result.partition_runtime_stats,
                result.subgraphs
            );
        });
    }

    return result;
}

IndexMaterializationResult materialize_index(
    const bag::Graph& graph,
    const std::vector<bag::Subgraph>& subgraphs,
    const std::string& partition_cache_key,
    bag::PartitionCacheMode cache_mode,
    const std::filesystem::path& cache_dir,
    bool factorized_transfer,
    double factorized_arc_threshold,
    std::size_t factorized_border_threshold
) {
    IndexMaterializationResult result;
    result.cache_state.enabled = cache_mode != bag::PartitionCacheMode::Off;
    result.cache_state.mode = cache_mode;
    if (result.cache_state.enabled) {
        result.cache_state.key = bag::SkeletonIndex::make_cache_key(
            partition_cache_key,
            factorized_transfer,
            factorized_arc_threshold,
            factorized_border_threshold
        );
        result.cache_state.path = cache_dir / (result.cache_state.key + ".bidx");
    }

    const auto compute_index = [&] {
        result.index_us = bag::calc_execution_time_in_us([&] {
            result.index = bag::SkeletonIndex::build(graph, subgraphs);
            if (factorized_transfer) {
                result.index.configure_factorized_transfer(
                    factorized_arc_threshold,
                    factorized_border_threshold
                );
            }
        });
        result.cache_state.compute_us = result.index_us;
    };

    if (result.cache_state.enabled && bag::partition_cache_should_try_read(cache_mode)) {
        bool loaded = false;
        result.cache_state.load_us = bag::calc_execution_time_in_us([&] {
            loaded = bag::SkeletonIndex::load_cache(
                result.cache_state.path,
                result.cache_state.key,
                graph,
                subgraphs,
                result.index
            );
        });
        if (loaded) {
            result.cache_state.hit = true;
            result.index_us = result.cache_state.load_us;
            return result;
        }
    }

    compute_index();

    if (result.cache_state.enabled && bag::partition_cache_should_write(cache_mode) && !result.cache_state.hit) {
        result.cache_state.save_us = bag::calc_execution_time_in_us([&] {
            result.index.save_cache(result.cache_state.path, result.cache_state.key);
        });
    }

    return result;
}

void append_partition_cache_json_fields(
    std::ostream& out,
    const bag::PartitionCacheState& cache_state
) {
    out << ",\"partition_cache_enabled\":" << (cache_state.enabled ? "true" : "false")
        << ",\"partition_cache_mode\":\"" << bag::partition_cache_mode_to_string(cache_state.mode) << "\""
        << ",\"partition_cache_hit\":" << (cache_state.hit ? "true" : "false")
        << ",\"partition_cache_key\":\"" << cache_state.key << "\""
        << ",\"partition_cache_path\":\"" << cache_state.path.generic_string() << "\""
        << ",\"partition_cache_load_us\":" << cache_state.load_us
        << ",\"partition_cache_save_us\":" << cache_state.save_us
        << ",\"partition_cache_compute_us\":" << cache_state.compute_us
        << ",\"partition_cache_deep_audit\":" << (cache_state.deep_audit_requested ? "true" : "false")
        << ",\"partition_cache_deep_audit_us\":" << cache_state.deep_audit_us
        << ",\"partition_cache_deep_audit_ok\":"
        << ((cache_state.deep_audit_requested && cache_state.deep_audit_ok) ? "true" : "false");
}

void append_partition_checkpoint_json_fields(
    std::ostream& out,
    bag::PartitionCheckpointMode mode,
    const std::filesystem::path& path,
    const bag::PartitionCheckpointState& checkpoint_state
) {
    out << ",\"partition_checkpoint_enabled\":" << (checkpoint_state.enabled ? "true" : "false")
        << ",\"partition_checkpoint_mode\":\"" << bag::partition_checkpoint_mode_to_string(mode) << "\""
        << ",\"partition_checkpoint_path\":\"" << path.generic_string() << "\""
        << ",\"partition_checkpoint_resume_requested\":" << (checkpoint_state.resume_requested ? "true" : "false")
        << ",\"partition_checkpoint_resume_hit\":" << (checkpoint_state.resume_hit ? "true" : "false")
        << ",\"partition_checkpoint_resumed_subgraphs\":" << checkpoint_state.resumed_subgraphs
        << ",\"partition_checkpoint_load_us\":" << checkpoint_state.load_us
        << ",\"partition_checkpoint_save_us\":" << checkpoint_state.save_us
        << ",\"partition_checkpoint_save_count\":" << checkpoint_state.save_count
        << ",\"partition_checkpoint_last_saved_subgraphs\":" << checkpoint_state.last_saved_subgraphs;
}

void append_index_cache_json_fields(
    std::ostream& out,
    const IndexCacheState& cache_state
) {
    out << ",\"index_cache_enabled\":" << (cache_state.enabled ? "true" : "false")
        << ",\"index_cache_mode\":\"" << bag::partition_cache_mode_to_string(cache_state.mode) << "\""
        << ",\"index_cache_hit\":" << (cache_state.hit ? "true" : "false")
        << ",\"index_cache_key\":\"" << cache_state.key << "\""
        << ",\"index_cache_path\":\"" << cache_state.path.generic_string() << "\""
        << ",\"index_cache_load_us\":" << cache_state.load_us
        << ",\"index_cache_save_us\":" << cache_state.save_us
        << ",\"index_cache_compute_us\":" << cache_state.compute_us;
}

bag::FcRule parse_fc_rule(const std::unordered_map<std::string, std::string>& args) {
    const auto it = args.find("fc-rule");
    if (it == args.end() || it->second == "paper") {
        return bag::FcRule::PaperStrict;
    }
    if (it->second == "ub") {
        return bag::FcRule::UpperBoundCandidate;
    }
    if (it->second == "all") {
        return bag::FcRule::AllBordersVisited;
    }
    throw std::runtime_error("unsupported --fc-rule, expected paper|ub|all");
}

std::string require_arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        throw std::runtime_error("missing required argument --" + key);
    }
    return it->second;
}

std::size_t require_usize(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key
) {
    return static_cast<std::size_t>(std::stoull(require_arg(args, key)));
}

std::size_t optional_usize(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    std::size_t fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    return static_cast<std::size_t>(std::stoull(it->second));
}

bag::VertexId require_vertex(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key
) {
    return static_cast<bag::VertexId>(std::stoul(require_arg(args, key)));
}

EdgeWeight require_weight(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key
) {
    return static_cast<EdgeWeight>(std::stoul(require_arg(args, key)));
}

bool optional_bool(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    bool fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    return it->second == "1" || it->second == "true";
}

double optional_double(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    double fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    return std::stod(it->second);
}

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  bag_cpp partition --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false]\n"
        << "  bag_cpp range --path <graph> --theta <z> [--partition-seed <v>] --query-u <u> --query-v <v> --offset <d>"
           " --radius <r> --objects <n> [--seed <s>] [--repeat <n>] [--verify true|false] [--fc-rule paper|ub|all]"
           " [--range-row-truncation true|false] [--factorized-transfer true|false] [--factorized-arc-threshold <x>] [--factorized-border-threshold <n>] [--compare-baseline true|false]"
           " [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false]\n"
        << "  bag_cpp knn --path <graph> --theta <z> [--partition-seed <v>] --query-u <u> --query-v <v> --offset <d>"
           " --k <k> --objects <n> [--seed <s>] [--repeat <n>] [--verify true|false]"
          " [--knn-streamed-clique true|false] [--knn-safe-coverage-shadow true|false] [--knn-parent-shadow-size <n>] [--knn-subgraph-admit true|false] [--factorized-transfer true|false] [--factorized-arc-threshold <x>] [--factorized-border-threshold <n>] [--compare-baseline true|false]"
           " [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false]\n"
        << "  bag_cpp update-only-baseline --path <graph> --baseline arne|amovnet|amovnet-query-aware|amovnet-query-indexed"
           " [--coords <co-file>] [--objects <n>] [--change-count <n>] [--epochs <n>]"
           " [--object-seed <n>] [--move-seed <n>] [--grid-side <n>]"
            " [--active-queries <n>] [--query-seed <n>] [--range-radius <r>] [--move-model relocate|local]\n"
        << "  bag_cpp frontier --path <graph> --theta <z> [--partition-seed <v>] --query-u <u> --query-v <v> --offset <d>"
           " [--radii <r1,r2,...>] [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>]"
           " [--border-min true|false] [--shortcut-repartition true|false]\n"
        << "  bag_cpp sample-queries --path <graph> --count <n> [--seed <s>]\n"
        << "  bag_cpp batch --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] --query-count <n> [--query-seed <s>]"
           " --objects <n> [--object-seed <s>] [--range-radius <r>] [--knn-k <k>]"
           " [--fc-rule paper|ub|all] [--factorized-transfer true|false]"
           " [--factorized-arc-threshold <x>] [--factorized-border-threshold <n>]\n"
        << "  bag_cpp knn-batch --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] --query-count <n> [--query-seed <s>]"
           " --objects <n> [--object-seed <s>] [--knn-k <k>] [--knn-subgraph-admit true|false]"
           " [--factorized-transfer true|false] [--factorized-arc-threshold <x>] [--factorized-border-threshold <n>]\n"
        << "  bag_cpp knn-build-diagnostics --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] --query-count <n> [--query-seed <s>]"
           " --objects <n> [--object-seed <s>] [--knn-k <k>] [--knn-subgraph-admit true|false]"
           " [--knn-direct-row-path true|false]"
           " [--factorized-transfer true|false] [--factorized-arc-threshold <x>] [--factorized-border-threshold <n>]\n"
        << "  bag_cpp hierarchy-probe --path <graph> --theta <z1> [--hier-theta <z2>]"
           " [--partition-seed <v>] [--hier-partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] [--hier-force true|false]\n"
        << "  bag_cpp hierarchy-lite --path <graph> --theta <z> [--partition-seed <v>]"
           " [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>]"
           " [--border-min true|false] [--shortcut-repartition true|false]"
           " [--hier-group-sizes <g1,g2,...>]\n"
        << "  bag_cpp factorized-scan --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] [--output-csv <file>]\n"
        << "  bag_cpp diagnostics --path <graph> --theta <z> --output-dir <dir> [--partition-seed <v>]"
           " [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] [--query-count <n>] [--objects <n>] [--query-seed <s>]"
           " [--object-seed <s>] [--range-radius <r>] [--knn-k <k>] [--fc-rule paper|ub|all]"
           " [--exact-demotion-limit <n>] [--growth-trace true|false]\n"
        << "  bag_cpp maint-bench --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] --car-percent <x> --change-percent <x>"
           " --query-per-update <n> [--epochs <n>] [--query-seed <s>] [--object-seed <s>]"
           " [--knn-k <k>] [--range-radius <r>] [--fc-rule paper|ub|all]"
          " [--object-layout random|spread-subgraphs] [--verify-updates true|false] [--knn-subgraph-admit true|false]"
          " [--unified-object-maintenance true|false] [--release-subgraph-distances true|false]"
          " [--mode knn|range]\n"
        << "  bag_cpp memory-report --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] [--objects <n>] [--object-seed <s>]"
           " [--mode knn|range] [--unified-object-maintenance true|false]"
           " [--release-subgraph-distances true|false]\n"
        << "Shared partition cache flags:\n"
        << "  [--partition-cache-mode off|read|write|auto|refresh]"
           " [--partition-cache-dir <dir>] [--partition-cache-deep-audit true|false]\n"
        << "Shared partition checkpoint flags:\n"
        << "  [--partition-checkpoint-mode off|write|resume|auto]"
           " [--partition-checkpoint-dir <dir>] [--partition-checkpoint-every-subgraphs <n>]"
           " [--partition-stop-after-subgraphs <n>]\n";
}

QueryPoint load_query_point(const std::unordered_map<std::string, std::string>& args) {
    return QueryPoint{
        bag::ordered_edge(require_vertex(args, "query-u"), require_vertex(args, "query-v")),
        require_weight(args, "offset"),
    };
}

IndexedMovingObjectSet build_objects(
    const bag::Graph& graph,
    const SkeletonIndex& index,
    std::size_t object_count,
    std::uint64_t seed,
    bool maintain_knn_metadata = true,
    bool maintain_edge_buckets = true
) {
    auto set = MovingObjectSet::random_uniform(graph, object_count, seed);
    return IndexedMovingObjectSet::from_partition(
        set.objects(),
        index.edge_to_subgraph(),
        index.subgraphs(),
        maintain_knn_metadata,
        maintain_edge_buckets
    );
}

IndexedMovingObjectSet build_objects(
    const bag::Graph& graph,
    const SkeletonIndex& index,
    const std::unordered_map<std::string, std::string>& args
) {
    const auto seed_it = args.find("seed");
    const std::uint64_t seed =
        (seed_it == args.end()) ? 7ULL : static_cast<std::uint64_t>(std::stoull(seed_it->second));
    return build_objects(graph, index, require_usize(args, "objects"), seed);
}

std::vector<QueryPoint> build_query_workload(
    const bag::Graph& graph,
    std::size_t query_count,
    std::uint64_t seed
) {
    std::vector<QueryPoint> queries;
    queries.reserve(query_count);
    const auto samples = MovingObjectSet::random_uniform(graph, query_count, seed);
    for (const auto& object : samples.objects()) {
        queries.push_back(QueryPoint{object.edge, object.offset});
    }
    return queries;
}

std::vector<QueryPoint> build_fixed_query_workload(
    const QueryPoint& query,
    std::size_t query_count
) {
    return std::vector<QueryPoint>(query_count, query);
}

std::vector<bag::MovingObject> build_object_vector(
    const bag::Graph& graph,
    const SkeletonIndex& index,
    std::size_t object_count,
    std::uint64_t seed,
    const std::string& layout = "random"
) {
    if (layout == "spread-subgraphs" || layout == "clustered-subgraphs") {
        struct EdgePlacement {
            bag::Edge edge{};
            EdgeWeight edge_weight{0};
            bag::SgId sg_id{0};
        };

        std::vector<EdgePlacement> placements;
        placements.reserve(index.subgraphs().size());
        const auto& edge_to_subgraph = index.edge_to_subgraph();
        for (const auto& sg : index.subgraphs()) {
            std::optional<std::pair<bag::Edge, EdgeWeight>> chosen;
            for (const auto& [edge, weight] : sg.graph.undirected_edges()) {
                if (weight <= 1) {
                    continue;
                }
                const auto ordered = bag::ordered_edge(edge.first, edge.second);
                const auto it = edge_to_subgraph.find(ordered);
                if (it == edge_to_subgraph.end() || it->second != sg.id) {
                    continue;
                }
                const bool internal_edge =
                    sg.internal_vertices.contains(ordered.first) &&
                    sg.internal_vertices.contains(ordered.second);
                if (internal_edge) {
                    chosen = std::make_pair(ordered, weight);
                    break;
                }
                if (!chosen.has_value()) {
                    chosen = std::make_pair(ordered, weight);
                }
            }
            if (chosen.has_value()) {
                placements.push_back(EdgePlacement{chosen->first, chosen->second, sg.id});
            }
        }

        if (placements.empty()) {
            throw std::runtime_error("no valid per-subgraph object placements found");
        }

        std::mt19937_64 rng(seed);
        std::shuffle(placements.begin(), placements.end(), rng);

        if (layout == "clustered-subgraphs") {
            const auto hotspot_count = std::min<std::size_t>(
                placements.size(),
                std::max<std::size_t>(4U, std::min<std::size_t>(64U, placements.size() / 64U))
            );
            placements.resize(hotspot_count);
        }

        std::vector<bag::MovingObject> objects;
        objects.reserve(object_count);
        bag::ObjId next_id = 0;
        std::size_t cursor = 0;
        while (next_id < object_count) {
            if (cursor == placements.size()) {
                std::shuffle(placements.begin(), placements.end(), rng);
                cursor = 0;
            }
            const auto& placement = placements[cursor++];
            std::uniform_int_distribution<EdgeWeight> offset_dist(1, placement.edge_weight - 1);
            objects.push_back(bag::MovingObject{
                next_id++,
                placement.edge,
                offset_dist(rng),
                placement.edge_weight,
            });
        }
        return objects;
    }

    auto set = MovingObjectSet::random_uniform(graph, object_count, seed);
    return set.objects();
}

IndexedMovingObjectSet build_objects_with_layout(
    const bag::Graph& graph,
    const SkeletonIndex& index,
    std::size_t object_count,
    std::uint64_t seed,
    const std::string& layout
) {
    const auto objects = build_object_vector(graph, index, object_count, seed, layout);
    return IndexedMovingObjectSet::from_partition(objects, index.edge_to_subgraph(), index.subgraphs());
}

std::vector<EdgeWeight> parse_weight_list(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::vector<EdgeWeight>& fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    std::vector<EdgeWeight> result;
    std::stringstream ss(it->second);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            result.push_back(static_cast<EdgeWeight>(std::stoul(token)));
        }
    }
    return result.empty() ? fallback : result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage();
            return 1;
        }

        const std::string command = argv[1];
        const auto args = bag::parse_cli_args(argc, argv, 2);
        if (command == "sample-queries") {
            const auto path = require_arg(args, "path");
            const auto graph = bag::load_graph_from_file(path);
            const auto query_count = require_usize(args, "count");
            const auto seed =
                static_cast<std::uint64_t>(args.contains("seed") ? require_usize(args, "seed") : 1ULL);
            const auto queries = build_query_workload(graph, query_count, seed);
            std::cout << "[";
            for (std::size_t i = 0; i < queries.size(); ++i) {
                if (i != 0) {
                    std::cout << ",";
                }
                std::cout << "{"
                          << "\"query_id\":" << i << ","
                          << "\"query_u\":" << queries[i].edge.first << ","
                          << "\"query_v\":" << queries[i].edge.second << ","
                          << "\"offset\":" << queries[i].offset
                          << "}";
            }
            std::cout << "]\n";
            return 0;
        }

        const auto path = require_arg(args, "path");
        const auto graph = bag::load_graph_from_file(path);
        if (command == "update-only-baseline") {
            const auto baseline = require_arg(args, "baseline");
            bag::UpdateOnlyBenchOptions options;
            options.object_count = optional_usize(args, "objects", 10000);
            options.change_count = optional_usize(args, "change-count", 1000);
            options.epochs = optional_usize(args, "epochs", 100);
            options.object_seed = static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            options.move_seed = static_cast<std::uint64_t>(optional_usize(args, "move-seed", 8));
            options.grid_side = optional_usize(args, "grid-side", 256);
            options.active_query_count = optional_usize(args, "active-queries", 1000);
            options.query_seed = static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            options.range_radius =
                static_cast<EdgeWeight>(optional_usize(args, "range-radius", 50000));
            options.local_move = args.contains("move-model") && require_arg(args, "move-model") == "local";

            bag::UpdateOnlyBenchResult result;
            if (baseline == "arne") {
                result = bag::benchmark_arne_update_only(graph, options);
            } else if (baseline == "amovnet") {
                const auto coords_path = require_arg(args, "coords");
                const auto coords = bag::load_coordinates_from_file(coords_path);
                result = bag::benchmark_amovnet_update_only(graph, coords, options);
            } else if (baseline == "amovnet-query-aware") {
                const auto coords_path = require_arg(args, "coords");
                const auto coords = bag::load_coordinates_from_file(coords_path);
                result = bag::benchmark_amovnet_query_aware_update_only(graph, coords, options);
            } else if (baseline == "amovnet-query-indexed") {
                const auto coords_path = require_arg(args, "coords");
                const auto coords = bag::load_coordinates_from_file(coords_path);
                result = bag::benchmark_amovnet_query_indexed_update_only(graph, coords, options);
            } else {
                throw std::runtime_error("unsupported --baseline, expected arne|amovnet|amovnet-query-aware|amovnet-query-indexed");
            }

            std::cout
                << "{"
                << "\"baseline\":\"" << result.baseline << "\","
                << "\"object_count\":" << result.object_count << ","
                << "\"change_count\":" << result.change_count << ","
                << "\"epochs\":" << result.epochs << ","
                << "\"grid_side\":" << result.grid_side << ","
                << "\"active_query_count\":" << result.active_query_count << ","
                << "\"range_radius\":" << result.range_radius << ","
                << "\"total_update_us\":" << result.total_update_us << ","
                << "\"avg_update_us\":" << result.avg_update_us
                << "}\n";
            return 0;
        }
        const bool diagnostics_mode = command == "diagnostics";
        const auto diagnostics_output_dir =
            diagnostics_mode ? std::filesystem::path(require_arg(args, "output-dir")) : std::filesystem::path{};
        const auto theta = require_usize(args, "theta");
        const auto partition_seed =
            static_cast<bag::VertexId>(args.contains("partition-seed") ? require_usize(args, "partition-seed") : 1U);
        const bool paper_strict = optional_bool(args, "paper-strict", false);
        const bool adaptive_z = optional_bool(args, "adaptive-z", false);
        const double alpha = optional_double(args, "alpha", 1.0);
        const bool border_min = optional_bool(args, "border-min", false);
        const bool shortcut_repartition = optional_bool(args, "shortcut-repartition", true);
        const auto shortcut_small_upper_bound =
            optional_usize(args, "shortcut-small-upper-bound", 3);
        const auto shortcut_k_neighbors =
            optional_usize(args, "shortcut-k-neighbors", 200);
        const auto shortcut_radius_limit =
            static_cast<EdgeWeight>(optional_usize(args, "shortcut-radius-limit", 6000));
        const auto shortcut_max_tiny_subgraphs =
            optional_usize(args, "shortcut-max-tiny-subgraphs", 100000);
        const bool phase1_inplace = optional_bool(args, "phase1-inplace", false);
        const bool defer_nearest_border_fill =
            optional_bool(args, "defer-nearest-border-fill", false);
        const bool skip_phase1_br_check =
            optional_bool(args, "skip-phase1-br-check", false);
        const bool phase1_local_audit =
            optional_bool(args, "phase1-local-audit", false);
        const bool phase2_incremental_distance_update =
            optional_bool(args, "phase2-incremental-distance-update", false);
        const bool progress_log = optional_bool(args, "progress-log", false);
        const auto progress_log_every_subgraphs =
            optional_usize(args, "progress-log-every-subgraphs", 5000);
        const auto partition_cache_mode =
            bag::parse_partition_cache_mode(args.contains("partition-cache-mode")
                                                ? require_arg(args, "partition-cache-mode")
                                                : "off");
        const auto partition_cache_dir =
            args.contains("partition-cache-dir")
                ? std::filesystem::path(require_arg(args, "partition-cache-dir"))
                : (std::filesystem::path("cache") / "bag_partition");
        const bool partition_cache_deep_audit =
            optional_bool(args, "partition-cache-deep-audit", false);
        const bool partition_cache_skip_distance =
            optional_bool(args, "partition-cache-skip-distance", false);
        const auto index_cache_mode =
            bag::parse_partition_cache_mode(args.contains("index-cache-mode")
                                                ? require_arg(args, "index-cache-mode")
                                                : "off");
        const auto index_cache_dir =
            args.contains("index-cache-dir")
                ? std::filesystem::path(require_arg(args, "index-cache-dir"))
                : (std::filesystem::path("cache") / "bag_index");
        const auto partition_checkpoint_mode =
            bag::parse_partition_checkpoint_mode(args.contains("partition-checkpoint-mode")
                                                     ? require_arg(args, "partition-checkpoint-mode")
                                                     : "off");
        const auto partition_checkpoint_dir =
            args.contains("partition-checkpoint-dir")
                ? std::filesystem::path(require_arg(args, "partition-checkpoint-dir"))
                : (std::filesystem::path("cache") / "bag_partition_ckpt");
        const auto partition_checkpoint_every_subgraphs =
            optional_usize(args, "partition-checkpoint-every-subgraphs", 5000);
        const auto partition_stop_after_subgraphs =
            optional_usize(args, "partition-stop-after-subgraphs", 0);

        bag::PartitionOptions partition_options;
        partition_options.theta = theta;
        partition_options.partition_seed = partition_seed;
        partition_options.paper_strict_mode = paper_strict;
        partition_options.adaptive_z = adaptive_z;
        partition_options.adaptive_alpha = alpha;
        partition_options.border_minimization = border_min;
        partition_options.shortcut_repartition = shortcut_repartition;
        partition_options.shortcut_small_upper_bound = shortcut_small_upper_bound;
        partition_options.shortcut_k_neighbors = shortcut_k_neighbors;
        partition_options.shortcut_radius_limit = shortcut_radius_limit;
        partition_options.shortcut_max_tiny_subgraphs = shortcut_max_tiny_subgraphs;
        partition_options.phase1_inplace = phase1_inplace;
        partition_options.defer_nearest_border_fill = defer_nearest_border_fill;
        partition_options.skip_phase1_br_check = skip_phase1_br_check;
        partition_options.phase1_local_audit = phase1_local_audit;
        partition_options.phase2_incremental_distance_update = phase2_incremental_distance_update;
        partition_options.progress_log = progress_log;
        partition_options.progress_log_every_subgraphs = progress_log_every_subgraphs;
        if (diagnostics_mode && optional_bool(args, "growth-trace", true)) {
            partition_options.growth_trace_output = diagnostics_output_dir / "vip_growth_trace.jsonl";
        }
        const auto partition_identity_key = bag::make_partition_cache_key(path, graph, partition_options);
        partition_options.checkpoint_key = partition_identity_key;
        partition_options.checkpoint_path = partition_checkpoint_dir / (partition_identity_key + ".bpckpt");
        partition_options.checkpoint_resume =
            partition_checkpoint_mode == bag::PartitionCheckpointMode::Resume ||
            partition_checkpoint_mode == bag::PartitionCheckpointMode::Auto;
        partition_options.checkpoint_resume_required =
            partition_checkpoint_mode == bag::PartitionCheckpointMode::Resume;
        partition_options.checkpoint_write =
            partition_checkpoint_mode == bag::PartitionCheckpointMode::Write ||
            partition_checkpoint_mode == bag::PartitionCheckpointMode::Auto;
        partition_options.checkpoint_every_subgraphs = partition_checkpoint_every_subgraphs;
        partition_options.checkpoint_stop_after_subgraphs = partition_stop_after_subgraphs;

        const bag::PartitionCacheConfig partition_cache_config{
            .mode = partition_cache_mode,
            .directory = partition_cache_dir,
            .skip_distance_on_read = partition_cache_skip_distance,
        };
        const auto partition_result = materialize_partition(
            path,
            graph,
            partition_options,
            partition_cache_config,
            partition_cache_deep_audit
        );
        const auto partition_us = partition_result.partition_us;
        const auto& partition_runtime_stats = partition_result.partition_runtime_stats;
        const auto& partition_checkpoint_state = partition_result.partition_checkpoint_state;
        const auto& partition_cache_state = partition_result.cache_state;
        const auto& subgraphs = partition_result.subgraphs;

        if (command == "partition" || command == "partition-audit") {
            const bool partition_audit_mode = command == "partition-audit";
            std::optional<PartitionAuditSummary> partition_audit_summary;
            if (partition_audit_mode) {
                const auto skeleton_graph = build_skeleton_graph_from_subgraphs(subgraphs);
                partition_audit_summary = audit_partition(graph, skeleton_graph, subgraphs);
            }
            std::unordered_set<bag::VertexId> unique_boundary_vertices;
            std::unordered_set<std::uint64_t> unique_skeleton_edges;
            std::map<std::size_t, std::size_t> subgraph_size_histogram;
            std::size_t total_subgraph_vertices = 0;
            std::size_t total_boundary_vertices = 0;
            std::size_t total_internal_vertices = 0;
            std::size_t max_boundary_vertices = 0;
            std::size_t br_violating_subgraphs = 0;
            std::size_t br_violating_boundaries = 0;
            for (const auto& sg : subgraphs) {
                ++subgraph_size_histogram[sg.graph.size()];
                total_subgraph_vertices += sg.graph.size();
                total_boundary_vertices += sg.bound_vertices.size();
                total_internal_vertices += sg.internal_vertices.size();
                max_boundary_vertices = std::max(max_boundary_vertices, sg.bound_vertices.size());
                unique_boundary_vertices.insert(sg.bound_vertices.begin(), sg.bound_vertices.end());

                bool subgraph_ok = true;
                const auto sg_edges = sg.graph.undirected_edges();
                for (const auto b : sg.bound_vertices) {
                    bag::HalfWeight rb{0, false};
                    for (const auto other : sg.bound_vertices) {
                        rb = std::max(rb, bag::HalfWeight{sg.distance.get_or_inf(b, other), false});
                    }

                    bag::HalfWeight ib{0, false};
                    for (const auto v : sg.internal_vertices) {
                        ib = std::max(ib, bag::HalfWeight{sg.distance.get_or_inf(b, v), false});
                    }
                    for (const auto& [edge, w] : sg_edges) {
                        const auto lhs = sg.distance.get_or_inf(b, edge.first);
                        const auto rhs = sg.distance.get_or_inf(b, edge.second);
                        if (lhs == bag::kInfWeight || rhs == bag::kInfWeight) {
                            continue;
                        }
                        ib = std::max(ib, bag::distal_point_distance(lhs, rhs, w));
                    }

                    if (rb < ib) {
                        ++br_violating_boundaries;
                        subgraph_ok = false;
                    }
                }
                if (!subgraph_ok) {
                    ++br_violating_subgraphs;
                }

                std::vector<bag::VertexId> borders(sg.bound_vertices.begin(), sg.bound_vertices.end());
                for (std::size_t i = 0; i < borders.size(); ++i) {
                    for (std::size_t j = i + 1; j < borders.size(); ++j) {
                        if (sg.distance.get_or_inf(borders[i], borders[j]) != bag::kInfWeight) {
                            const auto edge = bag::ordered_edge(borders[i], borders[j]);
                            unique_skeleton_edges.insert(bag::pack_pair(edge.first, edge.second));
                        }
                    }
                }
            }
            const auto skeleton_vertices = unique_boundary_vertices.size();
            const auto skeleton_edges = unique_skeleton_edges.size();
            const auto avg_boundary_per_subgraph =
                subgraphs.empty() ? 0.0 : static_cast<double>(total_boundary_vertices) / static_cast<double>(subgraphs.size());
            const auto vertex_duplication_factor =
                graph.size() == 0 ? 0.0 : static_cast<double>(total_subgraph_vertices) / static_cast<double>(graph.size());
            const auto unique_boundary_ratio =
                graph.size() == 0 ? 0.0 : static_cast<double>(skeleton_vertices) / static_cast<double>(graph.size());
            const auto skeleton_avg_degree =
                skeleton_vertices == 0 ? 0.0 : (2.0 * static_cast<double>(skeleton_edges)) / static_cast<double>(skeleton_vertices);
            std::cout
                << "{"
                << "\"subgraphs\":" << subgraphs.size() << ","
                << "\"graph_vertices\":" << graph.size() << ","
                << "\"graph_edges\":" << graph.edge_count() << ","
                << "\"skeleton_vertices\":" << skeleton_vertices << ","
                << "\"unique_boundary_vertices\":" << skeleton_vertices << ","
                << "\"skeleton_edges\":" << skeleton_edges << ","
                << "\"skeleton_avg_degree\":" << skeleton_avg_degree << ","
                << "\"total_subgraph_vertices\":" << total_subgraph_vertices << ","
                << "\"total_boundary_vertices\":" << total_boundary_vertices << ","
                << "\"total_internal_vertices\":" << total_internal_vertices << ","
                << "\"avg_boundary_per_subgraph\":" << avg_boundary_per_subgraph << ","
                << "\"max_boundary_vertices\":" << max_boundary_vertices << ","
                << "\"br_property_ok\":" << (br_violating_boundaries == 0 ? "true" : "false") << ","
                << "\"br_violating_subgraphs\":" << br_violating_subgraphs << ","
                << "\"br_violating_boundaries\":" << br_violating_boundaries << ","
                << "\"vertex_duplication_factor\":" << vertex_duplication_factor << ","
                << "\"unique_boundary_ratio\":" << unique_boundary_ratio << ","
                << "\"subgraph_size_histogram\":{";
            bool first_hist = true;
            for (const auto& [size, count] : subgraph_size_histogram) {
                if (!first_hist) {
                    std::cout << ",";
                }
                first_hist = false;
                std::cout << "\"" << size << "\":" << count;
            }
            std::cout
                << "},"
                << "\"partition_seed\":" << partition_seed << ","
                << "\"adaptive_z\":" << (adaptive_z ? "true" : "false") << ","
                << "\"alpha\":" << alpha << ","
                << "\"border_min\":" << (border_min ? "true" : "false") << ","
                << "\"shortcut_repartition\":" << (shortcut_repartition ? "true" : "false") << ","
                << "\"phase1_inplace\":" << (phase1_inplace ? "true" : "false") << ","
                << "\"defer_nearest_border_fill\":" << (defer_nearest_border_fill ? "true" : "false") << ","
                << "\"skip_phase1_br_check\":" << (skip_phase1_br_check ? "true" : "false") << ","
                << "\"phase1_local_audit\":" << (phase1_local_audit ? "true" : "false") << ","
                << "\"phase2_incremental_distance_update\":" << (phase2_incremental_distance_update ? "true" : "false") << ","
                << "\"partition_core_us\":" << partition_runtime_stats.core_partition_us << ","
                << "\"shortcut_repartition_us\":" << partition_runtime_stats.shortcut_repartition_us << ","
                << "\"subgraphs_before_shortcut\":" << partition_runtime_stats.subgraphs_before_shortcut << ","
                << "\"subgraphs_after_shortcut\":" << partition_runtime_stats.subgraphs_after_shortcut << ","
                << "\"phase1_attempts\":" << partition_runtime_stats.phase1_attempts << ","
                << "\"phase1_commits\":" << partition_runtime_stats.phase1_commits << ","
                << "\"phase1_rejects\":" << partition_runtime_stats.phase1_rejects << ","
                << "\"phase1_clone_us\":" << partition_runtime_stats.phase1_clone_us << ","
                << "\"phase1_finalize_us\":" << partition_runtime_stats.phase1_finalize_us << ","
                << "\"phase1_refresh_us\":" << partition_runtime_stats.phase1_refresh_us << ","
                << "\"phase1_extend_distance_us\":" << partition_runtime_stats.phase1_extend_distance_us << ","
                << "\"phase1_br_us\":" << partition_runtime_stats.phase1_br_us << ","
                << "\"phase1_nearest_border_us\":" << partition_runtime_stats.phase1_nearest_border_us << ","
                << "\"phase2_attempts\":" << partition_runtime_stats.phase2_attempts << ","
                << "\"phase2_commits\":" << partition_runtime_stats.phase2_commits << ","
                << "\"phase2_rejects\":" << partition_runtime_stats.phase2_rejects << ","
                << "\"phase2_clone_us\":" << partition_runtime_stats.phase2_clone_us << ","
                << "\"phase2_finalize_us\":" << partition_runtime_stats.phase2_finalize_us << ","
                << "\"phase2_refresh_us\":" << partition_runtime_stats.phase2_refresh_us << ","
                << "\"phase2_apsp_us\":" << partition_runtime_stats.phase2_apsp_us << ","
                << "\"phase2_distance_update_us\":" << partition_runtime_stats.phase2_distance_update_us << ","
                << "\"phase2_br_us\":" << partition_runtime_stats.phase2_br_us << ","
                << "\"phase2_nearest_border_us\":" << partition_runtime_stats.phase2_nearest_border_us << ","
                << "\"shortcut_tiny_subgraphs\":" << partition_runtime_stats.shortcut_stats.tiny_subgraphs << ","
                << "\"shortcut_merged_components\":" << partition_runtime_stats.shortcut_stats.merged_components << ","
                << "\"shortcut_merged_output_subgraphs\":" << partition_runtime_stats.shortcut_stats.merged_output_subgraphs << ","
                << "\"shortcut_skipped_by_tiny_budget\":" << (partition_runtime_stats.shortcut_stats.skipped_by_tiny_budget ? "true" : "false") << ","
                << "\"partition_us\":" << partition_us;
            append_partition_checkpoint_json_fields(
                std::cout,
                partition_checkpoint_mode,
                partition_options.checkpoint_path,
                partition_checkpoint_state
            );
            append_partition_cache_json_fields(std::cout, partition_cache_state);
            if (partition_audit_summary.has_value()) {
                std::cout
                    << ",\"covered_graph_edges\":" << partition_audit_summary->covered_graph_edges
                    << ",\"uncovered_graph_edges\":" << partition_audit_summary->uncovered_graph_edges
                    << ",\"duplicated_graph_edges\":" << partition_audit_summary->duplicated_graph_edges
                    << ",\"seed_missing_subgraphs\":" << partition_audit_summary->seed_missing_subgraphs
                    << ",\"seed_not_boundary_subgraphs\":" << partition_audit_summary->seed_not_boundary_subgraphs
                    << ",\"membership_overlap_vertices\":" << partition_audit_summary->membership_overlap_vertices
                    << ",\"membership_unassigned_vertices\":" << partition_audit_summary->membership_unassigned_vertices
                    << ",\"nearest_border_missing_entries\":" << partition_audit_summary->nearest_border_missing_entries
                    << ",\"nearest_border_mismatch_entries\":" << partition_audit_summary->nearest_border_mismatch_entries
                    << ",\"rb_map_missing_entries\":" << partition_audit_summary->rb_map_missing_entries
                    << ",\"rb_map_mismatch_entries\":" << partition_audit_summary->rb_map_mismatch_entries;
            }
            std::cout
                << "}\n";
            return 0;
        }

        if (command == "factorized-scan") {
            const auto stats = bag::scan_factorized_transfer(subgraphs);
            const auto summary = bag::summarize_factorized_transfer(stats);
            if (args.contains("output-csv")) {
                bag::write_factorized_transfer_csv(require_arg(args, "output-csv"), stats);
            }
            std::cout
                << "{"
                << "\"subgraphs\":" << summary.subgraph_count << ","
                << "\"scanned_subgraphs\":" << summary.scanned_subgraphs << ","
                << "\"compressible_subgraphs\":" << summary.compressible_subgraphs << ","
                << "\"strong_compressible_subgraphs\":" << summary.strong_compressible_subgraphs << ","
                << "\"explicit_directed_arcs\":" << summary.explicit_directed_arcs << ","
                << "\"factorized_directed_arcs\":" << summary.factorized_directed_arcs << ","
                << "\"overall_arc_ratio\":" << summary.overall_arc_ratio << ","
                << "\"avg_hubs_per_subgraph\":" << summary.avg_hubs_per_subgraph << ","
                << "\"avg_arc_ratio\":" << summary.avg_arc_ratio
                << "}\n";
            return 0;
        }

        if (command == "hierarchy-probe") {
            const auto hier_theta = optional_usize(args, "hier-theta", theta);
            const auto hier_partition_seed =
                static_cast<bag::VertexId>(optional_usize(args, "hier-partition-seed", partition_seed));
            const bool hier_force = optional_bool(args, "hier-force", false);

            const auto level1_graph = build_skeleton_graph_from_subgraphs(subgraphs);
            const auto level1_summary = summarize_partition(graph, level1_graph, subgraphs);

            bag::PartitionOptions level2_options = partition_options;
            level2_options.theta = hier_theta;
            level2_options.partition_seed = hier_partition_seed;
            level2_options.growth_trace_output.clear();

            constexpr std::size_t kSafeHierarchyVertexLimit = 50000;
            constexpr std::size_t kSafeHierarchyEdgeLimit = 500000;
            if (!hier_force &&
                (level1_graph.size() > kSafeHierarchyVertexLimit ||
                 level1_graph.edge_count() > kSafeHierarchyEdgeLimit)) {
                std::cout
                    << "{"
                    << "\"skipped\":true,"
                    << "\"reason\":\"level-1 skeleton exceeds safe recursive VFIP limits\","
                    << "\"level1_skeleton_vertices\":" << level1_graph.size() << ","
                    << "\"level1_skeleton_edges\":" << level1_graph.edge_count() << ","
                    << "\"safe_vertex_limit\":" << kSafeHierarchyVertexLimit << ","
                    << "\"safe_edge_limit\":" << kSafeHierarchyEdgeLimit
                    << "}\n";
                return 0;
            }
            long long level2_partition_us = 0;
            std::vector<bag::Subgraph> level2_subgraphs;
            level2_partition_us = bag::calc_execution_time_in_us([&] {
                level2_subgraphs = VfipPartition(level1_graph, level2_options).run();
            });
            const auto level2_graph = build_skeleton_graph_from_subgraphs(level2_subgraphs);
            const auto level2_summary = summarize_partition(level1_graph, level2_graph, level2_subgraphs);

            const auto vertex_ratio =
                level1_summary.skeleton_vertices == 0
                    ? 0.0
                    : static_cast<double>(level2_summary.skeleton_vertices) /
                          static_cast<double>(level1_summary.skeleton_vertices);
            const auto edge_ratio =
                level1_summary.skeleton_edges == 0
                    ? 0.0
                    : static_cast<double>(level2_summary.skeleton_edges) /
                          static_cast<double>(level1_summary.skeleton_edges);

            std::cout
                << "{"
                << "\"level1_theta\":" << theta << ","
                << "\"level2_theta\":" << hier_theta << ","
                << "\"level1_partition_us\":" << partition_us << ","
                << "\"level2_partition_us\":" << level2_partition_us << ","
                << "\"level1_subgraphs\":" << level1_summary.subgraphs << ","
                << "\"level1_skeleton_vertices\":" << level1_summary.skeleton_vertices << ","
                << "\"level1_skeleton_edges\":" << level1_summary.skeleton_edges << ","
                << "\"level1_avg_boundary_per_subgraph\":" << level1_summary.avg_boundary_per_subgraph << ","
                << "\"level1_skeleton_avg_degree\":" << level1_summary.skeleton_avg_degree << ","
                << "\"level1_br_property_ok\":" << (level1_summary.br_property_ok ? "true" : "false") << ","
                << "\"level2_graph_vertices\":" << level1_graph.size() << ","
                << "\"level2_graph_edges\":" << level1_graph.edge_count() << ","
                << "\"level2_subgraphs\":" << level2_summary.subgraphs << ","
                << "\"level2_skeleton_vertices\":" << level2_summary.skeleton_vertices << ","
                << "\"level2_skeleton_edges\":" << level2_summary.skeleton_edges << ","
                << "\"level2_avg_boundary_per_subgraph\":" << level2_summary.avg_boundary_per_subgraph << ","
                << "\"level2_skeleton_avg_degree\":" << level2_summary.skeleton_avg_degree << ","
                << "\"level2_br_property_ok\":" << (level2_summary.br_property_ok ? "true" : "false") << ","
                << "\"skeleton_vertex_ratio_l2_over_l1\":" << vertex_ratio << ","
                << "\"skeleton_edge_ratio_l2_over_l1\":" << edge_ratio;
            append_partition_checkpoint_json_fields(
                std::cout,
                partition_checkpoint_mode,
                partition_options.checkpoint_path,
                partition_checkpoint_state
            );
            append_partition_cache_json_fields(std::cout, partition_cache_state);
            std::cout << "}\n";
            return 0;
        }

        if (command == "hierarchy-lite") {
            const auto level1_graph = build_skeleton_graph_from_subgraphs(subgraphs);
            const auto level1_summary = summarize_partition(graph, level1_graph, subgraphs);
            const auto group_sizes = parse_size_list(args, "hier-group-sizes", {4, 8, 16, 32});

            std::cout
                << "{"
                << "\"level1_theta\":" << theta << ","
                << "\"level1_partition_us\":" << partition_us << ","
                << "\"level1_subgraphs\":" << level1_summary.subgraphs << ","
                << "\"level1_skeleton_vertices\":" << level1_summary.skeleton_vertices << ","
                << "\"level1_skeleton_edges\":" << level1_summary.skeleton_edges << ","
                << "\"level1_total_boundary_vertices\":" << level1_summary.total_boundary_vertices << ","
                << "\"level1_avg_boundary_per_subgraph\":" << level1_summary.avg_boundary_per_subgraph << ","
                << "\"group_summaries\":[";

            bool first = true;
            for (const auto group_size : group_sizes) {
                const auto summary = analyze_hierarchy_lite(subgraphs, level1_summary, group_size);
                if (!first) {
                    std::cout << ",";
                }
                first = false;
                std::cout
                    << "{"
                    << "\"group_size\":" << summary.group_size << ","
                    << "\"parent_regions\":" << summary.parent_regions << ","
                    << "\"avg_children_per_parent\":" << summary.avg_children_per_parent << ","
                    << "\"max_children_per_parent\":" << summary.max_children_per_parent << ","
                    << "\"upper_unique_boundary_vertices\":" << summary.upper_unique_boundary_vertices << ","
                    << "\"upper_total_boundary_vertices\":" << summary.upper_total_boundary_vertices << ","
                    << "\"upper_parent_graph_edges\":" << summary.upper_parent_graph_edges << ","
                    << "\"upper_estimated_clique_edges\":" << summary.upper_estimated_clique_edges << ","
                    << "\"upper_unique_boundary_ratio_vs_l1\":" << summary.upper_unique_boundary_ratio_vs_l1 << ","
                    << "\"upper_total_boundary_ratio_vs_l1_total\":" << summary.upper_total_boundary_ratio_vs_l1_total << ","
                    << "\"upper_estimated_clique_ratio_vs_l1\":" << summary.upper_estimated_clique_ratio_vs_l1
                    << "}";
            }
            std::cout << "]";
            append_partition_checkpoint_json_fields(
                std::cout,
                partition_checkpoint_mode,
                partition_options.checkpoint_path,
                partition_checkpoint_state
            );
            append_partition_cache_json_fields(std::cout, partition_cache_state);
            std::cout << "}\n";
            return 0;
        }

        if (command == "batch") {
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_count = require_usize(args, "objects");
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto range_radius =
                static_cast<EdgeWeight>(optional_usize(args, "range-radius", 50000));
            const auto knn_k = optional_usize(args, "knn-k", 10);
            const auto fc_rule = parse_fc_rule(args);
            const bool knn_subgraph_admit = optional_bool(args, "knn-subgraph-admit", false);
            const bool range_whole_subgraph_acceptance =
                optional_bool(args, "range-whole-subgraph-acceptance", true);

            bag::SkeletonIndex index = SkeletonIndex::build(graph, subgraphs);
            const bool factorized_transfer = optional_bool(args, "factorized-transfer", true);
            const double factorized_arc_threshold = optional_double(args, "factorized-arc-threshold", 0.5);
            const auto factorized_border_threshold = optional_usize(args, "factorized-border-threshold", 12);
            if (factorized_transfer) {
                index.configure_factorized_transfer(factorized_arc_threshold, factorized_border_threshold);
            }
            const auto objects = build_objects(graph, index, object_count, object_seed);
            const auto queries = build_query_workload(graph, query_count, query_seed);

            std::vector<long long> range_times;
            std::vector<long long> knn_times;
            range_times.reserve(query_count);
            knn_times.reserve(query_count);
            for (std::size_t i = 0; i < queries.size(); ++i) {
                range_times.push_back(bag::calc_execution_time_in_us([&] {
                    (void)index.range_query(
                        queries[i],
                        range_radius,
                        objects,
                        fc_rule,
                        i,
                        nullptr,
                        nullptr,
                        true,
                        factorized_transfer,
                        range_whole_subgraph_acceptance
                    );
                }));
                knn_times.push_back(bag::calc_execution_time_in_us([&] {
                    (void)index.knn_query(
                        queries[i],
                        knn_k,
                        objects,
                        i,
                        nullptr,
                        nullptr,
                        false,
                        false,
                        0,
                        factorized_transfer,
                        false,
                        knn_subgraph_admit
                    );
                }));
            }

            const auto range_total = std::accumulate(range_times.begin(), range_times.end(), 0LL);
            const auto knn_total = std::accumulate(knn_times.begin(), knn_times.end(), 0LL);
            const auto range_min = range_times.empty() ? 0LL : *std::min_element(range_times.begin(), range_times.end());
            const auto range_max = range_times.empty() ? 0LL : *std::max_element(range_times.begin(), range_times.end());
            const auto knn_min = knn_times.empty() ? 0LL : *std::min_element(knn_times.begin(), knn_times.end());
            const auto knn_max = knn_times.empty() ? 0LL : *std::max_element(knn_times.begin(), knn_times.end());

            std::cout
                << "{"
                << "\"query_count\":" << query_count << ","
                << "\"query_seed\":" << query_seed << ","
                << "\"objects\":" << object_count << ","
                << "\"object_seed\":" << object_seed << ","
                << "\"range_radius\":" << range_radius << ","
                << "\"knn_k\":" << knn_k << ","
                << "\"range_avg_us\":" << (queries.empty() ? 0.0 : static_cast<double>(range_total) / static_cast<double>(queries.size())) << ","
                << "\"range_min_us\":" << range_min << ","
                << "\"range_max_us\":" << range_max << ","
                << "\"knn_avg_us\":" << (queries.empty() ? 0.0 : static_cast<double>(knn_total) / static_cast<double>(queries.size())) << ","
                << "\"knn_min_us\":" << knn_min << ","
                << "\"knn_max_us\":" << knn_max << ","
                << "\"knn_subgraph_admit\":" << (knn_subgraph_admit ? "true" : "false")
                << "}\n";
            return 0;
        }

        if (command == "query-workload") {
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto queries = build_query_workload(graph, query_count, query_seed);
            std::cout << "{"
                      << "\"query_count\":" << queries.size() << ","
                      << "\"query_seed\":" << query_seed << ","
                      << "\"queries\":[";
            for (std::size_t i = 0; i < queries.size(); ++i) {
                const auto& q = queries[i];
                const auto weight = graph.get_weight(q.edge.first, q.edge.second).value_or(0);
                if (i != 0) {
                    std::cout << ",";
                }
                std::cout << "{"
                          << "\"query_id\":" << i << ","
                          << "\"u\":" << q.edge.first << ","
                          << "\"v\":" << q.edge.second << ","
                          << "\"offset\":" << q.offset << ","
                          << "\"edge_weight\":" << weight
                          << "}";
            }
            std::cout << "]}\n";
            return 0;
        }

        if (command == "knn-batch") {
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_count = require_usize(args, "objects");
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto knn_k = optional_usize(args, "knn-k", 10);
            const bool knn_subgraph_admit = optional_bool(args, "knn-subgraph-admit", false);
            const bool knn_admit_exact_cap = optional_bool(args, "knn-admit-exact-cap", false);
            const bool knn_admit_density_fallback = optional_bool(args, "knn-admit-density-fallback", false);
            const auto layout_it = args.find("object-layout");
            const std::string object_layout = (layout_it == args.end()) ? "random" : layout_it->second;

            bag::SkeletonIndex index = SkeletonIndex::build(graph, subgraphs);
            const bool factorized_transfer = optional_bool(args, "factorized-transfer", true);
            const double factorized_arc_threshold = optional_double(args, "factorized-arc-threshold", 0.5);
            const auto factorized_border_threshold = optional_usize(args, "factorized-border-threshold", 12);
            if (factorized_transfer) {
                index.configure_factorized_transfer(factorized_arc_threshold, factorized_border_threshold);
            }
            const auto objects = build_objects_with_layout(graph, index, object_count, object_seed, object_layout);
            const auto queries = build_query_workload(graph, query_count, query_seed);

            std::vector<long long> knn_times;
            knn_times.reserve(query_count);
            std::uint64_t total_visited_subgraphs = 0;
            std::uint64_t total_candidates_considered = 0;
            std::uint64_t total_final_candidates = 0;
            std::uint64_t total_exact_evaluated = 0;
            std::uint64_t total_exact_from_admitted = 0;
            std::uint64_t total_exact_from_unresolved = 0;
            std::uint64_t total_exact_reused_from_admitted_cache = 0;
            std::uint64_t total_admitted_subgraphs = 0;
            std::uint64_t total_admitted_objects = 0;
            std::uint64_t total_candidate_subgraphs = 0;
            std::uint64_t total_admission_refresh_calls = 0;
            std::uint64_t total_admission_refresh_successes = 0;
            std::uint64_t total_admission_ud_updates = 0;
            std::uint64_t total_admission_gate_heap_fail = 0;
            std::uint64_t total_admission_gate_mass_fail = 0;
            std::uint64_t total_admission_gate_avg_fail = 0;
            std::uint64_t total_admission_gate_dt_fail = 0;
            std::uint64_t total_admission_gate_no_admit_fail = 0;
            std::uint64_t total_admission_gate_object_fail = 0;
            std::uint64_t total_finite_ud_candidate_subgraphs = 0;
            long long total_best_admission_margin = 0;
            std::uint64_t total_explore_us = 0;
            std::uint64_t total_finalize_us = 0;
            std::uint64_t total_subgraph_admit_auto_disabled = 0;
            double total_object_density = 0.0;
            std::uint64_t total_occupied_subgraphs = 0;
            double total_occupied_subgraph_ratio = 0.0;
            double total_avg_objects_per_occupied_subgraph = 0.0;
            std::uint64_t total_max_objects_in_subgraph = 0;
            std::uint64_t finite_safe_coverage_count = 0;
            std::uint64_t total_final_safe_coverage_radius = 0;
            std::uint64_t finite_tau_count = 0;
            std::uint64_t total_first_finite_tau = 0;
            std::uint64_t total_first_finite_tau_boundary_visit_order = 0;

            for (std::size_t i = 0; i < queries.size(); ++i) {
                bag::KnnQueryResult result;
                knn_times.push_back(bag::calc_execution_time_in_us([&] {
                    result = index.knn_query(
                        queries[i],
                        knn_k,
                        objects,
                        i,
                        nullptr,
                        nullptr,
                        false,
                        false,
                        0,
                        factorized_transfer,
                        false,
                        knn_subgraph_admit,
                        knn_admit_exact_cap,
                        knn_admit_density_fallback
                    );
                }));
                total_visited_subgraphs += result.visited_subgraphs;
                total_candidates_considered += result.candidates_considered;
                total_final_candidates += result.final_candidates;
                total_exact_evaluated += result.exact_evaluated;
                total_exact_from_admitted += result.exact_from_admitted;
                total_exact_from_unresolved += result.exact_from_unresolved;
                total_exact_reused_from_admitted_cache += result.exact_reused_from_admitted_cache;
                total_admitted_subgraphs += result.admitted_subgraphs;
                total_admitted_objects += result.admitted_objects;
                total_candidate_subgraphs += result.candidate_subgraphs;
                total_admission_refresh_calls += result.admission_refresh_calls;
                total_admission_refresh_successes += result.admission_refresh_successes;
                total_admission_ud_updates += result.admission_ud_updates;
                total_admission_gate_heap_fail += result.admission_gate_heap_fail;
                total_admission_gate_mass_fail += result.admission_gate_mass_fail;
                total_admission_gate_avg_fail += result.admission_gate_avg_fail;
                total_admission_gate_dt_fail += result.admission_gate_dt_fail;
                total_admission_gate_no_admit_fail += result.admission_gate_no_admit_fail;
                total_admission_gate_object_fail += result.admission_gate_object_fail;
                total_finite_ud_candidate_subgraphs += result.finite_ud_candidate_subgraphs;
                total_best_admission_margin += result.best_admission_margin;
                total_explore_us += static_cast<std::uint64_t>(result.explore_us);
                total_finalize_us += static_cast<std::uint64_t>(result.finalize_us);
                total_subgraph_admit_auto_disabled += result.subgraph_admit_auto_disabled ? 1U : 0U;
                total_object_density += result.object_density;
                total_occupied_subgraphs += result.occupied_subgraphs;
                total_occupied_subgraph_ratio += result.occupied_subgraph_ratio;
                total_avg_objects_per_occupied_subgraph += result.avg_objects_per_occupied_subgraph;
                total_max_objects_in_subgraph += result.max_objects_in_subgraph;
                if (result.final_safe_coverage_radius != bag::kInfWeight) {
                    ++finite_safe_coverage_count;
                    total_final_safe_coverage_radius += result.final_safe_coverage_radius;
                }
                if (result.first_finite_tau != bag::kInfWeight) {
                    ++finite_tau_count;
                    total_first_finite_tau += result.first_finite_tau;
                    total_first_finite_tau_boundary_visit_order += result.first_finite_tau_boundary_visit_order;
                }
            }

            const auto knn_total = std::accumulate(knn_times.begin(), knn_times.end(), 0LL);
            const auto knn_min = knn_times.empty() ? 0LL : *std::min_element(knn_times.begin(), knn_times.end());
            const auto knn_max = knn_times.empty() ? 0LL : *std::max_element(knn_times.begin(), knn_times.end());
            const auto denom = queries.empty() ? 1.0 : static_cast<double>(queries.size());

            std::cout
                << "{"
                << "\"query_count\":" << query_count << ","
                << "\"query_seed\":" << query_seed << ","
                << "\"objects\":" << object_count << ","
                << "\"object_seed\":" << object_seed << ","
                << "\"object_layout\":\"" << object_layout << "\","
                << "\"knn_k\":" << knn_k << ","
                << "\"knn_avg_us\":" << (queries.empty() ? 0.0 : static_cast<double>(knn_total) / denom) << ","
                << "\"knn_min_us\":" << knn_min << ","
                << "\"knn_max_us\":" << knn_max << ","
                << "\"avg_visited_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_visited_subgraphs) / denom) << ","
                << "\"avg_candidates_considered\":" << (queries.empty() ? 0.0 : static_cast<double>(total_candidates_considered) / denom) << ","
                << "\"avg_final_candidates\":" << (queries.empty() ? 0.0 : static_cast<double>(total_final_candidates) / denom) << ","
                << "\"avg_exact_evaluated\":" << (queries.empty() ? 0.0 : static_cast<double>(total_exact_evaluated) / denom) << ","
                << "\"avg_exact_from_admitted\":" << (queries.empty() ? 0.0 : static_cast<double>(total_exact_from_admitted) / denom) << ","
                << "\"avg_exact_from_unresolved\":" << (queries.empty() ? 0.0 : static_cast<double>(total_exact_from_unresolved) / denom) << ","
                << "\"avg_exact_reused_from_admitted_cache\":"
                << (queries.empty() ? 0.0 : static_cast<double>(total_exact_reused_from_admitted_cache) / denom) << ","
                << "\"avg_admitted_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admitted_subgraphs) / denom) << ","
                << "\"avg_admitted_objects\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admitted_objects) / denom) << ","
                << "\"avg_candidate_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_candidate_subgraphs) / denom) << ","
                << "\"avg_admission_refresh_calls\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_refresh_calls) / denom) << ","
                << "\"avg_admission_refresh_successes\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_refresh_successes) / denom) << ","
                << "\"avg_admission_ud_updates\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_ud_updates) / denom) << ","
                << "\"avg_admission_gate_heap_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_heap_fail) / denom) << ","
                << "\"avg_admission_gate_mass_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_mass_fail) / denom) << ","
                << "\"avg_admission_gate_avg_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_avg_fail) / denom) << ","
                << "\"avg_admission_gate_dt_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_dt_fail) / denom) << ","
                << "\"avg_admission_gate_no_admit_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_no_admit_fail) / denom) << ","
                << "\"avg_admission_gate_object_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_object_fail) / denom) << ","
                << "\"avg_finite_ud_candidate_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_finite_ud_candidate_subgraphs) / denom) << ","
                << "\"avg_best_admission_margin\":" << (queries.empty() ? 0.0 : static_cast<double>(total_best_admission_margin) / denom) << ","
                << "\"avg_object_density\":" << (queries.empty() ? 0.0 : total_object_density / denom) << ","
                << "\"avg_occupied_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_occupied_subgraphs) / denom) << ","
                << "\"avg_occupied_subgraph_ratio\":" << (queries.empty() ? 0.0 : total_occupied_subgraph_ratio / denom) << ","
                << "\"avg_objects_per_occupied_subgraph\":" << (queries.empty() ? 0.0 : total_avg_objects_per_occupied_subgraph / denom) << ","
                << "\"avg_max_objects_in_subgraph\":" << (queries.empty() ? 0.0 : static_cast<double>(total_max_objects_in_subgraph) / denom) << ","
                << "\"subgraph_admit_auto_disabled_fraction\":" << (queries.empty() ? 0.0 : static_cast<double>(total_subgraph_admit_auto_disabled) / denom) << ","
                << "\"avg_explore_us\":" << (queries.empty() ? 0.0 : static_cast<double>(total_explore_us) / denom) << ","
                << "\"avg_finalize_us\":" << (queries.empty() ? 0.0 : static_cast<double>(total_finalize_us) / denom) << ","
                << "\"avg_final_safe_coverage_radius\":"
                << (finite_safe_coverage_count == 0 ? 0.0 : static_cast<double>(total_final_safe_coverage_radius) / static_cast<double>(finite_safe_coverage_count)) << ","
                << "\"avg_first_finite_tau\":"
                << (finite_tau_count == 0 ? 0.0 : static_cast<double>(total_first_finite_tau) / static_cast<double>(finite_tau_count)) << ","
                << "\"avg_first_finite_tau_boundary_visit_order\":"
                << (finite_tau_count == 0 ? 0.0 : static_cast<double>(total_first_finite_tau_boundary_visit_order) / static_cast<double>(finite_tau_count)) << ","
                << "\"finite_tau_fraction\":"
                << (queries.empty() ? 0.0 : static_cast<double>(finite_tau_count) / denom) << ","
                << "\"finite_safe_coverage_fraction\":"
                << (queries.empty() ? 0.0 : static_cast<double>(finite_safe_coverage_count) / denom) << ","
                << "\"factorized_transfer\":" << (factorized_transfer ? "true" : "false") << ","
                << "\"knn_subgraph_admit\":" << (knn_subgraph_admit ? "true" : "false") << ","
                << "\"knn_admit_exact_cap\":" << (knn_admit_exact_cap ? "true" : "false") << ","
                << "\"knn_admit_density_fallback\":" << (knn_admit_density_fallback ? "true" : "false")
                << "}\n";
            return 0;
        }

        if (command == "knn-build-diagnostics") {
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_count = require_usize(args, "objects");
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto knn_k = optional_usize(args, "knn-k", 10);
            const bool knn_subgraph_admit = optional_bool(args, "knn-subgraph-admit", false);
            const bool knn_admit_exact_cap = optional_bool(args, "knn-admit-exact-cap", false);
            const bool knn_admit_density_fallback = optional_bool(args, "knn-admit-density-fallback", false);
            const bool knn_direct_row_path = optional_bool(args, "knn-direct-row-path", false);
            const auto layout_it = args.find("object-layout");
            const std::string object_layout = (layout_it == args.end()) ? "random" : layout_it->second;
            const auto log_stage = [&](const std::string& name) {
                if (progress_log) {
                    std::cerr << "[knn-build-diagnostics] " << name << std::endl;
                }
            };

            const bool factorized_transfer = optional_bool(args, "factorized-transfer", true);
            const double factorized_arc_threshold = optional_double(args, "factorized-arc-threshold", 0.5);
            const auto factorized_border_threshold = optional_usize(args, "factorized-border-threshold", 12);
            long long index_build_us = 0;
            bag::SkeletonIndex index;
            log_stage("building_index");
            const auto index_result = materialize_index(
                graph,
                subgraphs,
                partition_cache_state.key.empty() ? partition_identity_key : partition_cache_state.key,
                index_cache_mode,
                index_cache_dir,
                factorized_transfer,
                factorized_arc_threshold,
                factorized_border_threshold
            );
            index = index_result.index;
            index_build_us = index_result.index_us;
            log_stage("built_index");

            long long factorized_transfer_us = 0;
            if (index_result.cache_state.hit) {
                log_stage("loaded_index_cache");
            } else {
                log_stage(factorized_transfer ? "configured_factorized_transfer_during_index_build" : "built_index_without_factorized_transfer");
            }
            log_stage("factorized_transfer_done");

            long long object_build_us = 0;
            IndexedMovingObjectSet objects;
            log_stage("building_objects");
            object_build_us = bag::calc_execution_time_in_us([&] {
                objects = build_objects_with_layout(graph, index, object_count, object_seed, object_layout);
            });
            log_stage("built_objects");

            long long query_workload_build_us = 0;
            std::vector<QueryPoint> queries;
            log_stage("building_query_workload");
            query_workload_build_us = bag::calc_execution_time_in_us([&] {
                queries = build_query_workload(graph, query_count, query_seed);
            });
            log_stage("built_query_workload");

            std::vector<long long> knn_times;
            knn_times.reserve(query_count);
            std::uint64_t total_init_us = 0;
            std::uint64_t total_explore_us = 0;
            std::uint64_t total_finalize_us = 0;
            std::uint64_t total_visited_subgraphs = 0;
            log_stage("running_queries");
            for (std::size_t i = 0; i < queries.size(); ++i) {
                bag::KnnQueryResult result;
                knn_times.push_back(bag::calc_execution_time_in_us([&] {
                    result = index.knn_query(
                        queries[i],
                        knn_k,
                        objects,
                        i,
                        nullptr,
                        nullptr,
                        false,
                        false,
                        0,
                        factorized_transfer,
                        knn_direct_row_path,
                        knn_subgraph_admit,
                        knn_admit_exact_cap,
                        knn_admit_density_fallback
                    );
                }));
                total_init_us += static_cast<std::uint64_t>(result.init_us);
                total_explore_us += static_cast<std::uint64_t>(result.explore_us);
                total_finalize_us += static_cast<std::uint64_t>(result.finalize_us);
                total_visited_subgraphs += result.visited_subgraphs;
                if (progress_log) {
                    std::cerr
                        << "[knn-build-diagnostics] query_progress "
                        << (i + 1) << "/" << queries.size()
                        << std::endl;
                }
            }
            log_stage("queries_done");

            const auto knn_total = std::accumulate(knn_times.begin(), knn_times.end(), 0LL);
            const auto knn_min = knn_times.empty() ? 0LL : *std::min_element(knn_times.begin(), knn_times.end());
            const auto knn_max = knn_times.empty() ? 0LL : *std::max_element(knn_times.begin(), knn_times.end());
            const auto denom = queries.empty() ? 1.0 : static_cast<double>(queries.size());

            std::cout
                << "{"
                << "\"query_count\":" << query_count << ","
                << "\"query_seed\":" << query_seed << ","
                << "\"objects\":" << object_count << ","
                << "\"object_seed\":" << object_seed << ","
                << "\"knn_k\":" << knn_k << ","
                << "\"knn_direct_row_path\":" << (knn_direct_row_path ? "true" : "false") << ","
                << "\"object_layout\":\"" << object_layout << "\","
                << "\"phase1_inplace\":" << (phase1_inplace ? "true" : "false") << ","
                << "\"defer_nearest_border_fill\":" << (defer_nearest_border_fill ? "true" : "false") << ","
                << "\"skip_phase1_br_check\":" << (skip_phase1_br_check ? "true" : "false") << ","
                << "\"phase1_local_audit\":" << (phase1_local_audit ? "true" : "false") << ","
                << "\"phase2_incremental_distance_update\":" << (phase2_incremental_distance_update ? "true" : "false") << ","
                << "\"partition_total_us\":" << partition_us << ","
                << "\"partition_core_us\":" << partition_runtime_stats.core_partition_us << ","
                << "\"shortcut_repartition_us\":" << partition_runtime_stats.shortcut_repartition_us << ","
                << "\"subgraphs_before_shortcut\":" << partition_runtime_stats.subgraphs_before_shortcut << ","
                << "\"subgraphs_after_shortcut\":" << partition_runtime_stats.subgraphs_after_shortcut << ","
                << "\"phase1_attempts\":" << partition_runtime_stats.phase1_attempts << ","
                << "\"phase1_commits\":" << partition_runtime_stats.phase1_commits << ","
                << "\"phase1_rejects\":" << partition_runtime_stats.phase1_rejects << ","
                << "\"phase1_clone_us\":" << partition_runtime_stats.phase1_clone_us << ","
                << "\"phase1_finalize_us\":" << partition_runtime_stats.phase1_finalize_us << ","
                << "\"phase1_refresh_us\":" << partition_runtime_stats.phase1_refresh_us << ","
                << "\"phase1_extend_distance_us\":" << partition_runtime_stats.phase1_extend_distance_us << ","
                << "\"phase1_br_us\":" << partition_runtime_stats.phase1_br_us << ","
                << "\"phase1_nearest_border_us\":" << partition_runtime_stats.phase1_nearest_border_us << ","
                << "\"phase2_attempts\":" << partition_runtime_stats.phase2_attempts << ","
                << "\"phase2_commits\":" << partition_runtime_stats.phase2_commits << ","
                << "\"phase2_rejects\":" << partition_runtime_stats.phase2_rejects << ","
                << "\"phase2_clone_us\":" << partition_runtime_stats.phase2_clone_us << ","
                << "\"phase2_finalize_us\":" << partition_runtime_stats.phase2_finalize_us << ","
                << "\"phase2_refresh_us\":" << partition_runtime_stats.phase2_refresh_us << ","
                << "\"phase2_apsp_us\":" << partition_runtime_stats.phase2_apsp_us << ","
                << "\"phase2_distance_update_us\":" << partition_runtime_stats.phase2_distance_update_us << ","
                << "\"phase2_br_us\":" << partition_runtime_stats.phase2_br_us << ","
                << "\"phase2_nearest_border_us\":" << partition_runtime_stats.phase2_nearest_border_us << ","
                << "\"shortcut_tiny_subgraphs\":" << partition_runtime_stats.shortcut_stats.tiny_subgraphs << ","
                << "\"shortcut_skipped_by_tiny_budget\":" << (partition_runtime_stats.shortcut_stats.skipped_by_tiny_budget ? "true" : "false") << ","
                << "\"index_build_us\":" << index_build_us << ","
                << "\"factorized_transfer_us\":" << factorized_transfer_us << ","
                << "\"object_build_us\":" << object_build_us << ","
                << "\"query_workload_build_us\":" << query_workload_build_us << ","
                << "\"knn_avg_us\":" << (queries.empty() ? 0.0 : static_cast<double>(knn_total) / denom) << ","
                << "\"knn_min_us\":" << knn_min << ","
                << "\"knn_max_us\":" << knn_max << ","
                << "\"avg_knn_init_us\":" << (queries.empty() ? 0.0 : static_cast<double>(total_init_us) / denom) << ","
                << "\"avg_knn_explore_us\":" << (queries.empty() ? 0.0 : static_cast<double>(total_explore_us) / denom) << ","
                << "\"avg_knn_finalize_us\":" << (queries.empty() ? 0.0 : static_cast<double>(total_finalize_us) / denom) << ","
                << "\"avg_visited_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_visited_subgraphs) / denom) << ","
                << "\"factorized_transfer\":" << (factorized_transfer ? "true" : "false") << ","
                << "\"knn_subgraph_admit\":" << (knn_subgraph_admit ? "true" : "false") << ","
                << "\"knn_admit_exact_cap\":" << (knn_admit_exact_cap ? "true" : "false") << ","
                << "\"knn_admit_density_fallback\":" << (knn_admit_density_fallback ? "true" : "false");
            append_partition_checkpoint_json_fields(
                std::cout,
                partition_checkpoint_mode,
                partition_options.checkpoint_path,
                partition_checkpoint_state
            );
            append_partition_cache_json_fields(std::cout, partition_cache_state);
            append_index_cache_json_fields(std::cout, index_result.cache_state);
            std::cout << "}\n";
            return 0;
        }

        if (command == "knn-sweep") {
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_counts = parse_size_list(args, "objects-list", {});
            if (object_counts.empty()) {
                throw std::runtime_error("knn-sweep requires --objects-list");
            }
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto k_values = parse_size_list(args, "knn-k-list", {});
            if (k_values.empty()) {
                throw std::runtime_error("knn-sweep requires --knn-k-list");
            }
            const auto layouts = parse_string_list(args, "object-layouts", {"random"});
            const auto admit_modes = parse_string_list(args, "admit-modes", {"false", "true"});
            const bool knn_admit_exact_cap = optional_bool(args, "knn-admit-exact-cap", false);
            const bool knn_admit_density_fallback = optional_bool(args, "knn-admit-density-fallback", false);

            bag::SkeletonIndex index = SkeletonIndex::build(graph, subgraphs);
            const bool factorized_transfer = optional_bool(args, "factorized-transfer", true);
            const double factorized_arc_threshold = optional_double(args, "factorized-arc-threshold", 0.5);
            const auto factorized_border_threshold = optional_usize(args, "factorized-border-threshold", 12);
            if (factorized_transfer) {
                index.configure_factorized_transfer(factorized_arc_threshold, factorized_border_threshold);
            }
            const auto queries = build_query_workload(graph, query_count, query_seed);

            std::cout << "[";
            bool first_record = true;
            for (const auto& object_layout : layouts) {
                for (const auto object_count : object_counts) {
                    const auto objects = build_objects_with_layout(graph, index, object_count, object_seed, object_layout);
                    for (const auto knn_k : k_values) {
                        for (const auto& admit_mode_str : admit_modes) {
                            const bool knn_subgraph_admit =
                                (admit_mode_str == "true" || admit_mode_str == "1" || admit_mode_str == "yes");
                            std::vector<long long> knn_times;
                            knn_times.reserve(query_count);
                            std::uint64_t total_visited_subgraphs = 0;
                            std::uint64_t total_candidates_considered = 0;
                            std::uint64_t total_final_candidates = 0;
                            std::uint64_t total_exact_evaluated = 0;
                            std::uint64_t total_exact_from_admitted = 0;
                            std::uint64_t total_exact_from_unresolved = 0;
                            std::uint64_t total_exact_reused_from_admitted_cache = 0;
                            std::uint64_t total_admitted_subgraphs = 0;
                            std::uint64_t total_admitted_objects = 0;
                            std::uint64_t total_candidate_subgraphs = 0;
                            std::uint64_t total_admission_refresh_calls = 0;
                            std::uint64_t total_admission_refresh_successes = 0;
                            std::uint64_t total_admission_ud_updates = 0;
                            std::uint64_t total_admission_gate_heap_fail = 0;
                            std::uint64_t total_admission_gate_mass_fail = 0;
                            std::uint64_t total_admission_gate_avg_fail = 0;
                            std::uint64_t total_admission_gate_dt_fail = 0;
                            std::uint64_t total_admission_gate_no_admit_fail = 0;
                            std::uint64_t total_admission_gate_object_fail = 0;
                            std::uint64_t total_finite_ud_candidate_subgraphs = 0;
                            long long total_best_admission_margin = 0;
                            std::uint64_t total_explore_us = 0;
                            std::uint64_t total_finalize_us = 0;
                            std::uint64_t total_subgraph_admit_auto_disabled = 0;
                            double total_object_density = 0.0;
                            std::uint64_t total_occupied_subgraphs = 0;
                            double total_occupied_subgraph_ratio = 0.0;
                            double total_avg_objects_per_occupied_subgraph = 0.0;
                            std::uint64_t total_max_objects_in_subgraph = 0;
                            std::uint64_t total_pq_us = 0;
                            std::uint64_t total_subgraph_bookkeeping_us = 0;
                            std::uint64_t total_clique_emit_us = 0;
                            std::uint64_t total_num_clique_relax_attempts = 0;
                            std::uint64_t total_num_successful_clique_relaxes = 0;
                            std::uint64_t total_num_pq_pushes_from_clique = 0;
                            std::uint64_t total_factorized_exits_emitted = 0;
                            std::uint64_t total_root_kth_exact = 0;
                            std::uint64_t finite_root_kth_exact_count = 0;
                            std::uint64_t total_final_kth_exact = 0;
                            std::uint64_t finite_final_kth_exact_count = 0;
                            std::uint64_t finite_safe_coverage_count = 0;
                            std::uint64_t total_final_safe_coverage_radius = 0;
                            std::uint64_t finite_tau_count = 0;
                            std::uint64_t total_first_finite_tau = 0;
                            std::uint64_t total_first_finite_tau_boundary_visit_order = 0;

                            for (std::size_t i = 0; i < queries.size(); ++i) {
                                bag::KnnQueryResult result;
                                knn_times.push_back(bag::calc_execution_time_in_us([&] {
                                    result = index.knn_query(
                                        queries[i],
                                        knn_k,
                                        objects,
                                        i,
                                        nullptr,
                                        nullptr,
                                        false,
                                        false,
                                        0,
                                        factorized_transfer,
                                        false,
                                        knn_subgraph_admit,
                                        knn_admit_exact_cap,
                                        knn_admit_density_fallback
                                    );
                                }));
                                total_visited_subgraphs += result.visited_subgraphs;
                                total_candidates_considered += result.candidates_considered;
                                total_final_candidates += result.final_candidates;
                                total_exact_evaluated += result.exact_evaluated;
                                total_exact_from_admitted += result.exact_from_admitted;
                                total_exact_from_unresolved += result.exact_from_unresolved;
                                total_exact_reused_from_admitted_cache += result.exact_reused_from_admitted_cache;
                                total_admitted_subgraphs += result.admitted_subgraphs;
                                total_admitted_objects += result.admitted_objects;
                                total_candidate_subgraphs += result.candidate_subgraphs;
                                total_admission_refresh_calls += result.admission_refresh_calls;
                                total_admission_refresh_successes += result.admission_refresh_successes;
                                total_admission_ud_updates += result.admission_ud_updates;
                                total_admission_gate_heap_fail += result.admission_gate_heap_fail;
                                total_admission_gate_mass_fail += result.admission_gate_mass_fail;
                                total_admission_gate_avg_fail += result.admission_gate_avg_fail;
                                total_admission_gate_dt_fail += result.admission_gate_dt_fail;
                                total_admission_gate_no_admit_fail += result.admission_gate_no_admit_fail;
                                total_admission_gate_object_fail += result.admission_gate_object_fail;
                                total_finite_ud_candidate_subgraphs += result.finite_ud_candidate_subgraphs;
                                total_best_admission_margin += result.best_admission_margin;
                                total_explore_us += static_cast<std::uint64_t>(result.explore_us);
                                total_finalize_us += static_cast<std::uint64_t>(result.finalize_us);
                                total_subgraph_admit_auto_disabled += result.subgraph_admit_auto_disabled ? 1U : 0U;
                                total_object_density += result.object_density;
                                total_occupied_subgraphs += result.occupied_subgraphs;
                                total_occupied_subgraph_ratio += result.occupied_subgraph_ratio;
                                total_avg_objects_per_occupied_subgraph += result.avg_objects_per_occupied_subgraph;
                                total_max_objects_in_subgraph += result.max_objects_in_subgraph;
                                total_pq_us += static_cast<std::uint64_t>(result.pq_us);
                                total_subgraph_bookkeeping_us += static_cast<std::uint64_t>(result.subgraph_bookkeeping_us);
                                total_clique_emit_us += static_cast<std::uint64_t>(result.clique_emit_us);
                                total_num_clique_relax_attempts += result.num_clique_relax_attempts;
                                total_num_successful_clique_relaxes += result.num_successful_clique_relaxes;
                                total_num_pq_pushes_from_clique += result.num_pq_pushes_from_clique;
                                total_factorized_exits_emitted += result.factorized_exits_emitted;
                                if (result.root_kth_exact != bag::kInfWeight) {
                                    ++finite_root_kth_exact_count;
                                    total_root_kth_exact += result.root_kth_exact;
                                }
                                if (result.final_kth_exact != bag::kInfWeight) {
                                    ++finite_final_kth_exact_count;
                                    total_final_kth_exact += result.final_kth_exact;
                                }
                                if (result.final_safe_coverage_radius != bag::kInfWeight) {
                                    ++finite_safe_coverage_count;
                                    total_final_safe_coverage_radius += result.final_safe_coverage_radius;
                                }
                                if (result.first_finite_tau != bag::kInfWeight) {
                                    ++finite_tau_count;
                                    total_first_finite_tau += result.first_finite_tau;
                                    total_first_finite_tau_boundary_visit_order += result.first_finite_tau_boundary_visit_order;
                                }
                            }

                            const auto knn_total = std::accumulate(knn_times.begin(), knn_times.end(), 0LL);
                            const auto knn_min = knn_times.empty() ? 0LL : *std::min_element(knn_times.begin(), knn_times.end());
                            const auto knn_max = knn_times.empty() ? 0LL : *std::max_element(knn_times.begin(), knn_times.end());
                            const auto denom = queries.empty() ? 1.0 : static_cast<double>(queries.size());

                            if (!first_record) {
                                std::cout << ",";
                            }
                            first_record = false;
                            std::cout
                                << "{"
                                << "\"query_count\":" << query_count << ","
                                << "\"query_seed\":" << query_seed << ","
                                << "\"objects\":" << object_count << ","
                                << "\"object_seed\":" << object_seed << ","
                                << "\"object_layout\":\"" << object_layout << "\","
                                << "\"knn_k\":" << knn_k << ","
                                << "\"knn_avg_us\":" << (queries.empty() ? 0.0 : static_cast<double>(knn_total) / denom) << ","
                                << "\"knn_min_us\":" << knn_min << ","
                                << "\"knn_max_us\":" << knn_max << ","
                                << "\"avg_visited_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_visited_subgraphs) / denom) << ","
                                << "\"avg_candidates_considered\":" << (queries.empty() ? 0.0 : static_cast<double>(total_candidates_considered) / denom) << ","
                                << "\"avg_final_candidates\":" << (queries.empty() ? 0.0 : static_cast<double>(total_final_candidates) / denom) << ","
                                << "\"avg_exact_evaluated\":" << (queries.empty() ? 0.0 : static_cast<double>(total_exact_evaluated) / denom) << ","
                                << "\"avg_exact_from_admitted\":" << (queries.empty() ? 0.0 : static_cast<double>(total_exact_from_admitted) / denom) << ","
                                << "\"avg_exact_from_unresolved\":" << (queries.empty() ? 0.0 : static_cast<double>(total_exact_from_unresolved) / denom) << ","
                                << "\"avg_exact_reused_from_admitted_cache\":"
                                << (queries.empty() ? 0.0 : static_cast<double>(total_exact_reused_from_admitted_cache) / denom) << ","
                                << "\"avg_admitted_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admitted_subgraphs) / denom) << ","
                                << "\"avg_admitted_objects\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admitted_objects) / denom) << ","
                                << "\"avg_candidate_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_candidate_subgraphs) / denom) << ","
                                << "\"avg_admission_refresh_calls\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_refresh_calls) / denom) << ","
                                << "\"avg_admission_refresh_successes\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_refresh_successes) / denom) << ","
                                << "\"avg_admission_ud_updates\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_ud_updates) / denom) << ","
                                << "\"avg_admission_gate_heap_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_heap_fail) / denom) << ","
                                << "\"avg_admission_gate_mass_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_mass_fail) / denom) << ","
                                << "\"avg_admission_gate_avg_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_avg_fail) / denom) << ","
                                << "\"avg_admission_gate_dt_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_dt_fail) / denom) << ","
                                << "\"avg_admission_gate_no_admit_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_no_admit_fail) / denom) << ","
                                << "\"avg_admission_gate_object_fail\":" << (queries.empty() ? 0.0 : static_cast<double>(total_admission_gate_object_fail) / denom) << ","
                                << "\"avg_finite_ud_candidate_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_finite_ud_candidate_subgraphs) / denom) << ","
                                << "\"avg_best_admission_margin\":" << (queries.empty() ? 0.0 : static_cast<double>(total_best_admission_margin) / denom) << ","
                                << "\"avg_object_density\":" << (queries.empty() ? 0.0 : total_object_density / denom) << ","
                                << "\"avg_occupied_subgraphs\":" << (queries.empty() ? 0.0 : static_cast<double>(total_occupied_subgraphs) / denom) << ","
                                << "\"avg_occupied_subgraph_ratio\":" << (queries.empty() ? 0.0 : total_occupied_subgraph_ratio / denom) << ","
                                << "\"avg_objects_per_occupied_subgraph\":" << (queries.empty() ? 0.0 : total_avg_objects_per_occupied_subgraph / denom) << ","
                                << "\"avg_max_objects_in_subgraph\":" << (queries.empty() ? 0.0 : static_cast<double>(total_max_objects_in_subgraph) / denom) << ","
                                << "\"subgraph_admit_auto_disabled_fraction\":" << (queries.empty() ? 0.0 : static_cast<double>(total_subgraph_admit_auto_disabled) / denom) << ","
                                << "\"avg_explore_us\":" << (queries.empty() ? 0.0 : static_cast<double>(total_explore_us) / denom) << ","
                                << "\"avg_finalize_us\":" << (queries.empty() ? 0.0 : static_cast<double>(total_finalize_us) / denom) << ","
                                << "\"avg_pq_us\":" << (queries.empty() ? 0.0 : static_cast<double>(total_pq_us) / denom) << ","
                                << "\"avg_subgraph_bookkeeping_us\":" << (queries.empty() ? 0.0 : static_cast<double>(total_subgraph_bookkeeping_us) / denom) << ","
                                << "\"avg_clique_emit_us\":" << (queries.empty() ? 0.0 : static_cast<double>(total_clique_emit_us) / denom) << ","
                                << "\"avg_num_clique_relax_attempts\":" << (queries.empty() ? 0.0 : static_cast<double>(total_num_clique_relax_attempts) / denom) << ","
                                << "\"avg_num_successful_clique_relaxes\":" << (queries.empty() ? 0.0 : static_cast<double>(total_num_successful_clique_relaxes) / denom) << ","
                                << "\"avg_num_pq_pushes_from_clique\":" << (queries.empty() ? 0.0 : static_cast<double>(total_num_pq_pushes_from_clique) / denom) << ","
                                << "\"avg_factorized_exits_emitted\":" << (queries.empty() ? 0.0 : static_cast<double>(total_factorized_exits_emitted) / denom) << ","
                                << "\"avg_root_kth_exact\":" << (finite_root_kth_exact_count == 0 ? 0.0 : static_cast<double>(total_root_kth_exact) / static_cast<double>(finite_root_kth_exact_count)) << ","
                                << "\"avg_final_kth_exact\":" << (finite_final_kth_exact_count == 0 ? 0.0 : static_cast<double>(total_final_kth_exact) / static_cast<double>(finite_final_kth_exact_count)) << ","
                                << "\"avg_final_safe_coverage_radius\":"
                                << (finite_safe_coverage_count == 0 ? 0.0 : static_cast<double>(total_final_safe_coverage_radius) / static_cast<double>(finite_safe_coverage_count)) << ","
                                << "\"avg_first_finite_tau\":"
                                << (finite_tau_count == 0 ? 0.0 : static_cast<double>(total_first_finite_tau) / static_cast<double>(finite_tau_count)) << ","
                                << "\"avg_first_finite_tau_boundary_visit_order\":"
                                << (finite_tau_count == 0 ? 0.0 : static_cast<double>(total_first_finite_tau_boundary_visit_order) / static_cast<double>(finite_tau_count)) << ","
                                << "\"finite_tau_fraction\":"
                                << (queries.empty() ? 0.0 : static_cast<double>(finite_tau_count) / denom) << ","
                                << "\"finite_safe_coverage_fraction\":"
                                << (queries.empty() ? 0.0 : static_cast<double>(finite_safe_coverage_count) / denom) << ","
                                << "\"factorized_transfer\":" << (factorized_transfer ? "true" : "false") << ","
                                << "\"knn_subgraph_admit\":" << (knn_subgraph_admit ? "true" : "false") << ","
                                << "\"knn_admit_exact_cap\":" << (knn_admit_exact_cap ? "true" : "false") << ","
                                << "\"knn_admit_density_fallback\":" << (knn_admit_density_fallback ? "true" : "false")
                                << "}";
                        }
                    }
                }
            }
            std::cout << "]\n";
            return 0;
        }

        if (command == "maint-bench") {
            const double car_percent = optional_double(args, "car-percent", 0.025);
            const double change_percent = optional_double(args, "change-percent", 0.01);
            const auto query_per_update = optional_usize(args, "query-per-update", 10);
            const auto epochs = optional_usize(args, "epochs", 100);
            const auto knn_k = optional_usize(args, "knn-k", 10);
            const auto range_radius =
                static_cast<EdgeWeight>(optional_usize(args, "range-radius", 50000));
            const auto mode_it = args.find("mode");
            const std::string maint_mode = (mode_it == args.end()) ? "knn" : mode_it->second;
            const auto fc_rule = parse_fc_rule(args);
            const bool verify_updates = optional_bool(args, "verify-updates", false);
            const bool knn_subgraph_admit = optional_bool(args, "knn-subgraph-admit", false);
            const bool knn_admit_exact_cap = optional_bool(args, "knn-admit-exact-cap", false);
            const bool unified_object_maintenance =
                optional_bool(args, "unified-object-maintenance", false);
            const bool release_subgraph_distances =
                optional_bool(args, "release-subgraph-distances", false);
            const bool local_move = optional_bool(args, "local-move", true);
            const bool eager_knn_finalize = optional_bool(args, "eager-knn-finalize", false);
            const bool range_whole_subgraph_acceptance =
                optional_bool(args, "range-whole-subgraph-acceptance", true);
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto layout_it = args.find("object-layout");
            const std::string object_layout = (layout_it == args.end()) ? "random" : layout_it->second;

            const bool factorized_transfer = optional_bool(args, "factorized-transfer", true);
            const double factorized_arc_threshold = optional_double(args, "factorized-arc-threshold", 0.5);
            const auto factorized_border_threshold = optional_usize(args, "factorized-border-threshold", 12);
            long long index_build_us = 0;
            bag::SkeletonIndex index;
            const auto index_result = materialize_index(
                graph,
                subgraphs,
                partition_cache_state.key.empty() ? partition_identity_key : partition_cache_state.key,
                index_cache_mode,
                index_cache_dir,
                factorized_transfer,
                factorized_arc_threshold,
                factorized_border_threshold
            );
            index = index_result.index;
            index_build_us = index_result.index_us;
            if (release_subgraph_distances) {
                index.release_subgraph_distances();
            }

            const std::size_t object_count = static_cast<std::size_t>(
                std::max(1.0, std::floor(static_cast<double>(graph.size()) * car_percent))
            );
            const std::size_t change_count = std::min(
                object_count,
                static_cast<std::size_t>(std::floor(static_cast<double>(object_count) * change_percent))
            );

            auto active_objects = build_object_vector(graph, index, object_count, object_seed, object_layout);
            const auto queries = build_query_workload(
                graph,
                std::max<std::size_t>(epochs, query_per_update),
                query_seed
            );
            const bool has_fixed_query =
                args.find("query-u") != args.end() &&
                args.find("query-v") != args.end() &&
                args.find("offset") != args.end();
            const auto effective_queries = has_fixed_query
                ? build_fixed_query_workload(
                      QueryPoint{bag::ordered_edge(require_vertex(args, "query-u"), require_vertex(args, "query-v")),
                                 require_weight(args, "offset")},
                      std::max<std::size_t>(epochs, query_per_update))
                : queries;

            long long initial_object_index_us = 0;
            IndexedMovingObjectSet indexed_objects;
            const bool maintain_knn_metadata =
                unified_object_maintenance || maint_mode != "range";
            const bool maintain_edge_buckets =
                unified_object_maintenance || maint_mode == "range";
            initial_object_index_us = bag::calc_execution_time_in_us([&] {
                indexed_objects = IndexedMovingObjectSet::from_partition(
                    active_objects,
                    index.edge_to_subgraph(),
                    index.subgraphs(),
                    maintain_knn_metadata,
                    maintain_edge_buckets,
                    eager_knn_finalize
                );
            });

            long long one_time_query_total_us = 0;
            for (std::size_t i = 0; i < epochs; ++i) {
                one_time_query_total_us += bag::calc_execution_time_in_us([&] {
                    if (maint_mode == "range") {
                        (void)index.range_query(
                            effective_queries[i % effective_queries.size()],
                            range_radius,
                            indexed_objects,
                            fc_rule,
                            i,
                            nullptr,
                            nullptr,
                            true,
                            false,
                            range_whole_subgraph_acceptance
                        );
                    } else {
                        (void)index.knn_query(
                            effective_queries[i % effective_queries.size()],
                            knn_k,
                            indexed_objects,
                            i,
                            nullptr,
                            nullptr,
                            false,
                            false,
                            0,
                            factorized_transfer,
                            false,
                            knn_subgraph_admit,
                            knn_admit_exact_cap
                        );
                    }
                });
            }

            std::vector<std::pair<bag::Edge, EdgeWeight>> move_slots;
            move_slots.reserve(graph.edge_count());
            for (const auto& [edge, weight] : graph.undirected_edges()) {
                if (weight > 1) {
                    move_slots.push_back({bag::ordered_edge(edge.first, edge.second), weight});
                }
            }
            if (move_slots.empty()) {
                throw std::runtime_error("graph does not have any valid move slots");
            }

            std::unordered_map<bag::VertexId, std::vector<std::size_t>> vertex_move_slot_indices;
            if (local_move) {
                vertex_move_slot_indices.reserve(graph.size() * 2U + 1U);
                for (std::size_t slot_index = 0; slot_index < move_slots.size(); ++slot_index) {
                    const auto& [edge, weight] = move_slots[slot_index];
                    (void)weight;
                    vertex_move_slot_indices[edge.first].push_back(slot_index);
                    vertex_move_slot_indices[edge.second].push_back(slot_index);
                }
            }

            std::mt19937_64 rng(object_seed + 1);
            std::uniform_int_distribution<std::size_t> object_pick(0, object_count - 1);
            auto sample_move_choice = [&](const bag::MovingObject& current) {
                if (!local_move) {
                    std::uniform_int_distribution<std::size_t> slot_pick(0, move_slots.size() - 1);
                    const auto& [edge, weight] = move_slots[slot_pick(rng)];
                    std::uniform_int_distribution<EdgeWeight> offset_dist(1, weight - 1);
                    return std::tuple<bag::Edge, EdgeWeight, EdgeWeight>{edge, weight, offset_dist(rng)};
                }

                const auto left_it = vertex_move_slot_indices.find(current.edge.first);
                const auto right_it = vertex_move_slot_indices.find(current.edge.second);
                const std::size_t left_size =
                    (left_it == vertex_move_slot_indices.end()) ? 0U : left_it->second.size();
                const std::size_t right_size =
                    (right_it == vertex_move_slot_indices.end()) ? 0U : right_it->second.size();
                const std::size_t total_local = left_size + right_size;
                if (total_local == 0U) {
                    std::uniform_int_distribution<std::size_t> slot_pick(0, move_slots.size() - 1);
                    const auto& [edge, weight] = move_slots[slot_pick(rng)];
                    std::uniform_int_distribution<EdgeWeight> offset_dist(1, weight - 1);
                    return std::tuple<bag::Edge, EdgeWeight, EdgeWeight>{edge, weight, offset_dist(rng)};
                }
                std::uniform_int_distribution<std::size_t> local_pick(0, total_local - 1U);
                const auto local_index = local_pick(rng);
                const std::size_t slot_index =
                    (local_index < left_size)
                        ? left_it->second[local_index]
                        : right_it->second[local_index - left_size];
                const auto& [edge, weight] = move_slots[slot_index];
                std::uniform_int_distribution<EdgeWeight> offset_dist(1, weight - 1);
                return std::tuple<bag::Edge, EdgeWeight, EdgeWeight>{edge, weight, offset_dist(rng)};
            };

            bool verify_knn_ok = true;
            bool verify_range_ok = true;
            long long total_update_us = 0;
            long long total_query_us = 0;
            for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
                total_update_us += bag::calc_execution_time_in_us([&] {
                    for (std::size_t i = 0; i < change_count; ++i) {
                        const auto obj_index = object_pick(rng);
                        auto& object = active_objects[obj_index];
                        const auto [edge, weight, offset] = sample_move_choice(object);
                        object.edge = edge;
                        object.edge_weight = weight;
                        object.offset = offset;
                        indexed_objects.move_object(
                            object,
                            index.edge_to_subgraph(),
                            index.subgraphs()
                        );
                    }
                    indexed_objects.finalize_updates();
                });

                if (verify_updates && (verify_knn_ok || verify_range_ok)) {
                    const auto rebuilt_objects = IndexedMovingObjectSet::from_partition(
                        active_objects,
                        index.edge_to_subgraph(),
                        index.subgraphs(),
                        maintain_knn_metadata,
                        maintain_edge_buckets,
                        eager_knn_finalize
                    );
                    for (std::size_t i = 0; i < query_per_update; ++i) {
                        const auto& query = effective_queries[(epoch * query_per_update + i) % effective_queries.size()];
                        if (verify_knn_ok && maint_mode != "range") {
                            const auto incremental_knn = index.knn_query(
                                query,
                                knn_k,
                                indexed_objects,
                                epoch * query_per_update + i,
                                nullptr,
                                nullptr,
                                false,
                                false,
                                0,
                                factorized_transfer,
                                false,
                                knn_subgraph_admit,
                                knn_admit_exact_cap
                            );
                            const auto rebuilt_knn = index.knn_query(
                                query,
                                knn_k,
                                rebuilt_objects,
                                epoch * query_per_update + i,
                                nullptr,
                                nullptr,
                                false,
                                false,
                                0,
                                factorized_transfer,
                                false,
                                knn_subgraph_admit,
                                knn_admit_exact_cap
                            );
                            if (incremental_knn.items.size() != rebuilt_knn.items.size()) {
                                verify_knn_ok = false;
                            } else {
                                for (std::size_t j = 0; j < incremental_knn.items.size(); ++j) {
                                    if (incremental_knn.items[j].id != rebuilt_knn.items[j].id ||
                                        incremental_knn.items[j].distance != rebuilt_knn.items[j].distance) {
                                        verify_knn_ok = false;
                                        break;
                                    }
                                }
                            }
                        }

                        if (verify_range_ok) {
                            const auto incremental_range = index.range_query(
                                query,
                                range_radius,
                                indexed_objects,
                                fc_rule,
                                epoch * query_per_update + i,
                                nullptr,
                                nullptr,
                                true,
                                false,
                                range_whole_subgraph_acceptance
                            );
                            const auto rebuilt_range = index.range_query(
                                query,
                                range_radius,
                                rebuilt_objects,
                                fc_rule,
                                epoch * query_per_update + i,
                                nullptr,
                                nullptr,
                                true,
                                false,
                                range_whole_subgraph_acceptance
                            );
                            if (incremental_range.object_ids != rebuilt_range.object_ids) {
                                verify_range_ok = false;
                            }
                        }

                        if (!verify_knn_ok && !verify_range_ok) {
                            break;
                        }
                    }
                }

                total_query_us += bag::calc_execution_time_in_us([&] {
                    for (std::size_t i = 0; i < query_per_update; ++i) {
                        if (maint_mode == "range") {
                            (void)index.range_query(
                                effective_queries[i % effective_queries.size()],
                                range_radius,
                                indexed_objects,
                                fc_rule,
                                epoch * query_per_update + i,
                                nullptr,
                                nullptr,
                                true,
                                false,
                                range_whole_subgraph_acceptance
                            );
                        } else {
                            (void)index.knn_query(
                                effective_queries[i % effective_queries.size()],
                                knn_k,
                                indexed_objects,
                                epoch * query_per_update + i,
                                nullptr,
                                nullptr,
                                false,
                                false,
                                0,
                                factorized_transfer,
                                false,
                                knn_subgraph_admit,
                                knn_admit_exact_cap
                            );
                        }
                    }
                });
            }

            const auto total_queries = epochs * query_per_update;
            std::cout
                << "{"
                << "\"graph_vertices\":" << graph.size() << ","
                << "\"subgraphs\":" << index.subgraphs().size() << ","
                << "\"partition_us\":" << partition_us << ","
                << "\"index_build_us\":" << index_build_us << ","
                << "\"car_percent\":" << car_percent << ","
                << "\"change_percent\":" << change_percent << ","
                << "\"object_count\":" << object_count << ","
                << "\"change_count\":" << change_count << ","
                << "\"query_per_update\":" << query_per_update << ","
                << "\"epochs\":" << epochs << ","
                << "\"knn_k\":" << knn_k << ","
                << "\"mode\":\"" << maint_mode << "\","
                << "\"initial_object_index_us\":" << initial_object_index_us << ","
                << "\"one_time_query_avg_us\":"
                << (epochs == 0 ? 0.0 : static_cast<double>(one_time_query_total_us) / static_cast<double>(epochs)) << ","
                << "\"factorized_transfer\":" << (factorized_transfer ? "true" : "false") << ","
                << "\"knn_subgraph_admit\":" << (knn_subgraph_admit ? "true" : "false") << ","
                << "\"knn_admit_exact_cap\":" << (knn_admit_exact_cap ? "true" : "false") << ","
                << "\"unified_object_maintenance\":"
                << (unified_object_maintenance ? "true" : "false") << ","
                << "\"local_move\":" << (local_move ? "true" : "false") << ","
                << "\"eager_knn_finalize\":" << (eager_knn_finalize ? "true" : "false") << ","
                << "\"range_whole_subgraph_acceptance\":"
                << (range_whole_subgraph_acceptance ? "true" : "false") << ","
                << "\"avg_update_us\":"
                << (epochs == 0 ? 0.0 : static_cast<double>(total_update_us) / static_cast<double>(epochs)) << ","
                << "\"avg_query_us\":"
                << (total_queries == 0 ? 0.0 : static_cast<double>(total_query_us) / static_cast<double>(total_queries)) << ","
                << "\"amortized_us\":"
                << (total_queries == 0 ? 0.0 : static_cast<double>(total_update_us + total_query_us) / static_cast<double>(total_queries)) << ","
                      << "\"verify_updates\":" << (verify_updates ? "true" : "false") << ","
                      << "\"fixed_query\":" << (has_fixed_query ? "true" : "false") << ","
                      << "\"verify_knn_ok\":" << (verify_knn_ok ? "true" : "false") << ","
                << "\"verify_range_ok\":" << (verify_range_ok ? "true" : "false");
            append_partition_checkpoint_json_fields(
                std::cout,
                partition_checkpoint_mode,
                partition_options.checkpoint_path,
                partition_checkpoint_state
            );
            append_partition_cache_json_fields(std::cout, partition_cache_state);
            append_index_cache_json_fields(std::cout, index_result.cache_state);
            std::cout << "}\n";
            return (verify_knn_ok && verify_range_ok) ? 0 : 2;
        }

        if (command == "memory-report") {
            const auto mode = require_arg(args, "mode");
            const auto object_count = optional_usize(args, "objects", 10000);
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const bool unified_object_maintenance = optional_bool(args, "unified-object-maintenance", false);
            const bool release_subgraph_distances = optional_bool(args, "release-subgraph-distances", false);

            bag::SkeletonIndex index = SkeletonIndex::build(graph, subgraphs);
            if (release_subgraph_distances) {
                index.release_subgraph_distances();
            }
            const bool maintain_knn_metadata = unified_object_maintenance || mode == "knn";
            const bool maintain_edge_buckets = unified_object_maintenance || mode == "range";
            auto objects = build_objects(
                graph,
                index,
                object_count,
                object_seed,
                maintain_knn_metadata,
                maintain_edge_buckets
            );

            const auto index_report = estimate_index_memory_report(graph, index);
            const auto object_report = estimate_object_memory_report(index, objects, maintain_knn_metadata);
            std::cout
                << "{"
                << "\"mode\":\"" << mode << "\","
                << "\"graph_vertices\":" << graph.size() << ","
                << "\"subgraphs\":" << index.subgraphs().size() << ","
                << "\"objects\":" << object_count << ","
                << "\"unified_object_maintenance\":" << (unified_object_maintenance ? "true" : "false") << ","
                << "\"release_subgraph_distances\":" << (release_subgraph_distances ? "true" : "false") << ","
                << "\"static_total_bytes\":" << index_report.total_bytes << ","
                << "\"skeleton_graph_bytes\":" << index_report.skeleton_graph_bytes << ","
                << "\"sorted_skeleton_rows_bytes\":" << index_report.sorted_skeleton_rows_bytes << ","
                << "\"sorted_skeleton_row_index_bytes\":" << index_report.sorted_skeleton_row_index_bytes << ","
                << "\"vertex_to_subgraph_bytes\":" << index_report.vertex_to_subgraph_bytes << ","
                << "\"inverted_index_bytes\":" << index_report.inverted_index_bytes << ","
                << "\"edge_to_subgraph_bytes\":" << index_report.edge_to_subgraph_bytes << ","
                << "\"subgraph_graph_bytes\":" << index_report.subgraph_graph_bytes << ","
                << "\"subgraph_distance_bytes\":" << index_report.subgraph_distance_bytes << ","
                << "\"boundary_bytes\":" << index_report.boundary_bytes << ","
                << "\"rb_bytes\":" << index_report.rb_bytes << ","
                << "\"internal_to_border_bytes\":" << index_report.internal_to_border_bytes << ","
                << "\"clique_rows_bytes\":" << index_report.clique_rows_bytes << ","
                << "\"local_index_bytes\":" << index_report.local_index_bytes << ","
                << "\"subgraph_adjacency_bytes\":" << index_report.subgraph_adjacency_bytes << ","
                << "\"object_total_bytes\":" << object_report.total_bytes << ","
                << "\"object_vector_bytes\":" << object_report.object_vector_bytes << ","
                << "\"subgraph_object_lists_bytes\":" << object_report.subgraph_object_lists_bytes << ","
                << "\"edge_bucket_bytes\":" << object_report.edge_bucket_bytes << ","
                << "\"knn_sorted_order_bytes\":" << object_report.knn_sorted_order_bytes << ","
                << "\"knn_suffix_bytes\":" << object_report.knn_suffix_bytes << ","
                << "\"knn_border_cost_bytes\":" << object_report.knn_border_cost_bytes << ","
                << "\"query_total_bytes\":" << (index_report.total_bytes + object_report.total_bytes)
                << "}\n";
            return 0;
        }

        if (command == "range-coverage-scan") {
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_count = require_usize(args, "objects");
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto range_radius =
                static_cast<EdgeWeight>(require_usize(args, "range-radius"));
            const auto fc_rule = parse_fc_rule(args);
            const auto layout_it = args.find("object-layout");
            const std::string object_layout = (layout_it == args.end()) ? "random" : layout_it->second;

            bag::SkeletonIndex index = SkeletonIndex::build(graph, subgraphs);
            const bool factorized_transfer = optional_bool(args, "factorized-transfer", true);
            const double factorized_arc_threshold = optional_double(args, "factorized-arc-threshold", 0.5);
            const auto factorized_border_threshold = optional_usize(args, "factorized-border-threshold", 12);
            if (factorized_transfer) {
                index.configure_factorized_transfer(factorized_arc_threshold, factorized_border_threshold);
            }
            const auto objects = build_objects_with_layout(graph, index, object_count, object_seed, object_layout);
            const auto queries = build_query_workload(graph, query_count, query_seed);

            std::cout << "{"
                      << "\"query_count\":" << queries.size() << ","
                      << "\"query_seed\":" << query_seed << ","
                      << "\"object_count\":" << object_count << ","
                      << "\"range_radius\":" << range_radius << ","
                      << "\"rows\":[";
            for (std::size_t i = 0; i < queries.size(); ++i) {
                const auto& q = queries[i];
                const auto weight = graph.get_weight(q.edge.first, q.edge.second).value_or(0);
                const auto result = index.range_query(
                    q,
                    range_radius,
                    objects,
                    fc_rule,
                    i,
                    nullptr,
                    nullptr,
                    true,
                    factorized_transfer,
                    true
                );
                const auto result_subgraphs = result.fc_subgraphs + result.pc_subgraphs;
                const auto ratio =
                    (result_subgraphs == 0)
                        ? 0.0
                        : static_cast<double>(result.fc_subgraphs) / static_cast<double>(result_subgraphs);
                if (i != 0) {
                    std::cout << ",";
                }
                std::cout << "{"
                          << "\"query_id\":" << i << ","
                          << "\"u\":" << q.edge.first << ","
                          << "\"v\":" << q.edge.second << ","
                          << "\"offset\":" << q.offset << ","
                          << "\"edge_weight\":" << weight << ","
                          << "\"fc_subgraphs\":" << result.fc_subgraphs << ","
                          << "\"pc_subgraphs\":" << result.pc_subgraphs << ","
                          << "\"result_subgraphs\":" << result_subgraphs << ","
                          << "\"ratio\":" << ratio << ","
                          << "\"hits\":" << result.object_ids.size()
                          << "}";
            }
            std::cout << "]}\n";
            return 0;
        }

        if (command == "knn-coverage-scan") {
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_count = require_usize(args, "objects");
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto knn_k = require_usize(args, "knn-k");
            const bool knn_subgraph_admit = optional_bool(args, "knn-subgraph-admit", true);
            const bool knn_admit_exact_cap = optional_bool(args, "knn-admit-exact-cap", false);
            const auto layout_it = args.find("object-layout");
            const std::string object_layout = (layout_it == args.end()) ? "random" : layout_it->second;

            bag::SkeletonIndex index = SkeletonIndex::build(graph, subgraphs);
            const bool factorized_transfer = optional_bool(args, "factorized-transfer", true);
            const double factorized_arc_threshold = optional_double(args, "factorized-arc-threshold", 0.5);
            const auto factorized_border_threshold = optional_usize(args, "factorized-border-threshold", 12);
            if (factorized_transfer) {
                index.configure_factorized_transfer(factorized_arc_threshold, factorized_border_threshold);
            }
            const auto objects = build_objects_with_layout(graph, index, object_count, object_seed, object_layout);
            const auto queries = build_query_workload(graph, query_count, query_seed);

            std::cout << "{"
                      << "\"query_count\":" << queries.size() << ","
                      << "\"query_seed\":" << query_seed << ","
                      << "\"object_count\":" << object_count << ","
                      << "\"knn_k\":" << knn_k << ","
                      << "\"rows\":[";
            for (std::size_t i = 0; i < queries.size(); ++i) {
                const auto& q = queries[i];
                const auto weight = graph.get_weight(q.edge.first, q.edge.second).value_or(0);
                const auto result = index.knn_query(
                    q,
                    knn_k,
                    objects,
                    i,
                    nullptr,
                    nullptr,
                    false,
                    false,
                    0,
                    factorized_transfer,
                    false,
                    knn_subgraph_admit,
                    knn_admit_exact_cap
                );
                const auto denom = result.candidate_subgraphs;
                const auto ratio =
                    (denom == 0) ? 0.0
                                 : static_cast<double>(result.admitted_subgraphs) /
                                       static_cast<double>(denom);
                if (i != 0) {
                    std::cout << ",";
                }
                std::cout << "{"
                          << "\"query_id\":" << i << ","
                          << "\"u\":" << q.edge.first << ","
                          << "\"v\":" << q.edge.second << ","
                          << "\"offset\":" << q.offset << ","
                          << "\"edge_weight\":" << weight << ","
                          << "\"admitted_subgraphs\":" << result.admitted_subgraphs << ","
                          << "\"candidate_subgraphs\":" << result.candidate_subgraphs << ","
                          << "\"ratio\":" << ratio << ","
                          << "\"admitted_objects\":" << result.admitted_objects << ","
                          << "\"returned\":" << result.items.size() << ","
                          << "\"visited_subgraphs\":" << result.visited_subgraphs
                          << "}";
            }
            std::cout << "]}\n";
            return 0;
        }

        if (command == "diagnostics") {
            bag::DiagnosticsOptions diagnostics_options;
            diagnostics_options.graph_name = std::filesystem::path(path).filename().string();
            diagnostics_options.impl_version = "cpp-current-diagnostics-v2";
            diagnostics_options.theta = theta;
            diagnostics_options.seed_id = partition_seed;
            diagnostics_options.paper_strict = paper_strict;
            diagnostics_options.shortcut_repartition = shortcut_repartition;
            diagnostics_options.output_dir = diagnostics_output_dir;
            const auto query_count = optional_usize(args, "query-count", 0);
            if (query_count == 0) {
                bag::write_partition_diagnostics(graph, subgraphs, partition_options, diagnostics_options);
            } else {
                const auto fc_rule = parse_fc_rule(args);
                const auto range_radius =
                    static_cast<EdgeWeight>(optional_usize(args, "range-radius", 5000));
                const auto knn_k = optional_usize(args, "knn-k", 10);
                const auto object_count = optional_usize(args, "objects", 1000);
                const auto object_seed =
                    static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
                const auto query_seed =
                    static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));

                bag::SkeletonIndex index = SkeletonIndex::build(graph, subgraphs);
                const auto objects = build_objects(graph, index, object_count, object_seed);
                const auto queries = build_query_workload(graph, query_count, query_seed);
                const auto& raw_objects = objects.objects();

                std::vector<bag::BorderExposureEvent> exposure_events;
                exposure_events.reserve(query_count * 32U);
                std::vector<bag::CliqueRowShadowRecord> row_shadow_records;
                row_shadow_records.reserve(query_count * 64U);
                std::vector<std::vector<bag::ObjId>> baseline_range_results(queries.size());
                std::vector<std::vector<bag::KnnItem>> baseline_knn_results(queries.size());
                std::vector<bag::QueryRuntimeSample> runtime_samples;
                runtime_samples.reserve(query_count * 2U);
                for (std::size_t i = 0; i < queries.size(); ++i) {
                    runtime_samples.push_back(bag::QueryRuntimeSample{
                        i,
                        bag::QueryType::Range,
                        bag::calc_execution_time_in_us([&] {
                                baseline_range_results[i] =
                                    index.range_query(
                                        queries[i],
                                        range_radius,
                                        objects,
                                        fc_rule,
                                        i,
                                        &exposure_events,
                                        &row_shadow_records,
                                        true,
                                        false
                                    ).object_ids;
                        }),
                    });
                    runtime_samples.push_back(bag::QueryRuntimeSample{
                        i,
                        bag::QueryType::Knn,
                        bag::calc_execution_time_in_us([&] {
                                baseline_knn_results[i] =
                                    index.knn_query(
                                        queries[i],
                                        knn_k,
                                        objects,
                                        i,
                                        &exposure_events,
                                        &row_shadow_records,
                                        false,
                                        false,
                                        0,
                                        false
                                    ).items;
                        }),
                    });
                }

                const auto touch_summary = bag::summarize_query_exposures(exposure_events);
                const auto exact_verdicts = bag::evaluate_exact_demotion_candidates(
                    graph,
                    subgraphs,
                    bag::DiagnosticsWorkloadData{
                        &queries,
                        &raw_objects,
                        &baseline_range_results,
                        &baseline_knn_results,
                        &exposure_events,
                        &runtime_samples,
                        fc_rule,
                        range_radius,
                        knn_k,
                        optional_usize(args, "exact-demotion-limit", 0),
                    }
                );
                bag::write_query_exposure_csv(
                    diagnostics_options.output_dir / "query_border_exposure.csv",
                    diagnostics_options,
                    exposure_events
                );
                bag::write_clique_row_shadow_csv(
                    diagnostics_options.output_dir / "first_entry_clique_rows.csv",
                    diagnostics_options,
                    row_shadow_records
                );
                bag::write_clique_row_shadow_summary_csv(
                    diagnostics_options.output_dir / "first_entry_clique_summary.csv",
                    diagnostics_options,
                    row_shadow_records
                );
                bag::write_skeleton_hotspots_csv(
                    diagnostics_options.output_dir / "skeleton_hotspots.csv",
                    graph,
                    subgraphs,
                    exposure_events,
                    runtime_samples
                );
                bag::write_query_runtime_samples_csv(
                    diagnostics_options.output_dir / "query_runtime_samples.csv",
                    runtime_samples
                );
                bag::write_query_runtime_summary_csv(
                    diagnostics_options.output_dir / "query_runtime_summary.csv",
                    runtime_samples
                );
                bag::write_partition_diagnostics(
                    graph,
                    subgraphs,
                    partition_options,
                    diagnostics_options,
                    &touch_summary,
                    &exact_verdicts
                );
            }
            std::cout
                << "{"
                << "\"output_dir\":\"" << diagnostics_options.output_dir.generic_string() << "\","
                << "\"subgraphs\":" << subgraphs.size() << ","
                << "\"partition_us\":" << partition_us;
            append_partition_checkpoint_json_fields(
                std::cout,
                partition_checkpoint_mode,
                partition_options.checkpoint_path,
                partition_checkpoint_state
            );
            append_partition_cache_json_fields(std::cout, partition_cache_state);
            std::cout << "}\n";
            return 0;
        }

        bag::SkeletonIndex index = SkeletonIndex::build(graph, subgraphs);
        const bool factorized_transfer = optional_bool(args, "factorized-transfer", command == "knn");
        const double factorized_arc_threshold = optional_double(args, "factorized-arc-threshold", 0.5);
        const auto factorized_border_threshold = optional_usize(args, "factorized-border-threshold", 12);
        if (factorized_transfer) {
            index.configure_factorized_transfer(factorized_arc_threshold, factorized_border_threshold);
        }
        const auto query = load_query_point(args);
        if (command == "frontier") {
            const auto radii = parse_weight_list(args, "radii", {1000, 2000, 4000, 8000, 16000, 32000});
            std::cout << "{"
                      << "\"query_u\":" << query.edge.first << ","
                      << "\"query_v\":" << query.edge.second << ","
                      << "\"offset\":" << query.offset << ","
                      << "\"radii\":[";
            for (std::size_t i = 0; i < radii.size(); ++i) {
                const auto stats = index.frontier_boundary_stats(query, radii[i]);
                if (i != 0) {
                    std::cout << ",";
                }
                std::cout << "{"
                          << "\"radius\":" << stats.radius << ","
                          << "\"reached_boundaries\":" << stats.reached_boundaries << ","
                          << "\"frontier_boundaries\":" << stats.frontier_boundaries << ","
                          << "\"outward_edges\":" << stats.outward_edges
                          << "}";
            }
            std::cout << "]}\n";
            return 0;
        }
        const auto objects = build_objects(graph, index, args);
        const auto repeat = optional_usize(args, "repeat", 1);

        if (command == "range") {
            const auto radius = require_weight(args, "radius");
            const auto fc_rule = parse_fc_rule(args);
            const bool range_row_truncation = optional_bool(args, "range-row-truncation", true);
            const bool range_whole_subgraph_acceptance =
                optional_bool(args, "range-whole-subgraph-acceptance", true);
            const bool compare_baseline = optional_bool(args, "compare-baseline", false);
            long long baseline_range_us = 0;
            bag::RangeQueryResult result;
            std::vector<long long> range_times;
            range_times.reserve(repeat);
            for (std::size_t i = 0; i < repeat; ++i) {
                range_times.push_back(bag::calc_execution_time_in_us([&] {
                    result = index.range_query(
                        query,
                        radius,
                        objects,
                        fc_rule,
                        0,
                        nullptr,
                        nullptr,
                        range_row_truncation,
                        factorized_transfer,
                        range_whole_subgraph_acceptance
                    );
                }));
            }
            const auto range_us = *std::min_element(range_times.begin(), range_times.end());
            const auto avg_range_us =
                static_cast<double>(std::accumulate(range_times.begin(), range_times.end(), 0LL)) /
                static_cast<double>(range_times.size());
            bool baseline_match = true;
            if (compare_baseline) {
                bag::RangeQueryResult baseline;
                baseline_range_us = bag::calc_execution_time_in_us([&] {
                    baseline = index.range_query(
                        query,
                        radius,
                        objects,
                        fc_rule,
                        0,
                        nullptr,
                        nullptr,
                        false,
                        false,
                        range_whole_subgraph_acceptance
                    );
                });
                baseline_match = baseline.object_ids == result.object_ids;
            }
            const bool verify = optional_bool(args, "verify", false);
            bool exact = true;
            long long exact_us = 0;
            if (verify) {
                std::vector<bag::ObjId> exact_ids;
                exact_us = bag::calc_execution_time_in_us([&] {
                    exact_ids = index.exact_range_query(query, radius, objects);
                });
                std::unordered_set<bag::ObjId> lhs(result.object_ids.begin(), result.object_ids.end());
                std::unordered_set<bag::ObjId> rhs(exact_ids.begin(), exact_ids.end());
                exact = lhs == rhs;
            }
            std::cout
                << "{"
                << "\"hits\":" << result.object_ids.size() << ","
                << "\"fc_subgraphs\":" << result.fc_subgraphs << ","
                << "\"pc_subgraphs\":" << result.pc_subgraphs << ","
                << "\"result_subgraphs\":" << (result.fc_subgraphs + result.pc_subgraphs) << ","
                << "\"auto_included_objects\":" << result.auto_included_objects << ","
                << "\"exact_checked_objects\":" << result.exact_checked_objects << ","
                << "\"initial_local_vertices\":" << result.initial_local_vertices << ","
                << "\"boundary_vertices_reached\":" << result.boundary_vertices_reached << ","
                << "\"touched_subgraphs\":" << result.touched_subgraphs << ","
                << "\"num_clique_relax_attempts\":" << result.num_clique_relax_attempts << ","
                << "\"num_successful_clique_relaxes\":" << result.num_successful_clique_relaxes << ","
                << "\"num_pq_pushes_from_clique\":" << result.num_pq_pushes_from_clique << ","
                << "\"num_rows_truncated\":" << result.num_rows_truncated << ","
                << "\"num_exits_skipped_by_truncation\":" << result.num_exits_skipped_by_truncation << ","
                << "\"num_redundant_subgraph_reentries\":" << result.num_redundant_subgraph_reentries << ","
                << "\"num_useful_subgraph_reentries\":" << result.num_useful_subgraph_reentries << ","
                << "\"factorized_rows_used\":" << result.factorized_rows_used << ","
                << "\"factorized_hubs_used\":" << result.factorized_hubs_used << ","
                << "\"factorized_exits_emitted\":" << result.factorized_exits_emitted << ","
                << "\"repeat\":" << repeat << ","
                << "\"range_us\":" << range_us << ","
                << "\"avg_range_us\":" << avg_range_us << ","
                << "\"baseline_range_us\":" << baseline_range_us << ","
                << "\"exact_us\":" << exact_us << ","
                << "\"exact\":" << (exact ? "true" : "false") << ","
                << "\"baseline_match\":" << (baseline_match ? "true" : "false") << ","
                << "\"range_row_truncation\":" << (range_row_truncation ? "true" : "false") << ","
                << "\"range_whole_subgraph_acceptance\":"
                << (range_whole_subgraph_acceptance ? "true" : "false") << ","
                << "\"factorized_transfer\":" << (factorized_transfer ? "true" : "false")
                << "}\n";
            return (exact && baseline_match) ? 0 : 2;
        }

        if (command == "knn") {
            const auto k = require_usize(args, "k");
            const bool knn_streamed_clique = optional_bool(args, "knn-streamed-clique", false);
            const bool knn_safe_coverage_shadow = optional_bool(args, "knn-safe-coverage-shadow", false);
            const auto knn_parent_shadow_size = optional_usize(args, "knn-parent-shadow-size", 0);
            const bool knn_subgraph_admit = optional_bool(args, "knn-subgraph-admit", false);
            const bool knn_admit_exact_cap = optional_bool(args, "knn-admit-exact-cap", false);
            const bool compare_baseline = optional_bool(args, "compare-baseline", false);
            long long baseline_knn_us = 0;
            bag::KnnQueryResult result;
            std::vector<long long> knn_times;
            knn_times.reserve(repeat);
            for (std::size_t i = 0; i < repeat; ++i) {
                knn_times.push_back(bag::calc_execution_time_in_us([&] {
                    result = index.knn_query(
                        query,
                        k,
                        objects,
                        0,
                        nullptr,
                        nullptr,
                        knn_streamed_clique,
                        knn_safe_coverage_shadow,
                        knn_parent_shadow_size,
                        factorized_transfer,
                        false,
                        knn_subgraph_admit,
                        knn_admit_exact_cap
                    );
                }));
            }
            const auto knn_us = *std::min_element(knn_times.begin(), knn_times.end());
            const auto avg_knn_us =
                static_cast<double>(std::accumulate(knn_times.begin(), knn_times.end(), 0LL)) /
                static_cast<double>(knn_times.size());
            bool baseline_match = true;
            if (compare_baseline) {
                bag::KnnQueryResult baseline;
                baseline_knn_us = bag::calc_execution_time_in_us([&] {
                    baseline = index.knn_query(
                        query,
                        k,
                        objects,
                        0,
                        nullptr,
                        nullptr,
                        false,
                        knn_safe_coverage_shadow,
                        knn_parent_shadow_size,
                        false,
                        false,
                        false
                    );
                });
                if (result.items.size() != baseline.items.size()) {
                    baseline_match = false;
                } else {
                    for (std::size_t i = 0; i < result.items.size(); ++i) {
                        if (result.items[i].id != baseline.items[i].id ||
                            result.items[i].distance != baseline.items[i].distance) {
                            baseline_match = false;
                            break;
                        }
                    }
                }
            }
            const bool verify = optional_bool(args, "verify", false);
            bool exact = true;
            long long exact_us = 0;
            if (verify) {
                bag::KnnQueryResult exact_result;
                exact_us = bag::calc_execution_time_in_us([&] {
                    exact_result = index.exact_knn_query(query, k, objects);
                });
                if (result.items.size() != exact_result.items.size()) {
                    exact = false;
                } else {
                    for (std::size_t i = 0; i < result.items.size(); ++i) {
                        if (result.items[i].id != exact_result.items[i].id ||
                            result.items[i].distance != exact_result.items[i].distance) {
                            exact = false;
                            break;
                        }
                    }
                }
            }
            std::cout
                << "{"
                << "\"k\":" << k << ","
                << "\"returned\":" << result.items.size() << ","
                << "\"visited_boundaries\":" << result.visited_boundaries << ","
                << "\"visited_subgraphs\":" << result.visited_subgraphs << ","
                << "\"candidates_considered\":" << result.candidates_considered << ","
                << "\"candidate_subgraphs\":" << result.candidate_subgraphs << ","
                << "\"final_candidates\":" << result.final_candidates << ","
                << "\"exact_evaluated\":" << result.exact_evaluated << ","
                << "\"exact_from_admitted\":" << result.exact_from_admitted << ","
                << "\"exact_from_unresolved\":" << result.exact_from_unresolved << ","
                << "\"exact_reused_from_admitted_cache\":" << result.exact_reused_from_admitted_cache << ","
                << "\"admitted_subgraphs\":" << result.admitted_subgraphs << ","
                << "\"admitted_objects\":" << result.admitted_objects << ","
                << "\"vertex_fast_path\":" << (result.vertex_fast_path ? "true" : "false") << ","
                << "\"subgraph_admit\":" << (result.subgraph_admit ? "true" : "false") << ","
                << "\"init_us\":" << result.init_us << ","
                << "\"explore_us\":" << result.explore_us << ","
                << "\"finalize_us\":" << result.finalize_us << ","
                << "\"pq_us\":" << result.pq_us << ","
                << "\"membership_us\":" << result.membership_us << ","
                << "\"subgraph_bookkeeping_us\":" << result.subgraph_bookkeeping_us << ","
                << "\"clique_emit_us\":" << result.clique_emit_us << ","
                << "\"num_clique_relax_attempts\":" << result.num_clique_relax_attempts << ","
                << "\"num_successful_clique_relaxes\":" << result.num_successful_clique_relaxes << ","
                << "\"num_pq_pushes_from_clique\":" << result.num_pq_pushes_from_clique << ","
                << "\"num_redundant_subgraph_reentries\":" << result.num_redundant_subgraph_reentries << ","
                << "\"num_useful_subgraph_reentries\":" << result.num_useful_subgraph_reentries << ","
                << "\"boundary_pq_pushes\":" << result.boundary_pq_pushes << ","
                << "\"boundary_pq_pops\":" << result.boundary_pq_pops << ","
                << "\"row_state_pushes\":" << result.row_state_pushes << ","
                << "\"row_state_pops\":" << result.row_state_pops << ","
                << "\"streamed_rows_started\":" << result.streamed_rows_started << ","
                << "\"streamed_row_emissions\":" << result.streamed_row_emissions << ","
                << "\"streamed_rows_stopped_by_threshold\":" << result.streamed_rows_stopped_by_threshold << ","
                << "\"streamed_exits_skipped_by_threshold\":" << result.streamed_exits_skipped_by_threshold << ","
                << "\"peak_frontier_queue_size\":" << result.peak_frontier_queue_size << ","
                << "\"safe_coverage_candidate_count\":" << result.safe_coverage_candidate_count << ","
                << "\"first_k_candidates_boundary_visit_order\":" << result.first_k_candidates_boundary_visit_order << ","
                << "\"first_finite_safe_coverage_radius\":" << result.first_finite_safe_coverage_radius << ","
                << "\"first_finite_safe_coverage_boundary_visit_order\":" << result.first_finite_safe_coverage_boundary_visit_order << ","
                << "\"final_safe_coverage_radius\":" << result.final_safe_coverage_radius << ","
                << "\"safe_coverage_updates\":" << result.safe_coverage_updates << ","
                << "\"first_finite_tau\":" << result.first_finite_tau << ","
                << "\"first_finite_tau_boundary_visit_order\":" << result.first_finite_tau_boundary_visit_order << ","
                << "\"final_tau\":" << result.final_tau << ","
                << "\"tau_updates\":" << result.tau_updates << ","
                << "\"would_stop_rows_by_safe_coverage\":" << result.would_stop_rows_by_safe_coverage << ","
                << "\"would_skip_exits_by_safe_coverage\":" << result.would_skip_exits_by_safe_coverage << ","
                << "\"would_save_pq_pushes_by_safe_coverage\":" << result.would_save_pq_pushes_by_safe_coverage << ","
                << "\"parent_regions_touched\":" << result.parent_regions_touched << ","
                << "\"parent_region_size\":" << result.parent_region_size << ","
                << "\"max_children_descended_in_parent\":" << result.max_children_descended_in_parent << ","
                << "\"single_child_parent_regions\":" << result.single_child_parent_regions << ","
                << "\"avg_children_descended_per_parent\":" << result.avg_children_descended_per_parent << ","
                << "\"parent_descent_ratio\":" << result.parent_descent_ratio << ","
                << "\"factorized_rows_used\":" << result.factorized_rows_used << ","
                << "\"factorized_hubs_used\":" << result.factorized_hubs_used << ","
                << "\"factorized_exits_emitted\":" << result.factorized_exits_emitted << ","
                << "\"repeat\":" << repeat << ","
                << "\"knn_us\":" << knn_us << ","
                << "\"avg_knn_us\":" << avg_knn_us << ","
                << "\"baseline_knn_us\":" << baseline_knn_us << ","
                << "\"exact_us\":" << exact_us << ","
                << "\"exact\":" << (exact ? "true" : "false") << ","
                << "\"baseline_match\":" << (baseline_match ? "true" : "false") << ","
                << "\"streamed_clique\":" << (result.streamed_clique ? "true" : "false") << ","
                << "\"safe_coverage_shadow\":" << (knn_safe_coverage_shadow ? "true" : "false") << ","
                << "\"knn_admit_exact_cap\":" << (knn_admit_exact_cap ? "true" : "false") << ","
                << "\"factorized_transfer\":" << (factorized_transfer ? "true" : "false")
                << "}\n";
            return (exact && baseline_match) ? 0 : 2;
        }

        print_usage();
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
