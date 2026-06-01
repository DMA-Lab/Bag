#include "diagnostics.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace bag {

namespace {

struct BorderSlackStats {
    double rb{0.0};
    double ib{0.0};
    double slack{0.0};
    std::size_t witness_count{0};
};

struct BorderTopologyStats {
    bool has_external_neighbor{false};
    std::size_t degree_total{0};
    std::size_t degree_inside_subgraph{0};
    std::size_t degree_outside_subgraph{0};
    std::size_t missing_incident_edges_inside{0};
    std::size_t missing_vertices_inside{0};
};

struct DemotionAuditRow {
    bool feasible_connectivity{false};
    bool feasible_br{false};
    std::string feasible_exact_range{"skipped"};
    std::string feasible_exact_knn{"skipped"};
    std::string first_failed_condition{"none"};
    std::size_t min_local_repair_edges{0};
    std::size_t min_local_repair_vertices{0};
    std::size_t affected_internal_vertices_count{0};
    std::size_t affected_border_vertices_count{0};
};

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (const auto ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += "\"";
    return escaped;
}

template <typename T>
std::string to_csv(T value) {
    std::ostringstream oss;
    oss << value;
    return csv_escape(oss.str());
}

void write_csv_row(std::ofstream& out, const std::vector<std::string>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "\n";
}

std::unordered_set<VertexId> subgraph_vertex_set(const Subgraph& sg) {
    std::unordered_set<VertexId> vertices;
    vertices.reserve(sg.graph.size() * 2U + 1U);
    for (const auto v : sg.graph.vertices()) {
        vertices.insert(v);
    }
    return vertices;
}

std::unordered_map<VertexId, std::size_t> insertion_steps(const Subgraph& sg) {
    std::unordered_map<VertexId, std::size_t> steps;
    steps.reserve(sg.insertion_order.size() * 2U + 1U);
    if (sg.seed_vertex != kInvalidVertex) {
        steps[sg.seed_vertex] = 0;
    }
    for (std::size_t i = 0; i < sg.insertion_order.size(); ++i) {
        steps.try_emplace(sg.insertion_order[i], i + 1U);
    }
    return steps;
}

BorderTopologyStats compute_border_topology(
    const Graph& global,
    const std::unordered_set<VertexId>& sg_vertices,
    VertexId vertex
) {
    BorderTopologyStats stats;
    stats.degree_total = global.neighbors(vertex).size();
    std::unordered_set<VertexId> outside_neighbors;
    for (const auto& [neighbor, weight] : global.neighbors(vertex)) {
        (void)weight;
        if (sg_vertices.contains(neighbor)) {
            ++stats.degree_inside_subgraph;
        } else {
            ++stats.degree_outside_subgraph;
            outside_neighbors.insert(neighbor);
        }
    }
    stats.has_external_neighbor = stats.degree_outside_subgraph != 0U;
    stats.missing_incident_edges_inside = stats.degree_outside_subgraph;
    stats.missing_vertices_inside = outside_neighbors.size();
    return stats;
}

BorderSlackStats compute_border_slack(const Subgraph& sg, VertexId boundary) {
    const auto edges = sg.graph.undirected_edges();
    HalfWeight rb{0, false};
    for (const auto other : sg.bound_vertices) {
        rb = std::max(rb, HalfWeight{sg.distance.get_or_inf(boundary, other), false});
    }

    HalfWeight ib{0, false};
    std::size_t witness_count = 0;
    for (const auto v : sg.internal_vertices) {
        const HalfWeight candidate{sg.distance.get_or_inf(boundary, v), false};
        if (candidate > ib) {
            ib = candidate;
            witness_count = 1;
        } else if (candidate == ib && candidate.to_double() > 0.0) {
            ++witness_count;
        }
    }
    for (const auto& [edge, weight] : edges) {
        const auto lhs = sg.distance.get_or_inf(boundary, edge.first);
        const auto rhs = sg.distance.get_or_inf(boundary, edge.second);
        if (lhs == kInfWeight || rhs == kInfWeight) {
            continue;
        }
        const auto candidate = distal_point_distance(lhs, rhs, weight);
        if (candidate > ib) {
            ib = candidate;
            witness_count = 1;
        } else if (candidate == ib && candidate.to_double() > 0.0) {
            ++witness_count;
        }
    }

    return BorderSlackStats{
        rb.to_double(),
        ib.to_double(),
        rb.to_double() - ib.to_double(),
        witness_count,
    };
}

bool compute_br_metrics(Subgraph& sg) {
    sg.rb_map.clear();
    if (sg.bound_vertices.empty()) {
        return false;
    }

    const auto edges = sg.graph.undirected_edges();
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

std::unordered_map<VertexId, double> compute_slack_map(const Subgraph& sg) {
    std::unordered_map<VertexId, double> slacks;
    slacks.reserve(sg.bound_vertices.size() * 2U + 1U);
    for (const auto b : sg.bound_vertices) {
        slacks[b] = compute_border_slack(sg, b).slack;
    }
    return slacks;
}

DemotionAuditRow audit_border_demotion(
    const Graph& global,
    const Subgraph& sg,
    VertexId vertex
) {
    const auto sg_vertices = subgraph_vertex_set(sg);
    const auto topology = compute_border_topology(global, sg_vertices, vertex);
    DemotionAuditRow row;
    row.min_local_repair_edges = topology.missing_incident_edges_inside;
    row.min_local_repair_vertices = topology.missing_vertices_inside;
    row.feasible_connectivity = sg.check_internal_vertex(global, vertex);
    if (!row.feasible_connectivity) {
        row.first_failed_condition = "connectivity";
        return row;
    }

    const auto before_slacks = compute_slack_map(sg);
    const auto before_nearest = sg.internal_to_nearest_border_dist;

    Subgraph candidate = sg;
    candidate.turn_to_internal(vertex);
    row.feasible_br = compute_br_metrics(candidate);
    if (!row.feasible_br) {
        row.first_failed_condition = "br";
        return row;
    }

    fill_internal_to_nearest_border(candidate);
    row.first_failed_condition = "exactness_skipped";

    for (const auto& [v, old_value] : before_nearest) {
        const auto it = candidate.internal_to_nearest_border_dist.find(v);
        const auto new_value = (it == candidate.internal_to_nearest_border_dist.end()) ? kInfWeight : it->second;
        if (new_value != old_value) {
            ++row.affected_internal_vertices_count;
        }
    }
    if (candidate.internal_to_nearest_border_dist.contains(vertex)) {
        ++row.affected_internal_vertices_count;
    }

    const auto after_slacks = compute_slack_map(candidate);
    std::unordered_set<VertexId> touched_boundaries;
    for (const auto& [b, slack] : before_slacks) {
        const auto it = after_slacks.find(b);
        if (it == after_slacks.end() || it->second != slack) {
            touched_boundaries.insert(b);
        }
    }
    for (const auto& [b, slack] : after_slacks) {
        (void)slack;
        if (!before_slacks.contains(b)) {
            touched_boundaries.insert(b);
        }
    }
    row.affected_border_vertices_count = touched_boundaries.size();
    return row;
}

std::size_t count_external_cut_edges(
    const Graph& global,
    const std::unordered_set<VertexId>& sg_vertices
) {
    std::unordered_set<std::uint64_t> unique_cut_edges;
    for (const auto u : sg_vertices) {
        for (const auto& [v, w] : global.neighbors(u)) {
            (void)w;
            if (sg_vertices.contains(v)) {
                continue;
            }
            const auto edge = ordered_edge(u, v);
            unique_cut_edges.insert(pack_pair(edge.first, edge.second));
        }
    }
    return unique_cut_edges.size();
}

double min_distance_to_other_borders(const Subgraph& sg, VertexId boundary) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto other : sg.bound_vertices) {
        if (other == boundary) {
            continue;
        }
        const auto distance = sg.distance.get_or_inf(boundary, other);
        if (distance != kInfWeight) {
            best = std::min(best, static_cast<double>(distance));
        }
    }
    return std::isfinite(best) ? best : -1.0;
}

void write_manifest(const DiagnosticsOptions& options) {
    std::ofstream out(options.output_dir / "diagnostics_manifest.md", std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open diagnostics manifest");
    }
    out
        << "# Partition Diagnostics\n\n"
        << "- graph_name: `" << options.graph_name << "`\n"
        << "- impl_version: `" << options.impl_version << "`\n"
        << "- theta: `" << options.theta << "`\n"
        << "- seed_id: `" << options.seed_id << "`\n"
        << "- paper_strict: `" << (options.paper_strict ? "true" : "false") << "`\n"
        << "- shortcut_repartition: `" << (options.shortcut_repartition ? "true" : "false") << "`\n\n"
        << "Notes:\n"
        << "- `discrepancy` is currently defined as `border_clique_edges - num_external_cut_edges`.\n"
        << "- `query_touch_count_*` and `first_touch_rank_*` are populated only when diagnostics are run with a query workload.\n"
        << "- `query_border_exposure.csv` is emitted only when a query workload is supplied.\n"
        << "- `demote_feasible_exact_range` and `demote_feasible_exact_knn` are workload-equivalence checks over the sampled diagnostics workload.\n"
        << "- `range_time_share` and `knn_time_share` in `skeleton_hotspots.csv` are allocated from per-query runtime in proportion to border-exposure events.\n";
}

bool same_knn_items(const std::vector<KnnItem>& lhs, const std::vector<KnnItem>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].id != rhs[i].id || lhs[i].distance != rhs[i].distance) {
            return false;
        }
    }
    return true;
}

struct ExactDemotionCandidate {
    SgId subgraph_id{0};
    VertexId vertex_id{kInvalidVertex};
    std::size_t touch_total{0};
    std::size_t local_repair_edges{0};
    std::size_t local_repair_vertices{0};
};

}  // namespace

QueryTouchSummary summarize_query_exposures(const std::vector<BorderExposureEvent>& events) {
    QueryTouchSummary summary;
    std::unordered_set<std::pair<std::size_t, SgId>, PairHash> seen_range_subgraphs;
    std::unordered_set<std::pair<std::size_t, SgId>, PairHash> seen_knn_subgraphs;

    for (const auto& event : events) {
        auto& border = summary.border_stats[std::make_pair(event.subgraph_id, event.border_vertex_id)];
        if (event.query_type == QueryType::Range) {
            ++border.range_touch_count;
            if (border.first_touch_rank_range < 0 ||
                static_cast<long long>(event.visit_order) < border.first_touch_rank_range) {
                border.first_touch_rank_range = static_cast<long long>(event.visit_order);
            }

            const auto subgraph_key = std::make_pair(event.query_id, event.subgraph_id);
            if (seen_range_subgraphs.insert(subgraph_key).second) {
                ++summary.subgraph_stats[event.subgraph_id].query_touch_count_range;
            }
        } else {
            ++border.knn_touch_count;
            if (border.first_touch_rank_knn < 0 ||
                static_cast<long long>(event.visit_order) < border.first_touch_rank_knn) {
                border.first_touch_rank_knn = static_cast<long long>(event.visit_order);
            }

            const auto subgraph_key = std::make_pair(event.query_id, event.subgraph_id);
            if (seen_knn_subgraphs.insert(subgraph_key).second) {
                ++summary.subgraph_stats[event.subgraph_id].query_touch_count_knn;
            }
        }
    }

    return summary;
}

void write_query_exposure_csv(
    const std::filesystem::path& output_path,
    const DiagnosticsOptions& diagnostics_options,
    const std::vector<BorderExposureEvent>& events
) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open query exposure csv");
    }

    write_csv_row(out, {
        "graph_name", "query_id", "query_type", "theta", "subgraph_id", "border_vertex_id",
        "visit_order", "was_needed_for_final_answer", "was_only_used_for_traversal",
        "triggered_full_cover", "tightened_upper_bound", "pruned_subgraph",
        "distance_from_query", "eventual_result_overlap"
    });

    for (const auto& event : events) {
        write_csv_row(out, {
            to_csv(diagnostics_options.graph_name),
            to_csv(event.query_id),
            to_csv(event.query_type == QueryType::Range ? "range" : "knn"),
            to_csv(diagnostics_options.theta),
            to_csv(event.subgraph_id),
            to_csv(event.border_vertex_id),
            to_csv(event.visit_order),
            to_csv(event.was_needed_for_final_answer ? 1 : 0),
            to_csv(event.was_only_used_for_traversal ? 1 : 0),
            to_csv(event.triggered_full_cover ? 1 : 0),
            to_csv(event.tightened_upper_bound ? 1 : 0),
            to_csv(event.pruned_subgraph ? 1 : 0),
            to_csv(event.distance_from_query),
            to_csv(event.eventual_result_overlap)
        });
    }
}

void write_clique_row_shadow_csv(
    const std::filesystem::path& output_path,
    const DiagnosticsOptions& diagnostics_options,
    const std::vector<CliqueRowShadowRecord>& records
) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open clique row shadow csv");
    }

    write_csv_row(out, {
        "graph_name", "theta_or_z", "query_id", "query_type", "subgraph_id", "entry_border",
        "entry_visit_order", "entry_distance", "threshold_at_entry", "num_exits_total",
        "num_exits_dist_dominated", "num_exits_threshold_dead", "num_pq_pushes_from_this_row",
        "num_pq_pushes_beyond_useful_prefix", "num_exits_useful", "num_useful_pushes_from_this_row",
        "useful_prefix_len"
    });

    for (const auto& record : records) {
        write_csv_row(out, {
            to_csv(diagnostics_options.graph_name),
            to_csv(diagnostics_options.theta),
            to_csv(record.query_id),
            to_csv(record.query_type == QueryType::Range ? "range" : "knn"),
            to_csv(record.subgraph_id),
            to_csv(record.entry_border),
            to_csv(record.entry_visit_order),
            to_csv(record.entry_distance),
            to_csv(record.threshold_at_entry == kInfWeight ? "inf" : to_csv(record.threshold_at_entry)),
            to_csv(record.num_exits_total),
            to_csv(record.num_exits_dist_dominated),
            to_csv(record.num_exits_threshold_dead),
            to_csv(record.num_pq_pushes_from_this_row),
            to_csv(record.num_pq_pushes_beyond_useful_prefix),
            to_csv(record.num_exits_useful),
            to_csv(record.num_useful_pushes_from_this_row),
            to_csv(record.useful_prefix_len)
        });
    }
}

void write_clique_row_shadow_summary_csv(
    const std::filesystem::path& output_path,
    const DiagnosticsOptions& diagnostics_options,
    const std::vector<CliqueRowShadowRecord>& records
) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open clique row shadow summary csv");
    }

    write_csv_row(out, {
        "graph_name", "theta_or_z", "query_type", "total_first_entry_rows", "total_exit_relax_attempts",
        "dist_dominated_ratio", "threshold_dead_ratio", "useful_exit_ratio",
        "would_save_clique_relax_attempts_if_drop_dist_dominated",
        "would_save_clique_relax_attempts_if_drop_threshold_dead",
        "would_save_pq_pushes_if_drop_dist_dominated",
        "would_save_pq_pushes_if_drop_threshold_dead",
        "avg_useful_prefix_len", "avg_useful_prefix_ratio"
    });

    for (const auto query_type : {QueryType::Range, QueryType::Knn}) {
        std::size_t total_rows = 0;
        std::size_t total_exits = 0;
        std::size_t total_dist_dominated = 0;
        std::size_t total_threshold_dead = 0;
        std::size_t total_useful = 0;
        std::size_t total_pushes = 0;
        std::size_t total_pushes_beyond_prefix = 0;
        std::size_t total_prefix_len = 0;
        for (const auto& record : records) {
            if (record.query_type != query_type) {
                continue;
            }
            ++total_rows;
            total_exits += record.num_exits_total;
            total_dist_dominated += record.num_exits_dist_dominated;
            total_threshold_dead += record.num_exits_threshold_dead;
            total_useful += record.num_exits_useful;
            total_pushes += record.num_pq_pushes_from_this_row;
            total_pushes_beyond_prefix += record.num_pq_pushes_beyond_useful_prefix;
            total_prefix_len += record.useful_prefix_len;
        }
        if (total_rows == 0) {
            continue;
        }
        write_csv_row(out, {
            to_csv(diagnostics_options.graph_name),
            to_csv(diagnostics_options.theta),
            to_csv(query_type == QueryType::Range ? "range" : "knn"),
            to_csv(total_rows),
            to_csv(total_exits),
            to_csv(total_exits == 0 ? 0.0 : static_cast<double>(total_dist_dominated) / static_cast<double>(total_exits)),
            to_csv(total_exits == 0 ? 0.0 : static_cast<double>(total_threshold_dead) / static_cast<double>(total_exits)),
            to_csv(total_exits == 0 ? 0.0 : static_cast<double>(total_useful) / static_cast<double>(total_exits)),
            to_csv(total_dist_dominated),
            to_csv(total_exits > total_prefix_len ? total_exits - total_prefix_len : 0),
            to_csv(0),
            to_csv(total_pushes_beyond_prefix),
            to_csv(static_cast<double>(total_prefix_len) / static_cast<double>(total_rows)),
            to_csv(total_exits == 0 ? 0.0 : static_cast<double>(total_prefix_len) / static_cast<double>(total_exits))
        });
    }
}

std::unordered_map<std::pair<SgId, VertexId>, ExactDemotionVerdict, PairHash> evaluate_exact_demotion_candidates(
    const Graph& global,
    const std::vector<Subgraph>& subgraphs,
    const DiagnosticsWorkloadData& workload
) {
    std::unordered_map<std::pair<SgId, VertexId>, ExactDemotionVerdict, PairHash> verdicts;
    if (workload.exact_demotion_limit == 0 || workload.queries == nullptr || workload.raw_objects == nullptr ||
        workload.baseline_range_results == nullptr || workload.baseline_knn_results == nullptr) {
        return verdicts;
    }

    QueryTouchSummary touch_summary;
    if (workload.exposure_events != nullptr) {
        touch_summary = summarize_query_exposures(*workload.exposure_events);
    }

    std::vector<ExactDemotionCandidate> candidates;
    for (const auto& sg : subgraphs) {
        const auto sg_vertices = subgraph_vertex_set(sg);
        for (const auto boundary : sg.bound_vertices) {
            const auto topology = compute_border_topology(global, sg_vertices, boundary);
            if (topology.has_external_neighbor) {
                continue;
            }
            const auto demotion = audit_border_demotion(global, sg, boundary);
            if (!demotion.feasible_connectivity || !demotion.feasible_br) {
                continue;
            }

            std::size_t touch_total = 0;
            if (const auto it = touch_summary.border_stats.find(std::make_pair(sg.id, boundary));
                it != touch_summary.border_stats.end()) {
                touch_total = it->second.range_touch_count + it->second.knn_touch_count;
            }

            candidates.push_back(ExactDemotionCandidate{
                sg.id,
                boundary,
                touch_total,
                demotion.min_local_repair_edges,
                demotion.min_local_repair_vertices,
            });
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.touch_total != rhs.touch_total) {
            return lhs.touch_total > rhs.touch_total;
        }
        if (lhs.local_repair_edges != rhs.local_repair_edges) {
            return lhs.local_repair_edges < rhs.local_repair_edges;
        }
        if (lhs.local_repair_vertices != rhs.local_repair_vertices) {
            return lhs.local_repair_vertices < rhs.local_repair_vertices;
        }
        if (lhs.subgraph_id != rhs.subgraph_id) {
            return lhs.subgraph_id < rhs.subgraph_id;
        }
        return lhs.vertex_id < rhs.vertex_id;
    });

    const auto limit = std::min(workload.exact_demotion_limit, candidates.size());
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& candidate_info = candidates[i];
        ExactDemotionVerdict verdict;
        verdict.feasible_exact_range = "true";
        verdict.feasible_exact_knn = "true";
        verdict.first_failed_condition = "none";

        auto candidate_subgraphs = subgraphs;
        auto& candidate_sg = candidate_subgraphs.at(candidate_info.subgraph_id);
        candidate_sg.turn_to_internal(candidate_info.vertex_id);
        compute_br_metrics(candidate_sg);
        fill_internal_to_nearest_border(candidate_sg);

        auto candidate_index = SkeletonIndex::build(global, std::move(candidate_subgraphs));
        const auto candidate_objects = IndexedMovingObjectSet::from_partition(
            *workload.raw_objects,
            candidate_index.edge_to_subgraph(),
            candidate_index.subgraphs()
        );

        for (std::size_t qid = 0; qid < workload.queries->size(); ++qid) {
            const auto candidate_range =
                candidate_index.range_query(
                    workload.queries->at(qid),
                    workload.range_radius,
                    candidate_objects,
                    workload.fc_rule
                ).object_ids;
            if (candidate_range != workload.baseline_range_results->at(qid)) {
                verdict.feasible_exact_range = "false";
                verdict.first_failed_condition = "exact_range";
                break;
            }
        }

        for (std::size_t qid = 0; qid < workload.queries->size(); ++qid) {
            const auto candidate_knn =
                candidate_index.knn_query(
                    workload.queries->at(qid),
                    workload.knn_k,
                    candidate_objects
                ).items;
            if (!same_knn_items(candidate_knn, workload.baseline_knn_results->at(qid))) {
                verdict.feasible_exact_knn = "false";
                if (verdict.first_failed_condition == "none") {
                    verdict.first_failed_condition = "exact_knn";
                }
                break;
            }
        }

        verdicts[std::make_pair(candidate_info.subgraph_id, candidate_info.vertex_id)] = std::move(verdict);
    }

    return verdicts;
}

void write_skeleton_hotspots_csv(
    const std::filesystem::path& output_path,
    const Graph& global,
    const std::vector<Subgraph>& subgraphs,
    const std::vector<BorderExposureEvent>& events,
    const std::vector<QueryRuntimeSample>& runtime_samples
) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open skeleton hotspots csv");
    }

    write_csv_row(out, {
        "subgraph_id",
        "num_vertices",
        "num_edges",
        "num_borders",
        "explicit_clique_edges",
        "external_cut_edges",
        "avg_skeleton_degree_contribution",
        "traversal_visits_from_queries",
        "range_time_share",
        "knn_time_share"
    });

    std::unordered_map<std::pair<std::size_t, int>, long long, PairHash> runtime_lookup;
    long long total_range_us = 0;
    long long total_knn_us = 0;
    for (const auto& sample : runtime_samples) {
        const auto type_id = (sample.query_type == QueryType::Range) ? 0 : 1;
        runtime_lookup[std::make_pair(sample.query_id, type_id)] = sample.elapsed_us;
        if (sample.query_type == QueryType::Range) {
            total_range_us += sample.elapsed_us;
        } else {
            total_knn_us += sample.elapsed_us;
        }
    }

    struct QueryBucket {
        std::size_t total_events{0};
        std::unordered_map<SgId, std::size_t> per_subgraph;
    };

    std::unordered_map<std::pair<std::size_t, int>, QueryBucket, PairHash> query_buckets;
    std::unordered_map<SgId, std::size_t> traversal_visits;
    for (const auto& event : events) {
        const auto type_id = (event.query_type == QueryType::Range) ? 0 : 1;
        auto& bucket = query_buckets[std::make_pair(event.query_id, type_id)];
        ++bucket.total_events;
        ++bucket.per_subgraph[event.subgraph_id];
        ++traversal_visits[event.subgraph_id];
    }

    std::unordered_map<SgId, double> allocated_range_time;
    std::unordered_map<SgId, double> allocated_knn_time;
    for (const auto& [key, bucket] : query_buckets) {
        if (bucket.total_events == 0) {
            continue;
        }
        const auto runtime_it = runtime_lookup.find(key);
        if (runtime_it == runtime_lookup.end()) {
            continue;
        }
        const auto query_type = key.second;
        for (const auto& [sg_id, count] : bucket.per_subgraph) {
            const auto share =
                static_cast<double>(runtime_it->second) * static_cast<double>(count) /
                static_cast<double>(bucket.total_events);
            if (query_type == 0) {
                allocated_range_time[sg_id] += share;
            } else {
                allocated_knn_time[sg_id] += share;
            }
        }
    }

    for (const auto& sg : subgraphs) {
        const auto sg_vertices = subgraph_vertex_set(sg);
        const auto num_edges = sg.graph.undirected_edges().size();
        const auto num_borders = sg.bound_vertices.size();
        const auto explicit_clique_edges = (num_borders * (num_borders - 1U)) / 2U;
        const auto external_cut_edges = count_external_cut_edges(global, sg_vertices);
        const auto avg_skeleton_degree_contribution =
            (num_borders == 0U) ? 0.0 :
            (2.0 * static_cast<double>(explicit_clique_edges)) / static_cast<double>(num_borders);
        const auto range_time_share =
            (total_range_us == 0) ? 0.0 : allocated_range_time[sg.id] / static_cast<double>(total_range_us);
        const auto knn_time_share =
            (total_knn_us == 0) ? 0.0 : allocated_knn_time[sg.id] / static_cast<double>(total_knn_us);

        write_csv_row(out, {
            to_csv(sg.id),
            to_csv(sg.graph.size()),
            to_csv(num_edges),
            to_csv(num_borders),
            to_csv(explicit_clique_edges),
            to_csv(external_cut_edges),
            to_csv(avg_skeleton_degree_contribution),
            to_csv(traversal_visits[sg.id]),
            to_csv(range_time_share),
            to_csv(knn_time_share)
        });
    }
}

void write_query_runtime_samples_csv(
    const std::filesystem::path& output_path,
    const std::vector<QueryRuntimeSample>& runtime_samples
) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open query runtime samples csv");
    }

    write_csv_row(out, {
        "query_id",
        "query_type",
        "elapsed_us"
    });

    for (const auto& sample : runtime_samples) {
        write_csv_row(out, {
            to_csv(sample.query_id),
            to_csv(sample.query_type == QueryType::Range ? "range" : "knn"),
            to_csv(sample.elapsed_us)
        });
    }
}

void write_query_runtime_summary_csv(
    const std::filesystem::path& output_path,
    const std::vector<QueryRuntimeSample>& runtime_samples
) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open query runtime summary csv");
    }

    write_csv_row(out, {
        "query_type",
        "count",
        "avg_us",
        "min_us",
        "max_us"
    });

    struct RuntimeBucket {
        std::size_t count{0};
        long long total{0};
        long long min{std::numeric_limits<long long>::max()};
        long long max{0};
    };

    RuntimeBucket range_bucket;
    RuntimeBucket knn_bucket;
    for (const auto& sample : runtime_samples) {
        auto& bucket = (sample.query_type == QueryType::Range) ? range_bucket : knn_bucket;
        ++bucket.count;
        bucket.total += sample.elapsed_us;
        bucket.min = std::min(bucket.min, sample.elapsed_us);
        bucket.max = std::max(bucket.max, sample.elapsed_us);
    }

    const auto write_bucket = [&](const char* label, const RuntimeBucket& bucket) {
        const double avg =
            (bucket.count == 0U) ? 0.0 : static_cast<double>(bucket.total) / static_cast<double>(bucket.count);
        write_csv_row(out, {
            to_csv(label),
            to_csv(bucket.count),
            to_csv(avg),
            to_csv(bucket.count == 0U ? 0LL : bucket.min),
            to_csv(bucket.count == 0U ? 0LL : bucket.max)
        });
    };

    write_bucket("range", range_bucket);
    write_bucket("knn", knn_bucket);
}

void write_partition_diagnostics(
    const Graph& global,
    const std::vector<Subgraph>& subgraphs,
    const PartitionOptions& partition_options,
    const DiagnosticsOptions& diagnostics_options,
    const QueryTouchSummary* touch_summary,
    const std::unordered_map<std::pair<SgId, VertexId>, ExactDemotionVerdict, PairHash>* exact_verdicts
) {
    std::filesystem::create_directories(diagnostics_options.output_dir);

    std::ofstream subgraph_out(diagnostics_options.output_dir / "subgraph_stats.csv", std::ios::trunc);
    std::ofstream border_out(diagnostics_options.output_dir / "border_vertex_diagnostics.csv", std::ios::trunc);
    std::ofstream demotion_out(diagnostics_options.output_dir / "border_demotion_audit.csv", std::ios::trunc);
    if (!subgraph_out || !border_out || !demotion_out) {
        throw std::runtime_error("failed to open diagnostics output files");
    }

    write_csv_row(subgraph_out, {
        "graph_name", "impl_version", "theta", "seed_id", "subgraph_id",
        "num_vertices", "num_edges", "num_border_vertices", "num_internal_vertices",
        "num_external_cut_edges", "border_clique_edges", "discrepancy",
        "min_BR_slack", "avg_BR_slack", "max_BR_slack",
        "num_forced_seed_borders", "num_forced_latest_added_borders",
        "num_true_interface_borders", "num_potentially_demotable_borders",
        "query_touch_count_range", "query_touch_count_knn"
    });

    write_csv_row(border_out, {
        "graph_name", "theta", "subgraph_id", "vertex_id", "time_added_step",
        "is_seed_border", "is_latest_added_border", "has_external_neighbor",
        "degree_total", "degree_inside_subgraph", "degree_outside_subgraph",
        "num_missing_incident_edges_inside", "BR_witness_count",
        "min_distance_to_other_borders", "query_touch_count_range",
        "query_touch_count_knn", "first_touch_rank_range", "first_touch_rank_knn",
        "can_demote_without_breaking_connectivity", "can_demote_without_breaking_BR",
        "local_fix_edges_needed_for_demotion", "local_fix_vertices_needed_for_demotion"
    });

    write_csv_row(demotion_out, {
        "graph_name", "theta_or_z", "subgraph_id", "vertex_id",
        "has_external_neighbor", "is_seed_border", "is_latest_added_border",
        "demote_feasible_connectivity", "demote_feasible_BR",
        "demote_feasible_exact_range", "demote_feasible_exact_knn", "first_failed_condition",
        "min_local_repair_edges", "min_local_repair_vertices",
        "affected_internal_vertices_count", "affected_border_vertices_count"
    });

    for (const auto& sg : subgraphs) {
        const auto sg_vertices = subgraph_vertex_set(sg);
        const auto insertion_step_map = insertion_steps(sg);
        const auto num_edges = sg.graph.undirected_edges().size();
        const auto num_external_cut_edges = count_external_cut_edges(global, sg_vertices);
        const auto border_clique_edges =
            (sg.bound_vertices.size() * (sg.bound_vertices.size() - 1U)) / 2U;
        const auto discrepancy =
            static_cast<long long>(border_clique_edges) - static_cast<long long>(num_external_cut_edges);

        std::size_t true_interface_borders = 0;
        std::size_t potentially_demotable_borders = 0;
        double min_br_slack = std::numeric_limits<double>::infinity();
        double max_br_slack = -std::numeric_limits<double>::infinity();
        double total_br_slack = 0.0;
        std::size_t query_touch_count_range = 0;
        std::size_t query_touch_count_knn = 0;
        if (touch_summary != nullptr) {
            if (const auto subgraph_touch_it = touch_summary->subgraph_stats.find(sg.id);
                subgraph_touch_it != touch_summary->subgraph_stats.end()) {
                query_touch_count_range = subgraph_touch_it->second.query_touch_count_range;
                query_touch_count_knn = subgraph_touch_it->second.query_touch_count_knn;
            }
        }

        const auto latest_added =
            sg.insertion_order.empty() ? kInvalidVertex : sg.insertion_order.back();

        for (const auto boundary : sg.bound_vertices) {
            const auto topology = compute_border_topology(global, sg_vertices, boundary);
            if (topology.has_external_neighbor) {
                ++true_interface_borders;
            }

            const auto slack = compute_border_slack(sg, boundary);
            min_br_slack = std::min(min_br_slack, slack.slack);
            max_br_slack = std::max(max_br_slack, slack.slack);
            total_br_slack += slack.slack;

            const auto demotion = audit_border_demotion(global, sg, boundary);
            if (demotion.feasible_connectivity && demotion.feasible_br) {
                ++potentially_demotable_borders;
            }
            std::string exact_range = "not_run";
            std::string exact_knn = "not_run";
            std::string first_failed_condition = demotion.first_failed_condition;
            if (demotion.feasible_connectivity && demotion.feasible_br) {
                first_failed_condition = "exact_not_checked";
                if (exact_verdicts != nullptr) {
                    if (const auto exact_it = exact_verdicts->find(std::make_pair(sg.id, boundary));
                        exact_it != exact_verdicts->end()) {
                        exact_range = exact_it->second.feasible_exact_range;
                        exact_knn = exact_it->second.feasible_exact_knn;
                        first_failed_condition = exact_it->second.first_failed_condition;
                    }
                }
            }
            std::size_t range_touch_count = 0;
            std::size_t knn_touch_count = 0;
            long long first_touch_rank_range = -1;
            long long first_touch_rank_knn = -1;
            if (touch_summary != nullptr) {
                if (const auto border_touch_it = touch_summary->border_stats.find(std::make_pair(sg.id, boundary));
                    border_touch_it != touch_summary->border_stats.end()) {
                    range_touch_count = border_touch_it->second.range_touch_count;
                    knn_touch_count = border_touch_it->second.knn_touch_count;
                    first_touch_rank_range = border_touch_it->second.first_touch_rank_range;
                    first_touch_rank_knn = border_touch_it->second.first_touch_rank_knn;
                }
            }

            write_csv_row(border_out, {
                to_csv(diagnostics_options.graph_name),
                to_csv(diagnostics_options.theta),
                to_csv(sg.id),
                to_csv(boundary),
                to_csv(insertion_step_map.contains(boundary) ? insertion_step_map.at(boundary) : 0U),
                to_csv(boundary == sg.seed_vertex ? 1 : 0),
                to_csv(boundary == latest_added ? 1 : 0),
                to_csv(topology.has_external_neighbor ? 1 : 0),
                to_csv(topology.degree_total),
                to_csv(topology.degree_inside_subgraph),
                to_csv(topology.degree_outside_subgraph),
                to_csv(topology.missing_incident_edges_inside),
                to_csv(slack.witness_count),
                to_csv(min_distance_to_other_borders(sg, boundary)),
                to_csv(range_touch_count),
                to_csv(knn_touch_count),
                to_csv(first_touch_rank_range),
                to_csv(first_touch_rank_knn),
                to_csv(demotion.feasible_connectivity ? 1 : 0),
                to_csv(demotion.feasible_br ? 1 : 0),
                to_csv(demotion.min_local_repair_edges),
                to_csv(demotion.min_local_repair_vertices)
            });

            write_csv_row(demotion_out, {
                to_csv(diagnostics_options.graph_name),
                to_csv(diagnostics_options.theta),
                to_csv(sg.id),
                to_csv(boundary),
                to_csv(topology.has_external_neighbor ? "true" : "false"),
                to_csv(boundary == sg.seed_vertex ? "true" : "false"),
                to_csv(boundary == latest_added ? "true" : "false"),
                to_csv(demotion.feasible_connectivity ? "true" : "false"),
                to_csv(demotion.feasible_br ? "true" : "false"),
                to_csv(exact_range),
                to_csv(exact_knn),
                to_csv(first_failed_condition),
                to_csv(demotion.min_local_repair_edges),
                to_csv(demotion.min_local_repair_vertices),
                to_csv(demotion.affected_internal_vertices_count),
                to_csv(demotion.affected_border_vertices_count)
            });
        }

        const auto avg_br_slack =
            sg.bound_vertices.empty() ? 0.0 : total_br_slack / static_cast<double>(sg.bound_vertices.size());

        write_csv_row(subgraph_out, {
            to_csv(diagnostics_options.graph_name),
            to_csv(diagnostics_options.impl_version),
            to_csv(diagnostics_options.theta),
            to_csv(diagnostics_options.seed_id),
            to_csv(sg.id),
            to_csv(sg.graph.size()),
            to_csv(num_edges),
            to_csv(sg.bound_vertices.size()),
            to_csv(sg.internal_vertices.size()),
            to_csv(num_external_cut_edges),
            to_csv(border_clique_edges),
            to_csv(discrepancy),
            to_csv(sg.bound_vertices.empty() ? 0.0 : min_br_slack),
            to_csv(avg_br_slack),
            to_csv(sg.bound_vertices.empty() ? 0.0 : max_br_slack),
            to_csv(sg.bound_vertices.contains(sg.seed_vertex) ? 1 : 0),
            to_csv(
                partition_options.paper_strict_mode && latest_added != kInvalidVertex &&
                sg.bound_vertices.contains(latest_added) ? 1 : 0
            ),
            to_csv(true_interface_borders),
            to_csv(potentially_demotable_borders),
            to_csv(query_touch_count_range),
            to_csv(query_touch_count_knn)
        });
    }

    write_manifest(diagnostics_options);
}

}  // namespace bag
