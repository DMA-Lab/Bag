#include "factorized_transfer.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <unordered_set>

namespace bag {

namespace {

struct BorderPair {
    VertexId lhs{kInvalidVertex};
    VertexId rhs{kInvalidVertex};
};

struct PairAssignment {
    VertexId lhs{kInvalidVertex};
    VertexId rhs{kInvalidVertex};
    VertexId hub{kInvalidVertex};
};

struct HubCoverage {
    VertexId hub{kInvalidVertex};
    bool is_internal{false};
    std::vector<std::size_t> covered_pairs;
};

struct AnalyzedSubgraph {
    FactorizedTransferSubgraphStats stats;
    FactorizedTransferSubgraphModel model;
    std::vector<PairAssignment> assignments;
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

bool exact_via_hub(const Subgraph& sg, VertexId lhs, VertexId rhs, VertexId hub) {
    const auto d_lr = sg.distance.get_or_inf(lhs, rhs);
    const auto d_lh = sg.distance.get_or_inf(lhs, hub);
    const auto d_hr = sg.distance.get_or_inf(hub, rhs);
    if (d_lr == kInfWeight || d_lh == kInfWeight || d_hr == kInfWeight) {
        return false;
    }
    if (d_lh > kInfWeight - d_hr) {
        return false;
    }
    return static_cast<EdgeWeight>(d_lh + d_hr) == d_lr;
}

AnalyzedSubgraph analyze_subgraph(const Subgraph& sg) {
    AnalyzedSubgraph analyzed;
    auto& stats = analyzed.stats;
    auto& model = analyzed.model;
    stats.subgraph_id = sg.id;
    stats.num_vertices = sg.graph.size();
    stats.num_borders = sg.bound_vertices.size();
    model.subgraph_id = sg.id;
    model.num_borders = stats.num_borders;
    if (stats.num_borders <= 1U) {
        return analyzed;
    }

    std::vector<VertexId> borders(sg.bound_vertices.begin(), sg.bound_vertices.end());
    std::sort(borders.begin(), borders.end());
    const auto vertices = sg.graph.vertices();

    std::vector<BorderPair> pairs;
    for (std::size_t i = 0; i < borders.size(); ++i) {
        for (std::size_t j = i + 1U; j < borders.size(); ++j) {
            pairs.push_back(BorderPair{borders[i], borders[j]});
        }
    }

    stats.explicit_directed_arcs = borders.size() * (borders.size() - 1U);
    stats.unordered_border_pairs = pairs.size();
    model.explicit_directed_arcs = stats.explicit_directed_arcs;
    if (pairs.empty()) {
        return analyzed;
    }

    std::vector<std::size_t> witness_counts(pairs.size(), 0U);
    std::vector<HubCoverage> candidate_hubs;
    candidate_hubs.reserve(vertices.size());
    for (const auto hub : vertices) {
        HubCoverage coverage;
        coverage.hub = hub;
        coverage.is_internal = sg.internal_vertices.contains(hub);
        for (std::size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
            const auto& pair = pairs[pair_idx];
            if (exact_via_hub(sg, pair.lhs, pair.rhs, hub)) {
                coverage.covered_pairs.push_back(pair_idx);
                ++witness_counts[pair_idx];
            }
        }
        if (!coverage.covered_pairs.empty()) {
            candidate_hubs.push_back(std::move(coverage));
        }
    }

    stats.pair_min_witness_count = *std::min_element(witness_counts.begin(), witness_counts.end());
    stats.pair_max_witness_count = *std::max_element(witness_counts.begin(), witness_counts.end());
    stats.pair_avg_witness_count =
        static_cast<double>(std::accumulate(witness_counts.begin(), witness_counts.end(), std::size_t{0})) /
        static_cast<double>(witness_counts.size());

    std::vector<std::uint8_t> uncovered(pairs.size(), 1U);
    std::size_t uncovered_count = pairs.size();
    std::vector<std::size_t> selected_candidate_indices;
    selected_candidate_indices.reserve(std::min(candidate_hubs.size(), borders.size()));

    while (uncovered_count != 0U) {
        std::size_t best_idx = std::numeric_limits<std::size_t>::max();
        std::size_t best_gain = 0U;
        bool best_internal = false;
        VertexId best_hub = kInvalidVertex;
        for (std::size_t idx = 0; idx < candidate_hubs.size(); ++idx) {
            const auto& candidate = candidate_hubs[idx];
            std::size_t gain = 0U;
            for (const auto pair_idx : candidate.covered_pairs) {
                gain += uncovered[pair_idx] != 0U ? 1U : 0U;
            }
            if (gain == 0U) {
                continue;
            }
            if (gain > best_gain ||
                (gain == best_gain && candidate.is_internal && !best_internal) ||
                (gain == best_gain && candidate.is_internal == best_internal && candidate.hub < best_hub)) {
                best_idx = idx;
                best_gain = gain;
                best_internal = candidate.is_internal;
                best_hub = candidate.hub;
            }
        }
        if (best_idx == std::numeric_limits<std::size_t>::max()) {
            break;
        }
        selected_candidate_indices.push_back(best_idx);
        for (const auto pair_idx : candidate_hubs[best_idx].covered_pairs) {
            if (uncovered[pair_idx] != 0U) {
                uncovered[pair_idx] = 0U;
                --uncovered_count;
            }
        }
    }

    stats.greedy_hub_count = selected_candidate_indices.size();
    model.greedy_hub_count = stats.greedy_hub_count;
    for (const auto idx : selected_candidate_indices) {
        if (candidate_hubs[idx].is_internal) {
            ++stats.greedy_internal_hub_count;
            ++model.greedy_internal_hub_count;
        } else {
            ++stats.greedy_border_hub_count;
            ++model.greedy_border_hub_count;
        }
    }

    std::unordered_set<std::uint64_t> entry_to_hub_arcs;
    std::unordered_set<std::uint64_t> hub_to_exit_arcs;
    std::vector<std::unordered_set<VertexId>> entry_hubs(borders.size());
    analyzed.assignments.reserve(pairs.size());

    for (const auto& pair : pairs) {
        std::size_t best_idx = std::numeric_limits<std::size_t>::max();
        std::size_t best_increment = std::numeric_limits<std::size_t>::max();
        bool best_internal_choice = false;
        VertexId best_hub_choice = kInvalidVertex;

        const auto lhs_it = std::lower_bound(borders.begin(), borders.end(), pair.lhs);
        const auto rhs_it = std::lower_bound(borders.begin(), borders.end(), pair.rhs);
        const auto lhs_idx = static_cast<std::size_t>(std::distance(borders.begin(), lhs_it));
        const auto rhs_idx = static_cast<std::size_t>(std::distance(borders.begin(), rhs_it));

        for (const auto selected_idx : selected_candidate_indices) {
            const auto& hub_info = candidate_hubs[selected_idx];
            if (!exact_via_hub(sg, pair.lhs, pair.rhs, hub_info.hub)) {
                continue;
            }
            std::size_t increment = 0U;
            if (pair.lhs != hub_info.hub) {
                increment += entry_to_hub_arcs.contains(pack_pair(pair.lhs, hub_info.hub)) ? 0U : 1U;
                increment += hub_to_exit_arcs.contains(pack_pair(hub_info.hub, pair.lhs)) ? 0U : 1U;
            }
            if (pair.rhs != hub_info.hub) {
                increment += entry_to_hub_arcs.contains(pack_pair(pair.rhs, hub_info.hub)) ? 0U : 1U;
                increment += hub_to_exit_arcs.contains(pack_pair(hub_info.hub, pair.rhs)) ? 0U : 1U;
            }
            if (increment < best_increment ||
                (increment == best_increment && hub_info.is_internal && !best_internal_choice) ||
                (increment == best_increment && hub_info.is_internal == best_internal_choice &&
                 hub_info.hub < best_hub_choice)) {
                best_idx = selected_idx;
                best_increment = increment;
                best_internal_choice = hub_info.is_internal;
                best_hub_choice = hub_info.hub;
            }
        }

        if (best_idx == std::numeric_limits<std::size_t>::max()) {
            continue;
        }

        const auto hub = candidate_hubs[best_idx].hub;
        analyzed.assignments.push_back(PairAssignment{pair.lhs, pair.rhs, hub});
        if (pair.lhs != hub) {
            entry_to_hub_arcs.insert(pack_pair(pair.lhs, hub));
            hub_to_exit_arcs.insert(pack_pair(hub, pair.lhs));
            entry_hubs[lhs_idx].insert(hub);
        }
        if (pair.rhs != hub) {
            entry_to_hub_arcs.insert(pack_pair(pair.rhs, hub));
            hub_to_exit_arcs.insert(pack_pair(hub, pair.rhs));
            entry_hubs[rhs_idx].insert(hub);
        }
    }

    stats.factorized_entry_to_hub_arcs = entry_to_hub_arcs.size();
    stats.factorized_hub_to_exit_arcs = hub_to_exit_arcs.size();
    stats.factorized_directed_arcs = stats.factorized_entry_to_hub_arcs + stats.factorized_hub_to_exit_arcs;
    stats.factorized_arc_ratio =
        (stats.explicit_directed_arcs == 0U) ? 1.0 :
        static_cast<double>(stats.factorized_directed_arcs) / static_cast<double>(stats.explicit_directed_arcs);

    std::size_t max_entry_hubs = 0U;
    std::size_t total_entry_hubs = 0U;
    for (const auto& hubs : entry_hubs) {
        max_entry_hubs = std::max(max_entry_hubs, hubs.size());
        total_entry_hubs += hubs.size();
    }
    stats.avg_entry_hubs =
        static_cast<double>(total_entry_hubs) / static_cast<double>(entry_hubs.size());
    stats.max_entry_hubs = max_entry_hubs;
    stats.improves_over_explicit = stats.factorized_directed_arcs < stats.explicit_directed_arcs;
    model.factorized_directed_arcs = stats.factorized_directed_arcs;
    model.factorized_arc_ratio = stats.factorized_arc_ratio;
    model.feasible = !analyzed.assignments.empty();

    std::unordered_map<VertexId, std::unordered_map<VertexId, std::vector<std::pair<VertexId, EdgeWeight>>>> grouped_rows;
    for (const auto& assignment : analyzed.assignments) {
        const auto lhs_to_hub = sg.distance.get_or_inf(assignment.lhs, assignment.hub);
        const auto hub_to_rhs = sg.distance.get_or_inf(assignment.hub, assignment.rhs);
        if (lhs_to_hub != kInfWeight && hub_to_rhs != kInfWeight) {
            grouped_rows[assignment.lhs][assignment.hub].push_back({assignment.rhs, hub_to_rhs});
        }

        const auto rhs_to_hub = sg.distance.get_or_inf(assignment.rhs, assignment.hub);
        const auto hub_to_lhs = sg.distance.get_or_inf(assignment.hub, assignment.lhs);
        if (rhs_to_hub != kInfWeight && hub_to_lhs != kInfWeight) {
            grouped_rows[assignment.rhs][assignment.hub].push_back({assignment.lhs, hub_to_lhs});
        }
    }

    for (auto& [entry_border, by_hub] : grouped_rows) {
        std::vector<FactorizedTransferHubRow> rows;
        rows.reserve(by_hub.size());
        for (auto& [hub, exits] : by_hub) {
            std::sort(exits.begin(), exits.end(), [](const auto& lhs, const auto& rhs) {
                if (lhs.second != rhs.second) {
                    return lhs.second < rhs.second;
                }
                return lhs.first < rhs.first;
            });
            exits.erase(std::unique(exits.begin(), exits.end()), exits.end());
            rows.push_back(FactorizedTransferHubRow{
                hub,
                sg.distance.get_or_inf(entry_border, hub),
                std::move(exits),
            });
        }
        std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.entry_to_hub_distance != rhs.entry_to_hub_distance) {
                return lhs.entry_to_hub_distance < rhs.entry_to_hub_distance;
            }
            return lhs.hub < rhs.hub;
        });
        model.entry_rows.emplace(entry_border, std::move(rows));
    }

    return analyzed;
}

}  // namespace

std::vector<FactorizedTransferSubgraphStats> scan_factorized_transfer(
    const std::vector<Subgraph>& subgraphs
) {
    std::vector<FactorizedTransferSubgraphStats> stats;
    stats.reserve(subgraphs.size());
    for (const auto& sg : subgraphs) {
        stats.push_back(analyze_subgraph(sg).stats);
    }
    return stats;
}

FactorizedTransferSubgraphModel build_factorized_transfer_model(const Subgraph& sg) {
    return analyze_subgraph(sg).model;
}

FactorizedTransferScanSummary summarize_factorized_transfer(
    const std::vector<FactorizedTransferSubgraphStats>& stats
) {
    FactorizedTransferScanSummary summary;
    summary.subgraph_count = stats.size();
    for (const auto& item : stats) {
        if (item.num_borders <= 1U) {
            continue;
        }
        ++summary.scanned_subgraphs;
        summary.explicit_directed_arcs += item.explicit_directed_arcs;
        summary.factorized_directed_arcs += item.factorized_directed_arcs;
        summary.avg_hubs_per_subgraph += static_cast<double>(item.greedy_hub_count);
        summary.avg_arc_ratio += item.factorized_arc_ratio;
        if (item.improves_over_explicit) {
            ++summary.compressible_subgraphs;
        }
        if (item.factorized_arc_ratio <= 0.5) {
            ++summary.strong_compressible_subgraphs;
        }
    }

    if (summary.scanned_subgraphs != 0U) {
        summary.avg_hubs_per_subgraph /= static_cast<double>(summary.scanned_subgraphs);
        summary.avg_arc_ratio /= static_cast<double>(summary.scanned_subgraphs);
    }
    summary.overall_arc_ratio =
        (summary.explicit_directed_arcs == 0U) ? 1.0 :
        static_cast<double>(summary.factorized_directed_arcs) / static_cast<double>(summary.explicit_directed_arcs);
    return summary;
}

void write_factorized_transfer_csv(
    const std::filesystem::path& output_path,
    const std::vector<FactorizedTransferSubgraphStats>& stats
) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open factorized transfer csv");
    }

    write_csv_row(out, {
        "subgraph_id",
        "num_vertices",
        "num_borders",
        "explicit_directed_arcs",
        "unordered_border_pairs",
        "greedy_hub_count",
        "greedy_internal_hub_count",
        "greedy_border_hub_count",
        "pair_min_witness_count",
        "pair_avg_witness_count",
        "pair_max_witness_count",
        "factorized_entry_to_hub_arcs",
        "factorized_hub_to_exit_arcs",
        "factorized_directed_arcs",
        "factorized_arc_ratio",
        "avg_entry_hubs",
        "max_entry_hubs",
        "improves_over_explicit"
    });

    for (const auto& item : stats) {
        write_csv_row(out, {
            to_csv(item.subgraph_id),
            to_csv(item.num_vertices),
            to_csv(item.num_borders),
            to_csv(item.explicit_directed_arcs),
            to_csv(item.unordered_border_pairs),
            to_csv(item.greedy_hub_count),
            to_csv(item.greedy_internal_hub_count),
            to_csv(item.greedy_border_hub_count),
            to_csv(item.pair_min_witness_count),
            to_csv(item.pair_avg_witness_count),
            to_csv(item.pair_max_witness_count),
            to_csv(item.factorized_entry_to_hub_arcs),
            to_csv(item.factorized_hub_to_exit_arcs),
            to_csv(item.factorized_directed_arcs),
            to_csv(item.factorized_arc_ratio),
            to_csv(item.avg_entry_hubs),
            to_csv(item.max_entry_hubs),
            to_csv(item.improves_over_explicit ? "true" : "false")
        });
    }
}

}  // namespace bag
