#pragma once

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "partition.h"

namespace bag {

struct FactorizedTransferHubRow {
    VertexId hub{kInvalidVertex};
    EdgeWeight entry_to_hub_distance{kInfWeight};
    std::vector<std::pair<VertexId, EdgeWeight>> exits;
};

struct FactorizedTransferSubgraphModel {
    SgId subgraph_id{0};
    bool feasible{false};
    std::size_t num_borders{0};
    std::size_t explicit_directed_arcs{0};
    std::size_t factorized_directed_arcs{0};
    double factorized_arc_ratio{1.0};
    std::size_t greedy_hub_count{0};
    std::size_t greedy_internal_hub_count{0};
    std::size_t greedy_border_hub_count{0};
    std::unordered_map<VertexId, std::vector<FactorizedTransferHubRow>> entry_rows;
};

struct FactorizedTransferSubgraphStats {
    SgId subgraph_id{0};
    std::size_t num_vertices{0};
    std::size_t num_borders{0};
    std::size_t explicit_directed_arcs{0};
    std::size_t unordered_border_pairs{0};
    std::size_t greedy_hub_count{0};
    std::size_t greedy_internal_hub_count{0};
    std::size_t greedy_border_hub_count{0};
    std::size_t pair_min_witness_count{0};
    double pair_avg_witness_count{0.0};
    std::size_t pair_max_witness_count{0};
    std::size_t factorized_entry_to_hub_arcs{0};
    std::size_t factorized_hub_to_exit_arcs{0};
    std::size_t factorized_directed_arcs{0};
    double factorized_arc_ratio{1.0};
    double avg_entry_hubs{0.0};
    std::size_t max_entry_hubs{0};
    bool improves_over_explicit{false};
};

struct FactorizedTransferScanSummary {
    std::size_t subgraph_count{0};
    std::size_t scanned_subgraphs{0};
    std::size_t compressible_subgraphs{0};
    std::size_t strong_compressible_subgraphs{0};
    std::size_t explicit_directed_arcs{0};
    std::size_t factorized_directed_arcs{0};
    double overall_arc_ratio{1.0};
    double avg_hubs_per_subgraph{0.0};
    double avg_arc_ratio{1.0};
};

std::vector<FactorizedTransferSubgraphStats> scan_factorized_transfer(
    const std::vector<Subgraph>& subgraphs
);

FactorizedTransferSubgraphModel build_factorized_transfer_model(
    const Subgraph& sg
);

FactorizedTransferScanSummary summarize_factorized_transfer(
    const std::vector<FactorizedTransferSubgraphStats>& stats
);

void write_factorized_transfer_csv(
    const std::filesystem::path& output_path,
    const std::vector<FactorizedTransferSubgraphStats>& stats
);

}  // namespace bag
