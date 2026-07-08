#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "index.h"
#include "partition.h"

namespace bag {

struct DiagnosticsOptions {
    std::string graph_name;
    std::string impl_version;
    std::size_t theta{0};
    VertexId seed_id{kInvalidVertex};
    bool paper_strict{false};
    bool shortcut_repartition{false};
    std::filesystem::path output_dir;
};

struct QueryRuntimeSample {
    std::size_t query_id{0};
    QueryType query_type{QueryType::Range};
    long long elapsed_us{0};
};

struct BorderTouchAggregate {
    std::size_t range_touch_count{0};
    std::size_t knn_touch_count{0};
    long long first_touch_rank_range{-1};
    long long first_touch_rank_knn{-1};
};

struct SubgraphTouchAggregate {
    std::size_t query_touch_count_range{0};
    std::size_t query_touch_count_knn{0};
};

struct QueryTouchSummary {
    std::unordered_map<std::pair<SgId, VertexId>, BorderTouchAggregate, PairHash> border_stats;
    std::unordered_map<SgId, SubgraphTouchAggregate> subgraph_stats;
};

struct ExactDemotionVerdict {
    std::string feasible_exact_range{"not_run"};
    std::string feasible_exact_knn{"not_run"};
    std::string first_failed_condition{"exact_not_checked"};
};

struct DiagnosticsWorkloadData {
    const std::vector<QueryPoint>* queries{nullptr};
    const std::vector<MovingObject>* raw_objects{nullptr};
    const std::vector<std::vector<ObjId>>* baseline_range_results{nullptr};
    const std::vector<std::vector<KnnItem>>* baseline_knn_results{nullptr};
    const std::vector<BorderExposureEvent>* exposure_events{nullptr};
    const std::vector<QueryRuntimeSample>* runtime_samples{nullptr};
    FcRule fc_rule{FcRule::PaperStrict};
    EdgeWeight range_radius{0};
    std::size_t knn_k{0};
    std::size_t exact_demotion_limit{0};
};

QueryTouchSummary summarize_query_exposures(const std::vector<BorderExposureEvent>& events);

void write_query_exposure_csv(
    const std::filesystem::path& output_path,
    const DiagnosticsOptions& diagnostics_options,
    const std::vector<BorderExposureEvent>& events
);

void write_clique_row_shadow_csv(
    const std::filesystem::path& output_path,
    const DiagnosticsOptions& diagnostics_options,
    const std::vector<CliqueRowShadowRecord>& records
);

void write_clique_row_shadow_summary_csv(
    const std::filesystem::path& output_path,
    const DiagnosticsOptions& diagnostics_options,
    const std::vector<CliqueRowShadowRecord>& records
);

std::unordered_map<std::pair<SgId, VertexId>, ExactDemotionVerdict, PairHash> evaluate_exact_demotion_candidates(
    const Graph& global,
    const std::vector<Subgraph>& subgraphs,
    const DiagnosticsWorkloadData& workload
);

void write_skeleton_hotspots_csv(
    const std::filesystem::path& output_path,
    const Graph& global,
    const std::vector<Subgraph>& subgraphs,
    const std::vector<BorderExposureEvent>& events,
    const std::vector<QueryRuntimeSample>& runtime_samples
);

void write_query_runtime_samples_csv(
    const std::filesystem::path& output_path,
    const std::vector<QueryRuntimeSample>& runtime_samples
);

void write_query_runtime_summary_csv(
    const std::filesystem::path& output_path,
    const std::vector<QueryRuntimeSample>& runtime_samples
);

void write_partition_diagnostics(
    const Graph& global,
    const std::vector<Subgraph>& subgraphs,
    const PartitionOptions& partition_options,
    const DiagnosticsOptions& diagnostics_options,
    const QueryTouchSummary* touch_summary = nullptr,
    const std::unordered_map<std::pair<SgId, VertexId>, ExactDemotionVerdict, PairHash>* exact_verdicts = nullptr
);

}  // namespace bag
