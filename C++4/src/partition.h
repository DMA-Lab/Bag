#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "distance.h"
#include "graph.h"

namespace bag {

struct PartitionOptions {
    std::size_t theta{30};
    VertexId partition_seed{1};
    bool paper_strict_mode{true};
    bool adaptive_z{false};
    double adaptive_alpha{1.0};
    bool border_minimization{false};
    bool shortcut_repartition{true};
    std::size_t shortcut_small_upper_bound{3};
    std::size_t shortcut_k_neighbors{200};
    EdgeWeight shortcut_radius_limit{6000};
    std::size_t shortcut_max_tiny_subgraphs{100000};
    bool phase1_inplace{false};
    bool defer_nearest_border_fill{false};
    bool skip_phase1_br_check{false};
    bool phase1_local_audit{false};
    bool phase2_incremental_distance_update{false};
    bool progress_log{false};
    std::size_t progress_log_every_subgraphs{5000};
    std::filesystem::path checkpoint_path;
    std::string checkpoint_key;
    bool checkpoint_resume{false};
    bool checkpoint_resume_required{false};
    bool checkpoint_write{false};
    std::size_t checkpoint_every_subgraphs{0};
    std::size_t checkpoint_stop_after_subgraphs{0};
    std::filesystem::path growth_trace_output;
};

struct ShortcutRepartitionStats {
    std::size_t tiny_subgraphs{0};
    std::size_t merged_components{0};
    std::size_t merged_output_subgraphs{0};
    bool skipped_by_tiny_budget{false};
};

struct PartitionRuntimeStats {
    long long core_partition_us{0};
    long long shortcut_repartition_us{0};
    std::size_t subgraphs_before_shortcut{0};
    std::size_t subgraphs_after_shortcut{0};
    std::uint64_t phase1_attempts{0};
    std::uint64_t phase1_commits{0};
    std::uint64_t phase1_rejects{0};
    long long phase1_clone_us{0};
    long long phase1_finalize_us{0};
    long long phase1_refresh_us{0};
    long long phase1_extend_distance_us{0};
    long long phase1_br_us{0};
    long long phase1_nearest_border_us{0};
    std::uint64_t phase2_attempts{0};
    std::uint64_t phase2_commits{0};
    std::uint64_t phase2_rejects{0};
    long long phase2_clone_us{0};
    long long phase2_finalize_us{0};
    long long phase2_refresh_us{0};
    long long phase2_apsp_us{0};
    long long phase2_distance_update_us{0};
    long long phase2_br_us{0};
    long long phase2_nearest_border_us{0};
    ShortcutRepartitionStats shortcut_stats;
};

struct PartitionCheckpointState {
    bool enabled{false};
    bool resume_requested{false};
    bool resume_hit{false};
    std::size_t resumed_subgraphs{0};
    long long load_us{0};
    long long save_us{0};
    std::size_t save_count{0};
    std::size_t last_saved_subgraphs{0};
};

struct Subgraph {
    SgId id{0};
    VertexId seed_vertex{kInvalidVertex};
    Graph graph;
    DistanceTable distance;
    std::unordered_set<VertexId> bound_vertices;
    std::unordered_set<VertexId> internal_vertices;
    std::unordered_map<VertexId, HalfWeight> rb_map;
    std::unordered_map<VertexId, EdgeWeight> internal_to_nearest_border_dist;
    std::vector<VertexId> insertion_order;

    [[nodiscard]] bool contains(VertexId v) const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::vector<VertexId> vertices() const;
    [[nodiscard]] bool check_internal_vertex(const Graph& global, VertexId v) const;
    void turn_to_internal(VertexId v);
};

class VfipPartition {
public:
    VfipPartition(const Graph& global, PartitionOptions options);
    std::vector<Subgraph> run();
    [[nodiscard]] const PartitionRuntimeStats& stats() const;
    [[nodiscard]] const PartitionCheckpointState& checkpoint_state() const;

private:
    [[nodiscard]] bool has_unallocated_adjacent_edge(VertexId source) const;
    [[nodiscard]] std::vector<std::pair<VertexId, EdgeWeight>> unallocated_adjacent_edges(VertexId source) const;
    [[nodiscard]] Subgraph expand_next(VertexId seed, SgId provisional_subgraph_id);
    void trace_growth_step(
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
    );

    const Graph& global_;
    PartitionOptions options_;
    PartitionRuntimeStats stats_;
    PartitionCheckpointState checkpoint_state_;
    std::unordered_set<std::uint64_t> added_edges_;
    std::uint64_t next_trace_step_{0};
    std::ofstream growth_trace_out_;
};

}  // namespace bag
