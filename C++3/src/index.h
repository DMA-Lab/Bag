#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "factorized_transfer.h"
#include "object.h"
#include "partition.h"

namespace bag {

struct QueryPoint {
    Edge edge{};
    EdgeWeight offset{0};
};

enum class FcRule {
    AllBordersVisited,
    PaperStrict,
    UpperBoundCandidate,
    Disabled,
};

enum class QueryType {
    Range,
    Knn,
};

struct BorderExposureEvent {
    std::size_t query_id{0};
    QueryType query_type{QueryType::Range};
    SgId subgraph_id{0};
    VertexId border_vertex_id{kInvalidVertex};
    std::size_t visit_order{0};
    bool was_needed_for_final_answer{false};
    bool was_only_used_for_traversal{false};
    bool triggered_full_cover{false};
    bool tightened_upper_bound{false};
    bool pruned_subgraph{false};
    EdgeWeight distance_from_query{kInfWeight};
    std::size_t eventual_result_overlap{0};
};

struct CliqueRowShadowRecord {
    std::size_t query_id{0};
    QueryType query_type{QueryType::Range};
    SgId subgraph_id{0};
    VertexId entry_border{kInvalidVertex};
    std::size_t entry_visit_order{0};
    EdgeWeight entry_distance{kInfWeight};
    EdgeWeight threshold_at_entry{kInfWeight};
    std::size_t num_exits_total{0};
    std::size_t num_exits_dist_dominated{0};
    std::size_t num_exits_threshold_dead{0};
    std::size_t num_pq_pushes_from_this_row{0};
    std::size_t num_pq_pushes_beyond_useful_prefix{0};
    std::size_t num_exits_useful{0};
    std::size_t num_useful_pushes_from_this_row{0};
    std::size_t useful_prefix_len{0};
};

struct RangeQueryResult {
    std::vector<ObjId> object_ids;
    std::size_t fc_subgraphs{0};
    std::size_t pc_subgraphs{0};
    std::size_t auto_included_objects{0};
    std::size_t br_fc_included_objects{0};
    std::size_t partial_edge_auto_included_objects{0};
    std::size_t exact_returned_objects{0};
    std::size_t exact_checked_objects{0};
    long long init_us{0};
    long long skeleton_trace_us{0};
    long long subgraph_eval_us{0};
    std::size_t initial_local_vertices{0};
    std::size_t boundary_vertices_reached{0};
    std::size_t touched_subgraphs{0};
    std::size_t num_clique_relax_attempts{0};
    std::size_t num_successful_clique_relaxes{0};
    std::size_t num_pq_pushes_from_clique{0};
    std::size_t num_rows_truncated{0};
    std::size_t num_exits_skipped_by_truncation{0};
    std::size_t num_redundant_subgraph_reentries{0};
    std::size_t num_useful_subgraph_reentries{0};
    std::size_t factorized_rows_used{0};
    std::size_t factorized_hubs_used{0};
    std::size_t factorized_exits_emitted{0};
};

struct KnnItem {
    ObjId id{0};
    EdgeWeight distance{kInfWeight};
};

struct KnnQueryResult {
    std::vector<KnnItem> items;
    std::size_t visited_boundaries{0};
    std::size_t visited_subgraphs{0};
    std::size_t candidates_considered{0};
    std::size_t final_candidates{0};
    std::size_t exact_evaluated{0};
    bool vertex_fast_path{false};
    long long init_us{0};
    long long explore_us{0};
    long long finalize_us{0};
    long long pq_us{0};
    long long membership_us{0};
    long long subgraph_bookkeeping_us{0};
    long long clique_emit_us{0};
    std::size_t num_clique_relax_attempts{0};
    std::size_t num_successful_clique_relaxes{0};
    std::size_t num_pq_pushes_from_clique{0};
    std::size_t num_redundant_subgraph_reentries{0};
    std::size_t num_useful_subgraph_reentries{0};
    std::size_t boundary_pq_pushes{0};
    std::size_t boundary_pq_pops{0};
    std::size_t row_state_pushes{0};
    std::size_t row_state_pops{0};
    std::size_t streamed_rows_started{0};
    std::size_t streamed_row_emissions{0};
    std::size_t streamed_rows_stopped_by_threshold{0};
    std::size_t streamed_exits_skipped_by_threshold{0};
    std::size_t peak_frontier_queue_size{0};
    std::size_t safe_coverage_candidate_count{0};
    std::size_t first_k_candidates_boundary_visit_order{0};
    EdgeWeight first_finite_safe_coverage_radius{kInfWeight};
    std::size_t first_finite_safe_coverage_boundary_visit_order{0};
    EdgeWeight final_safe_coverage_radius{kInfWeight};
    std::size_t safe_coverage_updates{0};
    EdgeWeight first_finite_tau{kInfWeight};
    std::size_t first_finite_tau_boundary_visit_order{0};
    EdgeWeight final_tau{kInfWeight};
    std::size_t tau_updates{0};
    std::size_t would_stop_rows_by_safe_coverage{0};
    std::size_t would_skip_exits_by_safe_coverage{0};
    std::size_t would_save_pq_pushes_by_safe_coverage{0};
    std::size_t parent_regions_touched{0};
    std::size_t parent_region_size{0};
    std::size_t max_children_descended_in_parent{0};
    std::size_t single_child_parent_regions{0};
    double avg_children_descended_per_parent{0.0};
    double parent_descent_ratio{0.0};
    bool streamed_clique{false};
    std::size_t factorized_rows_used{0};
    std::size_t factorized_hubs_used{0};
    std::size_t factorized_exits_emitted{0};
};

struct FrontierBoundaryStats {
    EdgeWeight radius{0};
    std::size_t reached_boundaries{0};
    std::size_t frontier_boundaries{0};
    std::size_t outward_edges{0};
};

struct LocalSubgraphIndex {
    std::unordered_map<VertexId, std::uint32_t> local_ids;
    std::vector<std::vector<std::pair<std::uint32_t, EdgeWeight>>> adjacency;
};

struct LocalDijkstraResult {
    const LocalSubgraphIndex* index{nullptr};
    std::vector<EdgeWeight> dist;
    std::size_t reached_count{0};
};

struct InvertedBoundaryEntry {
    std::uint32_t sg_id{0};
    std::uint32_t row_id{0};
};

struct CliqueRowSpan {
    const std::pair<VertexId, EdgeWeight>* first{nullptr};
    const std::pair<VertexId, EdgeWeight>* last{nullptr};

    [[nodiscard]] const std::pair<VertexId, EdgeWeight>* begin() const { return first; }
    [[nodiscard]] const std::pair<VertexId, EdgeWeight>* end() const { return last; }
    [[nodiscard]] std::size_t size() const { return static_cast<std::size_t>(last - first); }
    [[nodiscard]] bool empty() const { return first == last; }
    const std::pair<VertexId, EdgeWeight>& operator[](std::size_t i) const { return first[i]; }
};

// Monotone integer priority queue (single-level radix heap) for Dijkstra over
// non-negative integer weights: every pushed key must be >= the key of the most
// recently popped item, which Dijkstra guarantees. Amortized O(word-bits) per
// operation with no per-node decrease-key; stale duplicates are filtered by the
// caller's distance check exactly as with a binary heap.
class MonotoneRadixQueue {
public:
    void reset() {
        if (occupied_ != 0U) {
            for (auto& bucket : buckets_) {
                bucket.clear();
            }
        }
        occupied_ = 0U;
        last_ = 0U;
        size_ = 0U;
    }

    [[nodiscard]] bool empty() const { return size_ == 0U; }
    [[nodiscard]] std::size_t size() const { return size_; }

    void push(EdgeWeight key, VertexId value) {
        const auto idx = bucket_index(key);
        buckets_[idx].push_back({key, value});
        occupied_ |= (1ULL << idx);
        ++size_;
    }

    std::pair<EdgeWeight, VertexId> pop() {
        auto idx = static_cast<std::size_t>(std::countr_zero(occupied_));
        if (idx != 0U) {
            auto& bucket = buckets_[idx];
            EdgeWeight min_key = bucket.front().first;
            for (const auto& [key, value] : bucket) {
                min_key = std::min(min_key, key);
            }
            last_ = min_key;
            for (const auto& item : bucket) {
                const auto new_idx = bucket_index(item.first);
                buckets_[new_idx].push_back(item);
                occupied_ |= (1ULL << new_idx);
            }
            bucket.clear();
            occupied_ &= ~(1ULL << idx);
        }
        auto& zero_bucket = buckets_[0];
        const auto item = zero_bucket.back();
        zero_bucket.pop_back();
        if (zero_bucket.empty()) {
            occupied_ &= ~1ULL;
        }
        --size_;
        return item;
    }

private:
    [[nodiscard]] std::size_t bucket_index(EdgeWeight key) const {
        const auto diff = static_cast<std::uint32_t>(key) ^ static_cast<std::uint32_t>(last_);
        return diff == 0U ? 0U : static_cast<std::size_t>(std::bit_width(diff));
    }

    std::array<std::vector<std::pair<EdgeWeight, VertexId>>, 33> buckets_{};
    std::uint64_t occupied_{0U};
    EdgeWeight last_{0U};
    std::size_t size_{0U};
};

class SkeletonIndex {
public:
    static SkeletonIndex build(const Graph& global, std::vector<Subgraph> subgraphs);
    void configure_factorized_transfer(
        double max_arc_ratio = 0.5,
        std::size_t min_border_count = 12
    );
    std::size_t configure_gstar_shortcuts(
        std::size_t leaf_hops = 2,
        std::size_t shortcuts_per_subgraph = 1
    );

    [[nodiscard]] const Graph& skeleton() const;
    [[nodiscard]] const std::vector<Subgraph>& subgraphs() const;
    [[nodiscard]] const std::unordered_map<Edge, SgId, PairHash>& edge_to_subgraph() const;
    [[nodiscard]] const std::unordered_map<VertexId, SgId>& vertex_to_subgraph() const;
    [[nodiscard]] std::size_t gstar_shortcut_edges_added() const;

    RangeQueryResult range_query(
        const QueryPoint& query,
        EdgeWeight radius,
        const IndexedMovingObjectSet& objects,
        FcRule rule,
        std::size_t query_id = 0,
        std::vector<BorderExposureEvent>* exposure_events = nullptr,
        std::vector<CliqueRowShadowRecord>* row_shadow_records = nullptr,
        bool enable_row_truncation = true,
        bool use_factorized_transfer = false
    ) const;

    std::vector<ObjId> exact_range_query(
        const QueryPoint& query,
        EdgeWeight radius,
        const IndexedMovingObjectSet& objects
    ) const;

    KnnQueryResult knn_query(
        const QueryPoint& query,
        std::size_t k,
        const IndexedMovingObjectSet& objects,
        std::size_t query_id = 0,
        std::vector<BorderExposureEvent>* exposure_events = nullptr,
        std::vector<CliqueRowShadowRecord>* row_shadow_records = nullptr,
        bool streamed_clique = false,
        bool safe_coverage_shadow = false,
        std::size_t parent_shadow_size = 0,
        bool use_factorized_transfer = false,
        bool knn_sound_termination = true
    ) const;

    KnnQueryResult knn_query_local_dijkstra(
        const QueryPoint& query,
        std::size_t k,
        const IndexedMovingObjectSet& objects,
        FcRule rule,
        std::size_t query_id = 0
    ) const;

    KnnQueryResult knn_query_global_dijkstra(
        const QueryPoint& query,
        std::size_t k,
        const IndexedMovingObjectSet& objects,
        std::size_t query_id = 0
    ) const;

    KnnQueryResult exact_knn_query(
        const QueryPoint& query,
        std::size_t k,
        const IndexedMovingObjectSet& objects
    ) const;

    FrontierBoundaryStats frontier_boundary_stats(
        const QueryPoint& query,
        EdgeWeight radius
    ) const;

private:
    const Subgraph& require_initial_subgraph(const Edge& edge) const;
    void rebuild_sorted_skeleton_rows();

    const Graph* global_{nullptr};
    VertexId max_vertex_id_{0};
    Graph skeleton_;
    std::vector<Subgraph> subgraphs_;
    std::unordered_map<VertexId, std::vector<SgId>> inverted_map_;
    std::vector<std::vector<SgId>> inverted_index_fast_;
    std::unordered_map<VertexId, SgId> vertex_to_subgraph_;
    std::vector<SgId> vertex_to_subgraph_fast_;
    std::unordered_map<Edge, SgId, PairHash> edge_to_subgraph_;
    std::vector<std::int32_t> sorted_skeleton_row_index_;
    std::vector<std::uint32_t> sorted_skeleton_row_offsets_;
    std::vector<std::pair<VertexId, EdgeWeight>> sorted_skeleton_row_flat_;
    std::vector<std::vector<VertexId>> subgraph_boundaries_;
    std::vector<std::vector<HalfWeight>> subgraph_boundary_rbs_;
    std::vector<std::vector<EdgeWeight>> subgraph_boundary_ibs_;
    std::vector<std::vector<std::pair<Edge, EdgeWeight>>> subgraph_undirected_edges_;
    std::vector<std::vector<std::uint32_t>> subgraph_clique_row_offsets_;
    std::vector<std::vector<std::pair<VertexId, EdgeWeight>>> subgraph_clique_row_flat_;
    std::vector<std::vector<InvertedBoundaryEntry>> inverted_boundary_entries_;
    std::vector<LocalSubgraphIndex> subgraph_local_indices_;
    std::vector<std::vector<SgId>> subgraph_adjacency_;
    std::vector<FactorizedTransferSubgraphModel> factorized_transfer_models_;
    std::vector<std::uint8_t> factorized_transfer_enabled_;
    mutable std::vector<EdgeWeight> skeleton_scratch_dist_;
    mutable std::vector<std::uint32_t> skeleton_scratch_stamp_;
    mutable std::uint32_t skeleton_scratch_epoch_{0};
    mutable std::vector<std::size_t> range_subgraph_touch_count_scratch_;
    mutable std::vector<std::uint32_t> range_subgraph_touch_stamp_scratch_;
    mutable std::uint32_t range_subgraph_touch_epoch_{0};
    mutable MonotoneRadixQueue range_frontier_queue_;
    mutable MonotoneRadixQueue knn_frontier_queue_;
    mutable std::vector<std::uint32_t> range_object_seen_stamp_;
    mutable std::uint32_t range_object_seen_epoch_{0};
    mutable std::vector<EdgeWeight> knn_best_subgraph_lb_scratch_;
    mutable std::vector<std::uint32_t> knn_best_subgraph_lb_stamp_;
    mutable std::vector<EdgeWeight> knn_best_subgraph_ud_scratch_;
    mutable std::vector<std::uint32_t> knn_best_subgraph_ud_stamp_;
    mutable std::vector<std::uint32_t> knn_visited_subgraph_stamp_;
    mutable std::vector<std::uint32_t> knn_pruned_subgraph_stamp_;
    mutable std::vector<std::uint32_t> knn_tightened_subgraph_stamp_;
    mutable std::vector<std::size_t> knn_sg_touch_count_scratch_;
    mutable std::vector<std::uint32_t> knn_sg_touch_stamp_;
    mutable std::uint32_t knn_subgraph_epoch_{0};
    std::size_t gstar_shortcut_edges_added_{0};
};

}  // namespace bag
