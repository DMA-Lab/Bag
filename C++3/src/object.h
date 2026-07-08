#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "graph.h"

namespace bag {

struct Subgraph;

struct BorderCostSpan {
    const std::pair<std::uint32_t, EdgeWeight>* first{nullptr};
    const std::pair<std::uint32_t, EdgeWeight>* last{nullptr};

    [[nodiscard]] const std::pair<std::uint32_t, EdgeWeight>* begin() const { return first; }
    [[nodiscard]] const std::pair<std::uint32_t, EdgeWeight>* end() const { return last; }
    [[nodiscard]] std::size_t size() const { return static_cast<std::size_t>(last - first); }
    [[nodiscard]] bool empty() const { return first == last; }
};

struct MovingObject {
    ObjId unique_id{0};
    Edge edge{};
    EdgeWeight offset{0};
    EdgeWeight edge_weight{0};
};

struct PopulatedEdgeObjects {
    Edge edge{};
    EdgeWeight edge_weight{0};
    std::vector<ObjId> object_ids;
};

class MovingObjectSet {
public:
    static MovingObjectSet random_uniform(const Graph& graph, std::size_t count, std::uint64_t seed);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] const MovingObject& operator[](ObjId id) const;
    [[nodiscard]] const std::vector<MovingObject>& objects() const;
    void push(MovingObject object);

private:
    std::vector<MovingObject> objects_;
};

class IndexedMovingObjectSet {
public:
    IndexedMovingObjectSet() = default;

    static IndexedMovingObjectSet from_partition(
        const std::vector<bag::MovingObject>& objects,
        const std::unordered_map<Edge, SgId, PairHash>& edge_to_subgraph,
        const std::vector<Subgraph>& subgraphs
    );

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] const MovingObject& operator[](ObjId id) const;
    [[nodiscard]] const std::vector<MovingObject>& objects() const;
    [[nodiscard]] const std::vector<ObjId>& objects_incident_to(VertexId vertex) const;
    [[nodiscard]] const std::vector<ObjId>& objects_in(SgId sg_id) const;
    [[nodiscard]] const std::vector<ObjId>& objects_in_knn_order(SgId sg_id) const;
    [[nodiscard]] const std::vector<PopulatedEdgeObjects>& populated_edges_in(SgId sg_id) const;
    [[nodiscard]] EdgeWeight knn_suffix(ObjId id) const;
    [[nodiscard]] BorderCostSpan knn_border_costs(ObjId id) const;
    void prefetch_knn_metadata(ObjId id) const;
    // Per-subgraph, in knn (suffix-ascending) traversal order: the object suffix
    // and border-cost row of the j-th object are stored contiguously so the kNN
    // refinement scan reads them sequentially instead of through scattered
    // per-object-id indirection.
    [[nodiscard]] const std::vector<EdgeWeight>& knn_order_suffixes(SgId sg_id) const;
    [[nodiscard]] BorderCostSpan knn_order_costs(SgId sg_id, std::size_t order_index) const;
    // Objects whose cheapest-cost border (b0) is the subgraph's local border
    // `local_idx`, given as their via-b0 cost (== suffix). Used to push a tight
    // per-object kNN upper bound (d(b0)+suffix) when border b0 settles.
    [[nodiscard]] std::pair<const EdgeWeight*, const EdgeWeight*>
    knn_b0_group(SgId sg_id, std::uint32_t local_idx) const;
    [[nodiscard]] SgId object_subgraph(ObjId id) const;
    void move_object(
        MovingObject updated_object,
        const std::unordered_map<Edge, SgId, PairHash>& edge_to_subgraph,
        const std::vector<Subgraph>& subgraphs
    );
    void finalize_updates();

private:
    void add_object_to_subgraph(ObjId id, SgId sg_id);
    void remove_object_from_subgraph(ObjId id, SgId sg_id);
    void add_object_to_edge_bucket(ObjId id, SgId sg_id, const Edge& edge, EdgeWeight edge_weight);
    void remove_object_from_edge_bucket(ObjId id, SgId sg_id);
    void add_object_to_incidence(ObjId id, const MovingObject& object);
    void remove_object_from_incidence(ObjId id, const MovingObject& object);
    void recompute_knn_metadata(ObjId id, const std::vector<Subgraph>& subgraphs);
    void mark_subgraph_dirty(SgId sg_id);

    std::vector<MovingObject> objects_;
    std::unordered_map<VertexId, std::vector<ObjId>> incident_objects_;
    std::vector<std::vector<ObjId>> subgraph_objects_;
    std::vector<std::vector<PopulatedEdgeObjects>> subgraph_populated_edges_;
    std::vector<std::vector<ObjId>> knn_sorted_objects_;
    // Per-subgraph SoA in knn order (parallel to knn_sorted_objects_[sg]).
    std::vector<std::vector<EdgeWeight>> knn_sorted_suffix_;
    std::vector<std::vector<std::uint32_t>> knn_sorted_cost_rowptr_;
    std::vector<std::vector<std::pair<std::uint32_t, EdgeWeight>>> knn_sorted_cost_flat_;
    // Per subgraph, objects grouped by cheapest-cost border (b0): CSR offsets
    // over local border indices, and the objects' via-b0 costs (suffixes).
    std::vector<std::vector<std::uint32_t>> knn_b0_group_offsets_;
    std::vector<std::vector<EdgeWeight>> knn_b0_group_suffix_;
    std::vector<EdgeWeight> knn_suffix_;
    // Border-cost rows pooled in one arena: (offset, length) spans per object.
    // Rebuilt rows are appended; finalize_updates() compacts when waste grows.
    std::vector<std::pair<std::uint32_t, EdgeWeight>> knn_border_costs_arena_;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> knn_border_costs_span_;
    std::size_t knn_border_costs_live_entries_{0};
    std::vector<SgId> object_subgraph_;
    std::vector<std::size_t> object_position_in_subgraph_;
    std::vector<std::size_t> object_edge_bucket_index_;
    std::vector<std::size_t> object_position_in_edge_bucket_;
    std::vector<std::unordered_map<std::uint64_t, std::size_t>> populated_edge_index_;
    std::vector<bool> knn_sorted_dirty_;
};

}  // namespace bag
