#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "graph.h"

namespace bag {

struct Subgraph;

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
        const std::vector<Subgraph>& subgraphs,
        bool maintain_knn_metadata = true,
        bool maintain_edge_buckets = true,
        bool eager_knn_finalize = false
    );

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] const MovingObject& operator[](ObjId id) const;
    [[nodiscard]] const std::vector<MovingObject>& objects() const;
    [[nodiscard]] const std::vector<ObjId>& objects_in(SgId sg_id) const;
    [[nodiscard]] const std::vector<ObjId>& objects_in_knn_order(SgId sg_id) const;
    [[nodiscard]] const std::vector<PopulatedEdgeObjects>& populated_edges_in(SgId sg_id) const;
    [[nodiscard]] EdgeWeight knn_suffix(ObjId id) const;
    [[nodiscard]] const std::vector<std::pair<VertexId, EdgeWeight>>& knn_border_costs(ObjId id) const;
    [[nodiscard]] SgId object_subgraph(ObjId id) const;
    [[nodiscard]] std::size_t occupied_subgraphs() const;
    [[nodiscard]] double avg_objects_per_occupied_subgraph() const;
    [[nodiscard]] std::size_t max_objects_in_subgraph() const;
    void move_object(
        MovingObject updated_object,
        const std::unordered_map<Edge, SgId, PairHash>& edge_to_subgraph,
        const std::vector<Subgraph>& subgraphs
    );
    void finalize_updates();

private:
    void ensure_knn_metadata(ObjId id) const;
    void ensure_knn_order(SgId sg_id) const;
    void add_object_to_subgraph(ObjId id, SgId sg_id);
    void remove_object_from_subgraph(ObjId id, SgId sg_id);
    void add_object_to_edge_bucket(ObjId id, SgId sg_id, const Edge& edge, EdgeWeight edge_weight);
    void remove_object_from_edge_bucket(ObjId id, SgId sg_id);
    void recompute_knn_metadata(ObjId id, const std::vector<Subgraph>& subgraphs);
    void rebuild_knn_order_for_subgraph(SgId sg_id);
    void mark_subgraph_dirty(SgId sg_id);
    void refresh_occupancy_stats() const;

    std::vector<MovingObject> objects_;
    std::vector<std::vector<ObjId>> subgraph_objects_;
    std::vector<std::vector<PopulatedEdgeObjects>> subgraph_populated_edges_;
    mutable std::vector<std::vector<ObjId>> knn_sorted_objects_;
    mutable std::vector<EdgeWeight> knn_suffix_;
    mutable std::vector<std::vector<std::pair<VertexId, EdgeWeight>>> knn_border_costs_;
    std::vector<SgId> object_subgraph_;
    std::vector<std::size_t> object_position_in_subgraph_;
    std::vector<std::size_t> object_edge_bucket_index_;
    std::vector<std::size_t> object_position_in_edge_bucket_;
    std::vector<std::unordered_map<std::uint64_t, std::size_t>> populated_edge_index_;
    mutable std::vector<bool> knn_sorted_dirty_;
    mutable std::vector<bool> knn_metadata_dirty_;
    std::vector<SgId> dirty_subgraphs_;
    std::vector<bool> dirty_subgraph_enqueued_;
    mutable bool occupancy_stats_dirty_{true};
    mutable std::size_t occupied_subgraphs_count_{0};
    mutable std::size_t max_objects_in_any_subgraph_{0};
    bool maintain_knn_metadata_{true};
    bool maintain_edge_buckets_{true};
    bool eager_knn_finalize_{false};
    const std::vector<Subgraph>* knn_subgraphs_{nullptr};
};

}  // namespace bag
