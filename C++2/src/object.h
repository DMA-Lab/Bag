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
    [[nodiscard]] const std::vector<std::pair<VertexId, EdgeWeight>>& knn_border_costs(ObjId id) const;
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
    std::vector<EdgeWeight> knn_suffix_;
    std::vector<std::vector<std::pair<VertexId, EdgeWeight>>> knn_border_costs_;
    std::vector<SgId> object_subgraph_;
    std::vector<std::size_t> object_position_in_subgraph_;
    std::vector<std::size_t> object_edge_bucket_index_;
    std::vector<std::size_t> object_position_in_edge_bucket_;
    std::vector<std::unordered_map<std::uint64_t, std::size_t>> populated_edge_index_;
    std::vector<bool> knn_sorted_dirty_;
};

}  // namespace bag
