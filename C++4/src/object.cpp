#include "object.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "partition.h"

namespace bag {

namespace {

constexpr SgId kInvalidSubgraphId = static_cast<SgId>(-1);
constexpr std::size_t kInvalidIndex = std::numeric_limits<std::size_t>::max();

const std::vector<ObjId>& empty_ids() {
    static const std::vector<ObjId> empty;
    return empty;
}

EdgeWeight nearest_border_distance(const Subgraph& sg, VertexId v) {
    if (sg.bound_vertices.contains(v)) {
        return 0;
    }
    const auto it = sg.internal_to_nearest_border_dist.find(v);
    return (it == sg.internal_to_nearest_border_dist.end()) ? kInfWeight : it->second;
}

SgId resolve_subgraph(
    const Edge& edge,
    const std::unordered_map<Edge, SgId, PairHash>& edge_to_subgraph
) {
    const auto ordered = ordered_edge(edge.first, edge.second);
    const auto it = edge_to_subgraph.find(ordered);
    if (it == edge_to_subgraph.end()) {
        throw std::runtime_error("object edge is not indexed by any subgraph");
    }
    return it->second;
}

void compute_knn_metadata(
    const MovingObject& object,
    const Subgraph& sg,
    EdgeWeight& suffix_out,
    std::vector<std::pair<std::uint32_t, EdgeWeight>>& border_costs_out
) {
    const auto left = nearest_border_distance(sg, object.edge.first);
    const auto right = nearest_border_distance(sg, object.edge.second);
    EdgeWeight suffix = kInfWeight;
    if (left != kInfWeight && left <= kInfWeight - object.offset) {
        suffix = std::min(suffix, static_cast<EdgeWeight>(left + object.offset));
    }
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    if (right != kInfWeight && right <= kInfWeight - right_cost) {
        suffix = std::min(suffix, static_cast<EdgeWeight>(right + right_cost));
    }
    suffix_out = suffix;

    // Border costs are keyed by the border's local index within the subgraph's
    // ascending-sorted boundary list (the same order SkeletonIndex materializes),
    // so query-time evaluation can use flat per-subgraph border-distance arrays.
    std::vector<VertexId> sorted_boundaries(sg.bound_vertices.begin(), sg.bound_vertices.end());
    std::sort(sorted_boundaries.begin(), sorted_boundaries.end());

    border_costs_out.clear();
    border_costs_out.reserve(sorted_boundaries.size());
    for (std::uint32_t local_idx = 0; local_idx < sorted_boundaries.size(); ++local_idx) {
        const auto b = sorted_boundaries[local_idx];
        EdgeWeight best = kInfWeight;
        const auto left_dist = sg.distance.get_or_inf(b, object.edge.first);
        if (left_dist != kInfWeight && left_dist <= kInfWeight - object.offset) {
            best = std::min(best, static_cast<EdgeWeight>(left_dist + object.offset));
        }
        const auto right_dist = sg.distance.get_or_inf(b, object.edge.second);
        if (right_dist != kInfWeight && right_dist <= kInfWeight - (object.edge_weight - object.offset)) {
            best = std::min(best, static_cast<EdgeWeight>(right_dist + (object.edge_weight - object.offset)));
        }
        if (best != kInfWeight) {
            border_costs_out.push_back({local_idx, best});
        }
    }
    std::sort(border_costs_out.begin(), border_costs_out.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
            return lhs.second < rhs.second;
        }
        return lhs.first < rhs.first;
    });
}

}  // namespace

MovingObjectSet MovingObjectSet::random_uniform(
    const Graph& graph,
    std::size_t count,
    std::uint64_t seed
) {
    const auto edges = graph.undirected_edges();
    if (edges.empty()) {
        throw std::runtime_error("graph has no edges");
    }

    std::vector<std::uint64_t> cumulative;
    cumulative.reserve(edges.size());
    std::uint64_t total_slots = 0;
    for (const auto& [edge, weight] : edges) {
        (void)edge;
        if (weight > 1) {
            total_slots += static_cast<std::uint64_t>(weight - 1);
        }
        cumulative.push_back(total_slots);
    }
    if (total_slots == 0) {
        throw std::runtime_error("graph does not have any valid object slots");
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> edge_ticket(0, total_slots - 1);

    MovingObjectSet set;
    set.objects_.reserve(count);
    for (ObjId id = 0; id < count; ++id) {
        const auto ticket = edge_ticket(rng);
        const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), ticket + 1);
        const auto index = static_cast<std::size_t>(std::distance(cumulative.begin(), it));
        const auto& [edge, weight] = edges[index];
        std::uniform_int_distribution<EdgeWeight> offset_dist(1, weight - 1);
        set.objects_.push_back(MovingObject{
            id,
            ordered_edge(edge.first, edge.second),
            offset_dist(rng),
            weight,
        });
    }
    return set;
}

std::size_t MovingObjectSet::size() const {
    return objects_.size();
}

bool MovingObjectSet::empty() const {
    return objects_.empty();
}

const MovingObject& MovingObjectSet::operator[](ObjId id) const {
    return objects_.at(static_cast<std::size_t>(id));
}

const std::vector<MovingObject>& MovingObjectSet::objects() const {
    return objects_;
}

void MovingObjectSet::push(MovingObject object) {
    objects_.push_back(object);
}

IndexedMovingObjectSet IndexedMovingObjectSet::from_partition(
    const std::vector<bag::MovingObject>& objects,
    const std::unordered_map<Edge, SgId, PairHash>& edge_to_subgraph,
    const std::vector<Subgraph>& subgraphs
) {
    IndexedMovingObjectSet indexed;
    indexed.objects_ = objects;
    for (std::size_t i = 0; i < indexed.objects_.size(); ++i) {
        if (indexed.objects_[i].unique_id != i) {
            throw std::runtime_error("IndexedMovingObjectSet requires contiguous object ids");
        }
    }

    indexed.subgraph_objects_.resize(subgraphs.size());
    indexed.subgraph_populated_edges_.resize(subgraphs.size());
    indexed.knn_sorted_objects_.resize(subgraphs.size());
    indexed.knn_sorted_suffix_.resize(subgraphs.size());
    indexed.knn_sorted_cost_rowptr_.resize(subgraphs.size());
    indexed.knn_sorted_cost_flat_.resize(subgraphs.size());
    indexed.knn_b0_group_offsets_.resize(subgraphs.size());
    indexed.knn_b0_group_suffix_.resize(subgraphs.size());
    indexed.populated_edge_index_.resize(subgraphs.size());
    indexed.knn_sorted_dirty_.assign(subgraphs.size(), false);

    indexed.knn_suffix_.assign(objects.size(), kInfWeight);
    indexed.knn_border_costs_span_.assign(objects.size(), {0U, 0U});
    indexed.knn_border_costs_arena_.reserve(objects.size() * 8U);
    indexed.object_subgraph_.assign(objects.size(), kInvalidSubgraphId);
    indexed.object_position_in_subgraph_.assign(objects.size(), kInvalidIndex);
    indexed.object_edge_bucket_index_.assign(objects.size(), kInvalidIndex);
    indexed.object_position_in_edge_bucket_.assign(objects.size(), kInvalidIndex);
    indexed.incident_objects_.reserve(objects.size() * 2U + 1U);

    for (const auto& object : indexed.objects_) {
        const auto sg_id = resolve_subgraph(object.edge, edge_to_subgraph);
        indexed.add_object_to_incidence(object.unique_id, object);
        indexed.add_object_to_subgraph(object.unique_id, sg_id);
        indexed.add_object_to_edge_bucket(object.unique_id, sg_id, object.edge, object.edge_weight);
        indexed.recompute_knn_metadata(object.unique_id, subgraphs);
    }
    indexed.finalize_updates();
    return indexed;
}

std::size_t IndexedMovingObjectSet::size() const {
    return objects_.size();
}

const MovingObject& IndexedMovingObjectSet::operator[](ObjId id) const {
    return objects_.at(static_cast<std::size_t>(id));
}

const std::vector<MovingObject>& IndexedMovingObjectSet::objects() const {
    return objects_;
}

const std::vector<ObjId>& IndexedMovingObjectSet::objects_incident_to(VertexId vertex) const {
    const auto it = incident_objects_.find(vertex);
    return (it == incident_objects_.end()) ? empty_ids() : it->second;
}

const std::vector<ObjId>& IndexedMovingObjectSet::objects_in(SgId sg_id) const {
    if (sg_id >= subgraph_objects_.size()) {
        return empty_ids();
    }
    return subgraph_objects_[sg_id];
}

const std::vector<ObjId>& IndexedMovingObjectSet::objects_in_knn_order(SgId sg_id) const {
    if (sg_id >= knn_sorted_objects_.size()) {
        return empty_ids();
    }
    return knn_sorted_objects_[sg_id];
}

const std::vector<PopulatedEdgeObjects>& IndexedMovingObjectSet::populated_edges_in(SgId sg_id) const {
    static const std::vector<PopulatedEdgeObjects> empty;
    if (sg_id >= subgraph_populated_edges_.size()) {
        return empty;
    }
    return subgraph_populated_edges_[sg_id];
}

EdgeWeight IndexedMovingObjectSet::knn_suffix(ObjId id) const {
    return knn_suffix_[static_cast<std::size_t>(id)];
}

BorderCostSpan IndexedMovingObjectSet::knn_border_costs(ObjId id) const {
    const auto& [offset, length] = knn_border_costs_span_[static_cast<std::size_t>(id)];
    const auto* base = knn_border_costs_arena_.data() + offset;
    return BorderCostSpan{base, base + length};
}

void IndexedMovingObjectSet::prefetch_knn_metadata(ObjId id) const {
    const auto i = static_cast<std::size_t>(id);
    __builtin_prefetch(&knn_suffix_[i]);
    __builtin_prefetch(&knn_border_costs_span_[i]);
    const auto offset = knn_border_costs_span_[i].first;
    __builtin_prefetch(knn_border_costs_arena_.data() + offset);
}

SgId IndexedMovingObjectSet::object_subgraph(ObjId id) const {
    return object_subgraph_.at(static_cast<std::size_t>(id));
}

void IndexedMovingObjectSet::move_object(
    MovingObject updated_object,
    const std::unordered_map<Edge, SgId, PairHash>& edge_to_subgraph,
    const std::vector<Subgraph>& subgraphs
) {
    if (updated_object.unique_id >= objects_.size()) {
        throw std::runtime_error("object id out of range");
    }

    const auto id = updated_object.unique_id;
    const auto old_sg = object_subgraph_[id];
    const auto old_object = objects_[id];
    if (old_sg == kInvalidSubgraphId) {
        throw std::runtime_error("object does not belong to any subgraph");
    }

    updated_object.edge = ordered_edge(updated_object.edge.first, updated_object.edge.second);
    const auto new_sg = resolve_subgraph(updated_object.edge, edge_to_subgraph);
    const auto old_edge = objects_[id].edge;

    if (old_sg != new_sg || old_edge != updated_object.edge) {
        remove_object_from_edge_bucket(id, old_sg);
    }
    if (old_sg != new_sg) {
        remove_object_from_subgraph(id, old_sg);
    }

    remove_object_from_incidence(id, old_object);
    objects_[id] = updated_object;
    add_object_to_incidence(id, objects_[id]);

    if (old_sg != new_sg) {
        add_object_to_subgraph(id, new_sg);
    }
    if (old_sg != new_sg || old_edge != updated_object.edge) {
        add_object_to_edge_bucket(id, new_sg, updated_object.edge, updated_object.edge_weight);
    }

    object_subgraph_[id] = new_sg;
    recompute_knn_metadata(id, subgraphs);
    mark_subgraph_dirty(new_sg);
    if (old_sg != new_sg) {
        mark_subgraph_dirty(old_sg);
    }
}

void IndexedMovingObjectSet::finalize_updates() {
    if (knn_border_costs_arena_.size() > 2U * knn_border_costs_live_entries_ + 1024U) {
        std::vector<std::pair<std::uint32_t, EdgeWeight>> compacted;
        compacted.reserve(knn_border_costs_live_entries_);
        for (auto& [offset, length] : knn_border_costs_span_) {
            const auto new_offset = static_cast<std::uint32_t>(compacted.size());
            compacted.insert(
                compacted.end(),
                knn_border_costs_arena_.begin() + offset,
                knn_border_costs_arena_.begin() + offset + length
            );
            offset = new_offset;
        }
        knn_border_costs_arena_ = std::move(compacted);
    }
    for (SgId sg_id = 0; sg_id < knn_sorted_dirty_.size(); ++sg_id) {
        if (!knn_sorted_dirty_[sg_id]) {
            continue;
        }
        auto& ordered = knn_sorted_objects_[sg_id];
        ordered = subgraph_objects_[sg_id];
        std::sort(ordered.begin(), ordered.end(), [&](ObjId lhs, ObjId rhs) {
            if (knn_suffix_[lhs] != knn_suffix_[rhs]) {
                return knn_suffix_[lhs] < knn_suffix_[rhs];
            }
            return lhs < rhs;
        });
        // Materialize the knn-order SoA (suffix + border-cost rows) so the query
        // refinement scan is fully sequential.
        auto& suf = knn_sorted_suffix_[sg_id];
        auto& rowptr = knn_sorted_cost_rowptr_[sg_id];
        auto& flat = knn_sorted_cost_flat_[sg_id];
        suf.clear();
        suf.reserve(ordered.size());
        rowptr.clear();
        rowptr.reserve(ordered.size() + 1U);
        rowptr.push_back(0U);
        flat.clear();
        // First pass over the (cost-sorted) rows: build the SoA and find the
        // largest cheapest-cost border index so the b0 groups can be sized.
        std::uint32_t max_b0 = 0;
        bool any_b0 = false;
        for (const auto obj_id : ordered) {
            suf.push_back(knn_suffix_[obj_id]);
            const auto& [offset, length] = knn_border_costs_span_[obj_id];
            flat.insert(
                flat.end(),
                knn_border_costs_arena_.begin() + offset,
                knn_border_costs_arena_.begin() + offset + length
            );
            rowptr.push_back(static_cast<std::uint32_t>(flat.size()));
            if (length != 0U) {
                const auto b0 = knn_border_costs_arena_[offset].first;
                max_b0 = std::max(max_b0, b0);
                any_b0 = true;
            }
        }
        // Group objects by b0 into CSR form (counting sort over local indices).
        auto& goff = knn_b0_group_offsets_[sg_id];
        auto& gsuf = knn_b0_group_suffix_[sg_id];
        goff.assign(any_b0 ? static_cast<std::size_t>(max_b0) + 2U : 1U, 0U);
        for (std::size_t j = 0; j < ordered.size(); ++j) {
            if (rowptr[j + 1U] != rowptr[j]) {
                const auto b0 = flat[rowptr[j]].first;
                ++goff[b0 + 1U];
            }
        }
        for (std::size_t i = 1; i < goff.size(); ++i) {
            goff[i] += goff[i - 1U];
        }
        gsuf.resize(goff.empty() ? 0U : goff.back());
        std::vector<std::uint32_t> cursor(goff.begin(), goff.end());
        for (std::size_t j = 0; j < ordered.size(); ++j) {
            if (rowptr[j + 1U] != rowptr[j]) {
                const auto& entry = flat[rowptr[j]];
                gsuf[cursor[entry.first]++] = entry.second;
            }
        }
        knn_sorted_dirty_[sg_id] = false;
    }
}

std::pair<const EdgeWeight*, const EdgeWeight*>
IndexedMovingObjectSet::knn_b0_group(SgId sg_id, std::uint32_t local_idx) const {
    const auto& goff = knn_b0_group_offsets_[sg_id];
    if (local_idx + 1U >= goff.size()) {
        return {nullptr, nullptr};
    }
    const auto* base = knn_b0_group_suffix_[sg_id].data();
    return {base + goff[local_idx], base + goff[local_idx + 1U]};
}

const std::vector<EdgeWeight>& IndexedMovingObjectSet::knn_order_suffixes(SgId sg_id) const {
    return knn_sorted_suffix_[sg_id];
}

BorderCostSpan IndexedMovingObjectSet::knn_order_costs(SgId sg_id, std::size_t order_index) const {
    const auto& rowptr = knn_sorted_cost_rowptr_[sg_id];
    const auto* base = knn_sorted_cost_flat_[sg_id].data();
    return BorderCostSpan{base + rowptr[order_index], base + rowptr[order_index + 1U]};
}

void IndexedMovingObjectSet::add_object_to_subgraph(ObjId id, SgId sg_id) {
    auto& ids = subgraph_objects_[sg_id];
    object_position_in_subgraph_[id] = ids.size();
    ids.push_back(id);
    object_subgraph_[id] = sg_id;
    mark_subgraph_dirty(sg_id);
}

void IndexedMovingObjectSet::remove_object_from_subgraph(ObjId id, SgId sg_id) {
    auto& ids = subgraph_objects_[sg_id];
    const auto pos = object_position_in_subgraph_[id];
    const auto last_id = ids.back();
    ids[pos] = last_id;
    object_position_in_subgraph_[last_id] = pos;
    ids.pop_back();
    object_position_in_subgraph_[id] = kInvalidIndex;
    mark_subgraph_dirty(sg_id);
}

void IndexedMovingObjectSet::add_object_to_edge_bucket(ObjId id, SgId sg_id, const Edge& edge, EdgeWeight edge_weight) {
    auto& lookup = populated_edge_index_[sg_id];
    auto& buckets = subgraph_populated_edges_[sg_id];
    const auto packed = pack_pair(edge.first, edge.second);
    auto it = lookup.find(packed);
    if (it == lookup.end()) {
        const auto idx = buckets.size();
        lookup.emplace(packed, idx);
        buckets.push_back(PopulatedEdgeObjects{edge, edge_weight, {}});
        it = lookup.find(packed);
    }

    auto& bucket = buckets[it->second];
    object_edge_bucket_index_[id] = it->second;
    object_position_in_edge_bucket_[id] = bucket.object_ids.size();
    bucket.object_ids.push_back(id);
}

void IndexedMovingObjectSet::remove_object_from_edge_bucket(ObjId id, SgId sg_id) {
    auto& buckets = subgraph_populated_edges_[sg_id];
    auto& lookup = populated_edge_index_[sg_id];
    const auto bucket_idx = object_edge_bucket_index_[id];
    auto& bucket = buckets[bucket_idx];
    const auto pos = object_position_in_edge_bucket_[id];
    const auto last_id = bucket.object_ids.back();
    bucket.object_ids[pos] = last_id;
    object_position_in_edge_bucket_[last_id] = pos;
    bucket.object_ids.pop_back();
    object_edge_bucket_index_[id] = kInvalidIndex;
    object_position_in_edge_bucket_[id] = kInvalidIndex;

    if (!bucket.object_ids.empty()) {
        return;
    }

    const auto removed_key = pack_pair(bucket.edge.first, bucket.edge.second);
    const auto last_bucket_idx = buckets.size() - 1;
    if (bucket_idx != last_bucket_idx) {
        buckets[bucket_idx] = std::move(buckets[last_bucket_idx]);
        const auto moved_key = pack_pair(buckets[bucket_idx].edge.first, buckets[bucket_idx].edge.second);
        lookup[moved_key] = bucket_idx;
        for (const auto moved_id : buckets[bucket_idx].object_ids) {
            object_edge_bucket_index_[moved_id] = bucket_idx;
        }
    }
    buckets.pop_back();
    lookup.erase(removed_key);
}

void IndexedMovingObjectSet::add_object_to_incidence(ObjId id, const MovingObject& object) {
    incident_objects_[object.edge.first].push_back(id);
    if (object.edge.second != object.edge.first) {
        incident_objects_[object.edge.second].push_back(id);
    }
}

void IndexedMovingObjectSet::remove_object_from_incidence(ObjId id, const MovingObject& object) {
    const auto remove_from_vertex = [&](VertexId vertex) {
        const auto it = incident_objects_.find(vertex);
        if (it == incident_objects_.end()) {
            return;
        }
        auto& ids = it->second;
        ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
        if (ids.empty()) {
            incident_objects_.erase(it);
        }
    };
    remove_from_vertex(object.edge.first);
    if (object.edge.second != object.edge.first) {
        remove_from_vertex(object.edge.second);
    }
}

void IndexedMovingObjectSet::recompute_knn_metadata(ObjId id, const std::vector<Subgraph>& subgraphs) {
    auto& span = knn_border_costs_span_[id];
    knn_border_costs_live_entries_ -= span.second;
    if (const auto sg_id = object_subgraph_[id]; sg_id == kInvalidSubgraphId) {
        knn_suffix_[id] = kInfWeight;
        span = {0U, 0U};
        return;
    } else {
        std::vector<std::pair<std::uint32_t, EdgeWeight>> row;
        compute_knn_metadata(objects_[id], subgraphs.at(sg_id), knn_suffix_[id], row);
        span.first = static_cast<std::uint32_t>(knn_border_costs_arena_.size());
        span.second = static_cast<std::uint32_t>(row.size());
        knn_border_costs_arena_.insert(knn_border_costs_arena_.end(), row.begin(), row.end());
        knn_border_costs_live_entries_ += span.second;
    }
}

void IndexedMovingObjectSet::mark_subgraph_dirty(SgId sg_id) {
    if (sg_id < knn_sorted_dirty_.size()) {
        knn_sorted_dirty_[sg_id] = true;
    }
}

}  // namespace bag
