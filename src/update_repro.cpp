#include "update_repro.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "distance.h"
#include "utils.h"

namespace bag {

namespace {

constexpr std::size_t kInvalidIndex = std::numeric_limits<std::size_t>::max();

struct BenchState {
    std::vector<MovingObject> objects;
    std::vector<std::pair<Edge, EdgeWeight>> move_slots;
};

struct ContinuousRangeQuery {
    Edge edge{};
    EdgeWeight edge_weight{0};
    EdgeWeight offset{0};
    EdgeWeight radius{0};
    DistanceMap dist;
    std::size_t result_count{0};
};

struct MoveChoice {
    Edge edge{};
    EdgeWeight weight{0};
    EdgeWeight offset{0};
};

BenchState make_bench_state(const Graph& graph, const UpdateOnlyBenchOptions& options) {
    BenchState state;
    auto object_set = MovingObjectSet::random_uniform(graph, options.object_count, options.object_seed);
    state.objects = object_set.objects();
    state.move_slots.reserve(graph.edge_count());
    for (const auto& [edge, weight] : graph.undirected_edges()) {
        if (weight > 1) {
            state.move_slots.push_back({ordered_edge(edge.first, edge.second), weight});
        }
    }
    if (state.move_slots.empty()) {
        throw std::runtime_error("graph does not have any valid move slots");
    }
    return state;
}

std::vector<ContinuousRangeQuery> make_continuous_queries(
    const Graph& graph,
    const UpdateOnlyBenchOptions& options,
    const std::vector<std::pair<Edge, EdgeWeight>>& move_slots
) {
    if (options.active_query_count == 0) {
        return {};
    }

    std::mt19937_64 rng(options.query_seed);
    std::uniform_int_distribution<std::size_t> slot_pick(0, move_slots.size() - 1U);
    std::vector<ContinuousRangeQuery> queries;
    queries.reserve(options.active_query_count);
    for (std::size_t i = 0; i < options.active_query_count; ++i) {
        const auto& [edge, weight] = move_slots[slot_pick(rng)];
        std::uniform_int_distribution<EdgeWeight> offset_dist(1, weight - 1U);
        const auto offset = offset_dist(rng);

        ContinuousRangeQuery query;
        query.edge = edge;
        query.edge_weight = weight;
        query.offset = offset;
        query.radius = options.range_radius;
        query.dist = dijkstra(
            graph,
            {
                {edge.first, offset},
                {edge.second, static_cast<EdgeWeight>(weight - offset)},
            },
            options.range_radius
        );
        queries.push_back(std::move(query));
    }
    return queries;
}

Coordinate interpolate_coordinate(
    const MovingObject& object,
    const std::unordered_map<VertexId, Coordinate>& coords
) {
    const auto left_it = coords.find(object.edge.first);
    const auto right_it = coords.find(object.edge.second);
    if (left_it == coords.end() || right_it == coords.end()) {
        throw std::runtime_error("missing coordinates for edge endpoint");
    }
    const auto& lhs = left_it->second;
    const auto& rhs = right_it->second;
    const double ratio = static_cast<double>(object.offset) /
        static_cast<double>(std::max<EdgeWeight>(1U, object.edge_weight));
    return Coordinate{
        lhs.x + (rhs.x - lhs.x) * ratio,
        lhs.y + (rhs.y - lhs.y) * ratio,
    };
}

bool object_within_query(const MovingObject& object, const ContinuousRangeQuery& query) {
    EdgeWeight best = kInfWeight;
    if (const auto it = query.dist.find(object.edge.first); it != query.dist.end() &&
        it->second <= kInfWeight - object.offset) {
        best = std::min(best, static_cast<EdgeWeight>(it->second + object.offset));
    }
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    if (const auto it = query.dist.find(object.edge.second); it != query.dist.end() &&
        it->second <= kInfWeight - right_cost) {
        best = std::min(best, static_cast<EdgeWeight>(it->second + right_cost));
    }
    return best <= query.radius;
}

MoveChoice sample_move_choice(
    const Graph& graph,
    const std::vector<std::pair<Edge, EdgeWeight>>& move_slots,
    const MovingObject& current,
    bool local_move,
    std::mt19937_64& rng
) {
    if (!local_move) {
        std::uniform_int_distribution<std::size_t> slot_pick(0, move_slots.size() - 1U);
        const auto& [edge, weight] = move_slots[slot_pick(rng)];
        std::uniform_int_distribution<EdgeWeight> offset_dist(1, weight - 1U);
        return MoveChoice{edge, weight, offset_dist(rng)};
    }

    std::vector<std::pair<Edge, EdgeWeight>> candidates;
    auto collect_from = [&](VertexId u) {
        for (const auto& [v, w] : graph.neighbors(u)) {
            if (w > 1) {
                const auto [a, b] = ordered_edge(u, v);
                candidates.push_back({{a, b}, w});
            }
        }
    };
    collect_from(current.edge.first);
    collect_from(current.edge.second);
    if (candidates.empty()) {
        std::uniform_int_distribution<std::size_t> slot_pick(0, move_slots.size() - 1U);
        const auto& [edge, weight] = move_slots[slot_pick(rng)];
        std::uniform_int_distribution<EdgeWeight> offset_dist(1, weight - 1U);
        return MoveChoice{edge, weight, offset_dist(rng)};
    }
    std::uniform_int_distribution<std::size_t> local_pick(0, candidates.size() - 1U);
    const auto& [edge, weight] = candidates[local_pick(rng)];
    std::uniform_int_distribution<EdgeWeight> offset_dist(1, weight - 1U);
    return MoveChoice{edge, weight, offset_dist(rng)};
}

template <typename Key, typename Hash = std::hash<Key>>
class BucketIndex {
public:
    std::size_t add(const Key& key, ObjId id) {
        auto& bucket = buckets_[key];
        bucket.push_back(id);
        return bucket.size() - 1U;
    }

    void remove(const Key& key, std::size_t position, std::vector<std::size_t>& reverse_positions) {
        auto it = buckets_.find(key);
        if (it == buckets_.end()) {
            throw std::runtime_error("bucket remove on missing key");
        }
        auto& bucket = it->second;
        if (position >= bucket.size()) {
            throw std::runtime_error("bucket remove position out of bounds");
        }
        const ObjId moved = bucket.back();
        bucket[position] = moved;
        reverse_positions[moved] = position;
        bucket.pop_back();
        if (bucket.empty()) {
            buckets_.erase(it);
        }
    }

    [[nodiscard]] std::size_t total_size() const {
        std::size_t total = 0;
        for (const auto& [key, bucket] : buckets_) {
            (void)key;
            total += bucket.size();
        }
        return total;
    }

    [[nodiscard]] std::size_t bucket_count() const {
        return buckets_.size();
    }

    [[nodiscard]] std::size_t bucket_size(const Key& key) const {
        const auto it = buckets_.find(key);
        return (it == buckets_.end()) ? 0U : it->second.size();
    }

private:
    std::unordered_map<Key, std::vector<ObjId>, Hash> buckets_;
};

class ArneUpdateOverlay {
public:
    explicit ArneUpdateOverlay(std::vector<MovingObject> objects)
        : objects_(std::move(objects)),
          edge_bucket_pos_(objects_.size(), kInvalidIndex),
          left_anchor_pos_(objects_.size(), kInvalidIndex),
          right_anchor_pos_(objects_.size(), kInvalidIndex) {
        for (const auto& object : objects_) {
            add_object(object.unique_id);
        }
    }

    void move_object(ObjId id, const Edge& edge, EdgeWeight weight, EdgeWeight offset) {
        remove_object(id);
        auto& object = objects_.at(static_cast<std::size_t>(id));
        object.edge = edge;
        object.edge_weight = weight;
        object.offset = offset;
        add_object(id);
    }

    void verify() const {
        if (edge_buckets_.total_size() != objects_.size()) {
            throw std::runtime_error("ARNE edge bucket size mismatch");
        }
        if (left_anchor_buckets_.total_size() != objects_.size()) {
            throw std::runtime_error("ARNE left anchor bucket size mismatch");
        }
        if (right_anchor_buckets_.total_size() != objects_.size()) {
            throw std::runtime_error("ARNE right anchor bucket size mismatch");
        }
    }

private:
    void add_object(ObjId id) {
        const auto& object = objects_.at(static_cast<std::size_t>(id));
        const auto edge_key = pack_pair(object.edge.first, object.edge.second);
        edge_bucket_pos_[id] = edge_buckets_.add(edge_key, id);
        left_anchor_pos_[id] = left_anchor_buckets_.add(object.edge.first, id);
        right_anchor_pos_[id] = right_anchor_buckets_.add(object.edge.second, id);
    }

    void remove_object(ObjId id) {
        const auto& object = objects_.at(static_cast<std::size_t>(id));
        const auto edge_key = pack_pair(object.edge.first, object.edge.second);
        if (edge_bucket_pos_[id] >= edge_buckets_.bucket_size(edge_key)) {
            throw std::runtime_error("ARNE edge bucket position mismatch");
        }
        if (left_anchor_pos_[id] >= left_anchor_buckets_.bucket_size(object.edge.first)) {
            throw std::runtime_error("ARNE left anchor bucket position mismatch");
        }
        if (right_anchor_pos_[id] >= right_anchor_buckets_.bucket_size(object.edge.second)) {
            throw std::runtime_error("ARNE right anchor bucket position mismatch");
        }
        edge_buckets_.remove(edge_key, edge_bucket_pos_[id], edge_bucket_pos_);
        left_anchor_buckets_.remove(object.edge.first, left_anchor_pos_[id], left_anchor_pos_);
        right_anchor_buckets_.remove(object.edge.second, right_anchor_pos_[id], right_anchor_pos_);
    }

    std::vector<MovingObject> objects_;
    BucketIndex<std::uint64_t> edge_buckets_;
    BucketIndex<VertexId> left_anchor_buckets_;
    BucketIndex<VertexId> right_anchor_buckets_;
    std::vector<std::size_t> edge_bucket_pos_;
    std::vector<std::size_t> left_anchor_pos_;
    std::vector<std::size_t> right_anchor_pos_;
};

class AmovnetUpdateOverlay {
public:
    AmovnetUpdateOverlay(
        std::vector<MovingObject> objects,
        const std::unordered_map<VertexId, Coordinate>& coords,
        std::size_t grid_side
    )
        : objects_(std::move(objects)),
          coords_(coords),
          grid_side_(std::max<std::size_t>(1U, grid_side)),
          edge_bucket_pos_(objects_.size(), kInvalidIndex),
          cell_bucket_pos_(objects_.size(), kInvalidIndex) {
        initialize_bbox();
        for (const auto& object : objects_) {
            add_object(object.unique_id);
        }
    }

    void move_object(ObjId id, const Edge& edge, EdgeWeight weight, EdgeWeight offset) {
        remove_object(id);
        auto& object = objects_.at(static_cast<std::size_t>(id));
        object.edge = edge;
        object.edge_weight = weight;
        object.offset = offset;
        add_object(id);
    }

    void verify() const {
        if (edge_buckets_.total_size() != objects_.size()) {
            throw std::runtime_error("AMOVNet edge bucket size mismatch");
        }
        if (cell_buckets_.total_size() != objects_.size()) {
            throw std::runtime_error("AMOVNet cell bucket size mismatch");
        }
    }

    [[nodiscard]] const std::vector<MovingObject>& objects() const {
        return objects_;
    }

private:
    void initialize_bbox() {
        bool first = true;
        for (const auto& [vertex, coord] : coords_) {
            (void)vertex;
            if (first) {
                min_x_ = max_x_ = coord.x;
                min_y_ = max_y_ = coord.y;
                first = false;
                continue;
            }
            min_x_ = std::min(min_x_, coord.x);
            max_x_ = std::max(max_x_, coord.x);
            min_y_ = std::min(min_y_, coord.y);
            max_y_ = std::max(max_y_, coord.y);
        }
        if (first) {
            throw std::runtime_error("empty coordinate set for AMOVNet");
        }
        if (max_x_ <= min_x_) {
            max_x_ = min_x_ + 1.0;
        }
        if (max_y_ <= min_y_) {
            max_y_ = min_y_ + 1.0;
        }
    }

    std::uint64_t cell_id_for(const MovingObject& object) const {
        const auto coord = interpolate_coordinate(object, coords_);
        const double nx = (coord.x - min_x_) / (max_x_ - min_x_);
        const double ny = (coord.y - min_y_) / (max_y_ - min_y_);
        const auto gx = static_cast<std::uint32_t>(std::min<double>(
            static_cast<double>(grid_side_ - 1U),
            std::max<double>(0.0, std::floor(nx * static_cast<double>(grid_side_)))
        ));
        const auto gy = static_cast<std::uint32_t>(std::min<double>(
            static_cast<double>(grid_side_ - 1U),
            std::max<double>(0.0, std::floor(ny * static_cast<double>(grid_side_)))
        ));
        return (static_cast<std::uint64_t>(gx) << 32U) | static_cast<std::uint64_t>(gy);
    }

    void add_object(ObjId id) {
        const auto& object = objects_.at(static_cast<std::size_t>(id));
        const auto edge_key = pack_pair(object.edge.first, object.edge.second);
        const auto cell_key = cell_id_for(object);
        edge_bucket_pos_[id] = edge_buckets_.add(edge_key, id);
        cell_bucket_pos_[id] = cell_buckets_.add(cell_key, id);
    }

    void remove_object(ObjId id) {
        const auto& object = objects_.at(static_cast<std::size_t>(id));
        const auto edge_key = pack_pair(object.edge.first, object.edge.second);
        const auto cell_key = cell_id_for(object);
        edge_buckets_.remove(edge_key, edge_bucket_pos_[id], edge_bucket_pos_);
        cell_buckets_.remove(cell_key, cell_bucket_pos_[id], cell_bucket_pos_);
    }

    std::vector<MovingObject> objects_;
    const std::unordered_map<VertexId, Coordinate>& coords_;
    std::size_t grid_side_{0};
    double min_x_{0.0};
    double max_x_{1.0};
    double min_y_{0.0};
    double max_y_{1.0};
    BucketIndex<std::uint64_t> edge_buckets_;
    BucketIndex<std::uint64_t> cell_buckets_;
    std::vector<std::size_t> edge_bucket_pos_;
    std::vector<std::size_t> cell_bucket_pos_;
};

class AmovnetQueryAwareOverlay {
public:
    AmovnetQueryAwareOverlay(
        const Graph& graph,
        std::vector<MovingObject> objects,
        const std::unordered_map<VertexId, Coordinate>& coords,
        const UpdateOnlyBenchOptions& options,
        const std::vector<std::pair<Edge, EdgeWeight>>& move_slots
    )
        : object_overlay_(std::move(objects), coords, options.grid_side),
          objects_(object_overlay_.objects()),
          queries_(make_continuous_queries(graph, options, move_slots)) {
        initialize_query_results();
    }

    void move_object(ObjId id, const Edge& edge, EdgeWeight weight, EdgeWeight offset) {
        const auto old_object = objects_.at(static_cast<std::size_t>(id));
        MovingObject new_object = old_object;
        new_object.edge = edge;
        new_object.edge_weight = weight;
        new_object.offset = offset;

        object_overlay_.move_object(id, edge, weight, offset);
        objects_[static_cast<std::size_t>(id)] = new_object;

        for (auto& query : queries_) {
            const bool old_in = object_within_query(old_object, query);
            const bool new_in = object_within_query(new_object, query);
            if (old_in && !new_in) {
                if (query.result_count == 0) {
                    throw std::runtime_error("AMOVNet query-aware result count underflow");
                }
                --query.result_count;
            } else if (!old_in && new_in) {
                ++query.result_count;
            }
        }
    }

    void verify() const {
        object_overlay_.verify();
        for (const auto& query : queries_) {
            std::size_t count = 0;
            for (const auto& object : objects_) {
                if (object_within_query(object, query)) {
                    ++count;
                }
            }
            if (count != query.result_count) {
                throw std::runtime_error("AMOVNet query-aware result count mismatch");
            }
        }
    }

private:
    void initialize_query_results() {
        for (auto& query : queries_) {
            std::size_t count = 0;
            for (const auto& object : objects_) {
                if (object_within_query(object, query)) {
                    ++count;
                }
            }
            query.result_count = count;
        }
    }

    AmovnetUpdateOverlay object_overlay_;
    std::vector<MovingObject> objects_;
    std::vector<ContinuousRangeQuery> queries_;
};

class AmovnetQueryIndexedOverlay {
public:
    AmovnetQueryIndexedOverlay(
        const Graph& graph,
        std::vector<MovingObject> objects,
        const std::unordered_map<VertexId, Coordinate>& coords,
        const UpdateOnlyBenchOptions& options,
        const std::vector<std::pair<Edge, EdgeWeight>>& move_slots
    )
        : object_overlay_(std::move(objects), coords, options.grid_side),
          objects_(object_overlay_.objects()),
          queries_(make_continuous_queries(graph, options, move_slots)),
          query_seen_(queries_.size(), 0) {
        build_edge_query_index(graph);
        initialize_query_results();
    }

    void move_object(ObjId id, const Edge& edge, EdgeWeight weight, EdgeWeight offset) {
        const auto old_object = objects_.at(static_cast<std::size_t>(id));
        MovingObject new_object = old_object;
        new_object.edge = edge;
        new_object.edge_weight = weight;
        new_object.offset = offset;

        object_overlay_.move_object(id, edge, weight, offset);
        objects_[static_cast<std::size_t>(id)] = new_object;

        affected_queries_.clear();
        ++query_seen_generation_;
        if (query_seen_generation_ == 0) {
            std::fill(query_seen_.begin(), query_seen_.end(), 0U);
            query_seen_generation_ = 1U;
        }
        append_affected_queries(old_object.edge);
        append_affected_queries(new_object.edge);

        for (const auto query_id : affected_queries_) {
            auto& query = queries_[query_id];
            const bool old_in = object_within_query(old_object, query);
            const bool new_in = object_within_query(new_object, query);
            if (old_in && !new_in) {
                if (query.result_count == 0) {
                    throw std::runtime_error("AMOVNet indexed result count underflow");
                }
                --query.result_count;
            } else if (!old_in && new_in) {
                ++query.result_count;
            }
        }
    }

    void verify() const {
        object_overlay_.verify();
        for (const auto& query : queries_) {
            std::size_t count = 0;
            for (const auto& object : objects_) {
                if (object_within_query(object, query)) {
                    ++count;
                }
            }
            if (count != query.result_count) {
                throw std::runtime_error("AMOVNet indexed result count mismatch");
            }
        }
    }

private:
    void append_affected_queries(const Edge& edge) {
        const auto key = pack_pair(edge.first, edge.second);
        const auto it = edge_to_queries_.find(key);
        if (it == edge_to_queries_.end()) {
            return;
        }
        for (const auto qid : it->second) {
            if (query_seen_[qid] != query_seen_generation_) {
                query_seen_[qid] = query_seen_generation_;
                affected_queries_.push_back(qid);
            }
        }
    }

    void build_edge_query_index(const Graph& graph) {
        for (std::size_t qid = 0; qid < queries_.size(); ++qid) {
            const auto& query = queries_[qid];
            for (const auto& [u, du] : query.dist) {
                if (du > query.radius - 1U) {
                    continue;
                }
                for (const auto& [v, w] : graph.neighbors(u)) {
                    if (w <= 1U) {
                        continue;
                    }
                    const auto [a, b] = ordered_edge(u, v);
                    edge_to_queries_[pack_pair(a, b)].push_back(qid);
                }
            }
        }
    }

    void initialize_query_results() {
        for (auto& query : queries_) {
            std::size_t count = 0;
            for (const auto& object : objects_) {
                if (object_within_query(object, query)) {
                    ++count;
                }
            }
            query.result_count = count;
        }
    }

    AmovnetUpdateOverlay object_overlay_;
    std::vector<MovingObject> objects_;
    std::vector<ContinuousRangeQuery> queries_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> edge_to_queries_;
    mutable std::vector<std::uint32_t> query_seen_;
    std::uint32_t query_seen_generation_{1U};
    std::vector<std::size_t> affected_queries_;
};

template <typename OverlayBuilder, typename MoveFn>
UpdateOnlyBenchResult run_benchmark(
    const std::string& baseline,
    const Graph& graph,
    const UpdateOnlyBenchOptions& options,
    OverlayBuilder&& builder,
    MoveFn&& move_fn
) {
    auto state = make_bench_state(graph, options);
    auto overlay = builder(std::move(state.objects));

    std::mt19937_64 rng(options.move_seed);
    std::uniform_int_distribution<std::size_t> object_pick(0, options.object_count - 1U);
    std::uniform_int_distribution<std::size_t> slot_pick(0, state.move_slots.size() - 1U);

    long long total_update_us = 0;
    for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
        total_update_us += calc_execution_time_in_us([&] {
            for (std::size_t i = 0; i < options.change_count; ++i) {
                const auto obj_index = static_cast<ObjId>(object_pick(rng));
                const auto& [edge, weight] = state.move_slots[slot_pick(rng)];
                std::uniform_int_distribution<EdgeWeight> offset_dist(1, weight - 1U);
                move_fn(overlay, obj_index, edge, weight, offset_dist(rng));
            }
        });
    }
    overlay.verify();

    UpdateOnlyBenchResult result;
    result.baseline = baseline;
    result.object_count = options.object_count;
    result.change_count = options.change_count;
    result.epochs = options.epochs;
    result.grid_side = options.grid_side;
    result.active_query_count = options.active_query_count;
    result.range_radius = options.range_radius;
    result.total_update_us = total_update_us;
    result.avg_update_us = static_cast<double>(total_update_us) /
        static_cast<double>(std::max<std::size_t>(1U, options.epochs));
    return result;
}

}  // namespace

std::unordered_map<VertexId, Coordinate> load_coordinates_from_file(const std::string& file) {
    std::ifstream in(file);
    if (!in) {
        throw std::runtime_error("failed to open coordinate file: " + file);
    }

    std::unordered_map<VertexId, Coordinate> coords;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        char op = '\0';
        ss >> op;
        if (op == 'c') {
            continue;
        }
        if (op == 'p') {
            continue;
        }
        if (op != 'v') {
            continue;
        }
        VertexId v = 0;
        double x = 0.0;
        double y = 0.0;
        ss >> v >> x >> y;
        coords[v] = Coordinate{x, y};
    }
    if (coords.empty()) {
        throw std::runtime_error("no coordinates found in file: " + file);
    }
    return coords;
}

UpdateOnlyBenchResult benchmark_arne_update_only(
    const Graph& graph,
    const UpdateOnlyBenchOptions& options
) {
    return run_benchmark(
        "ARNE",
        graph,
        options,
        [](std::vector<MovingObject> objects) {
            return ArneUpdateOverlay(std::move(objects));
        },
        [](ArneUpdateOverlay& overlay, ObjId id, const Edge& edge, EdgeWeight weight, EdgeWeight offset) {
            overlay.move_object(id, edge, weight, offset);
        }
    );
}

UpdateOnlyBenchResult benchmark_amovnet_update_only(
    const Graph& graph,
    const std::unordered_map<VertexId, Coordinate>& coords,
    const UpdateOnlyBenchOptions& options
) {
    return run_benchmark(
        "AMOVNet",
        graph,
        options,
        [&](std::vector<MovingObject> objects) {
            return AmovnetUpdateOverlay(std::move(objects), coords, options.grid_side);
        },
        [](AmovnetUpdateOverlay& overlay, ObjId id, const Edge& edge, EdgeWeight weight, EdgeWeight offset) {
            overlay.move_object(id, edge, weight, offset);
        }
    );
}

UpdateOnlyBenchResult benchmark_amovnet_query_aware_update_only(
    const Graph& graph,
    const std::unordered_map<VertexId, Coordinate>& coords,
    const UpdateOnlyBenchOptions& options
) {
    auto state = make_bench_state(graph, options);
    auto objects = std::move(state.objects);
    auto current_objects = objects;
    auto move_slots = std::move(state.move_slots);

    auto overlay = AmovnetQueryAwareOverlay(graph, std::move(objects), coords, options, move_slots);

    std::mt19937_64 rng(options.move_seed);
    std::uniform_int_distribution<std::size_t> object_pick(0, options.object_count - 1U);

    long long total_update_us = 0;
    for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
        total_update_us += calc_execution_time_in_us([&] {
            for (std::size_t i = 0; i < options.change_count; ++i) {
                const auto obj_index = static_cast<ObjId>(object_pick(rng));
                const auto choice = sample_move_choice(
                    graph,
                    move_slots,
                    current_objects[static_cast<std::size_t>(obj_index)],
                    options.local_move,
                    rng
                );
                overlay.move_object(obj_index, choice.edge, choice.weight, choice.offset);
                auto& object = current_objects[static_cast<std::size_t>(obj_index)];
                object.edge = choice.edge;
                object.edge_weight = choice.weight;
                object.offset = choice.offset;
            }
        });
    }
    overlay.verify();

    UpdateOnlyBenchResult result;
    result.baseline = "AMOVNet(query-aware)";
    result.object_count = options.object_count;
    result.change_count = options.change_count;
    result.epochs = options.epochs;
    result.grid_side = options.grid_side;
    result.active_query_count = options.active_query_count;
    result.range_radius = options.range_radius;
    result.total_update_us = total_update_us;
    result.avg_update_us = static_cast<double>(total_update_us) /
        static_cast<double>(std::max<std::size_t>(1U, options.epochs));
    return result;
}

UpdateOnlyBenchResult benchmark_amovnet_query_indexed_update_only(
    const Graph& graph,
    const std::unordered_map<VertexId, Coordinate>& coords,
    const UpdateOnlyBenchOptions& options
) {
    auto state = make_bench_state(graph, options);
    auto objects = std::move(state.objects);
    auto current_objects = objects;
    auto move_slots = std::move(state.move_slots);

    auto overlay = AmovnetQueryIndexedOverlay(graph, std::move(objects), coords, options, move_slots);

    std::mt19937_64 rng(options.move_seed);
    std::uniform_int_distribution<std::size_t> object_pick(0, options.object_count - 1U);

    long long total_update_us = 0;
    for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
        total_update_us += calc_execution_time_in_us([&] {
            for (std::size_t i = 0; i < options.change_count; ++i) {
                const auto obj_index = static_cast<ObjId>(object_pick(rng));
                const auto choice = sample_move_choice(
                    graph,
                    move_slots,
                    current_objects[static_cast<std::size_t>(obj_index)],
                    options.local_move,
                    rng
                );
                overlay.move_object(obj_index, choice.edge, choice.weight, choice.offset);
                auto& object = current_objects[static_cast<std::size_t>(obj_index)];
                object.edge = choice.edge;
                object.edge_weight = choice.weight;
                object.offset = choice.offset;
            }
        });
    }
    overlay.verify();

    UpdateOnlyBenchResult result;
    result.baseline = options.local_move ? "AMOVNet(query-indexed,local)" : "AMOVNet(query-indexed)";
    result.object_count = options.object_count;
    result.change_count = options.change_count;
    result.epochs = options.epochs;
    result.grid_side = options.grid_side;
    result.active_query_count = options.active_query_count;
    result.range_radius = options.range_radius;
    result.total_update_us = total_update_us;
    result.avg_update_us = static_cast<double>(total_update_us) /
        static_cast<double>(std::max<std::size_t>(1U, options.epochs));
    return result;
}

}  // namespace bag
