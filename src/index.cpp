#include "index.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <string_view>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include "distance.h"

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef BAG_ENABLE_KNN_FINE_TIMING
#define BAG_ENABLE_KNN_FINE_TIMING 0
#endif

namespace bag {

namespace {

constexpr bool kEnableKnnFineTiming = BAG_ENABLE_KNN_FINE_TIMING != 0;
constexpr std::uint64_t kIndexCacheMagic = 0x4241474944584331ULL; // BAGIDXC1
constexpr std::uint32_t kIndexCacheVersion = 1U;

std::uint64_t fnv1a64_append(std::uint64_t hash, std::string_view text) {
    constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
    for (const auto ch : text) {
        hash ^= static_cast<std::uint8_t>(ch);
        hash *= kFnvPrime;
    }
    return hash;
}

std::uint64_t fnv1a64_seed() {
    return 1469598103934665603ULL;
}

std::string hex_u64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << value;
    return out.str();
}

template <typename T>
void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("failed to write index cache payload");
    }
}

template <typename T>
T read_pod(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("failed to read index cache payload");
    }
    return value;
}

void write_string(std::ostream& out, const std::string& value) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!out) {
        throw std::runtime_error("failed to write index cache string");
    }
}

std::string read_string(std::istream& in) {
    const auto size = read_pod<std::uint64_t>(in);
    std::string value(size, '\0');
    in.read(value.data(), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("failed to read index cache string");
    }
    return value;
}

void write_graph(std::ostream& out, const Graph& graph) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(graph.vertex_set().size()));
    for (const auto v : graph.vertex_set()) {
        write_pod<VertexId>(out, v);
    }
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(graph.edge_count()));
    for (const auto u : graph.vertex_set()) {
        const auto& row = graph.neighbors(u);
        for (const auto& [v, w] : row) {
            write_pod<VertexId>(out, u);
            write_pod<VertexId>(out, v);
            write_pod<EdgeWeight>(out, w);
        }
    }
}

Graph read_graph(std::istream& in) {
    Graph graph;
    const auto vertex_count = read_pod<std::uint64_t>(in);
    for (std::uint64_t i = 0; i < vertex_count; ++i) {
        graph.insert(read_pod<VertexId>(in));
    }
    const auto edge_count = read_pod<std::uint64_t>(in);
    for (std::uint64_t i = 0; i < edge_count; ++i) {
        const auto u = read_pod<VertexId>(in);
        const auto v = read_pod<VertexId>(in);
        const auto w = read_pod<EdgeWeight>(in);
        graph.add_directed_edge(u, v, w);
    }
    return graph;
}

template <typename T>
void write_pod_vector(std::ostream& out, const std::vector<T>& values) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) {
        write_pod<T>(out, value);
    }
}

template <typename T>
std::vector<T> read_pod_vector(std::istream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::vector<T> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        values.push_back(read_pod<T>(in));
    }
    return values;
}

void write_vertex_weight_pairs(std::ostream& out, const std::vector<std::pair<VertexId, EdgeWeight>>& row) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(row.size()));
    for (const auto& [v, w] : row) {
        write_pod<VertexId>(out, v);
        write_pod<EdgeWeight>(out, w);
    }
}

std::vector<std::pair<VertexId, EdgeWeight>> read_vertex_weight_pairs(std::istream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::vector<std::pair<VertexId, EdgeWeight>> row;
    row.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        row.push_back({read_pod<VertexId>(in), read_pod<EdgeWeight>(in)});
    }
    return row;
}

void write_u32_weight_pairs(std::ostream& out, const std::vector<std::pair<std::uint32_t, EdgeWeight>>& row) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(row.size()));
    for (const auto& [v, w] : row) {
        write_pod<std::uint32_t>(out, v);
        write_pod<EdgeWeight>(out, w);
    }
}

std::vector<std::pair<std::uint32_t, EdgeWeight>> read_u32_weight_pairs(std::istream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::vector<std::pair<std::uint32_t, EdgeWeight>> row;
    row.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        row.push_back({read_pod<std::uint32_t>(in), read_pod<EdgeWeight>(in)});
    }
    return row;
}

void write_nested_vertex_weight_rows(
    std::ostream& out,
    const std::vector<std::vector<std::pair<VertexId, EdgeWeight>>>& rows
) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(rows.size()));
    for (const auto& row : rows) {
        write_vertex_weight_pairs(out, row);
    }
}

std::vector<std::vector<std::pair<VertexId, EdgeWeight>>> read_nested_vertex_weight_rows(std::istream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::vector<std::vector<std::pair<VertexId, EdgeWeight>>> rows;
    rows.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        rows.push_back(read_vertex_weight_pairs(in));
    }
    return rows;
}

void write_vector_of_vertex_lists(std::ostream& out, const std::vector<std::vector<VertexId>>& values) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& row : values) {
        write_pod_vector(out, row);
    }
}

std::vector<std::vector<VertexId>> read_vector_of_vertex_lists(std::istream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::vector<std::vector<VertexId>> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        values.push_back(read_pod_vector<VertexId>(in));
    }
    return values;
}

void write_vector_of_sgid_lists(std::ostream& out, const std::vector<std::vector<SgId>>& values) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& row : values) {
        write_pod_vector(out, row);
    }
}

std::vector<std::vector<SgId>> read_vector_of_sgid_lists(std::istream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::vector<std::vector<SgId>> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        values.push_back(read_pod_vector<SgId>(in));
    }
    return values;
}

void write_edge_to_subgraph_map(
    std::ostream& out,
    const std::unordered_map<Edge, SgId, PairHash>& values
) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& [edge, sg_id] : values) {
        write_pod<VertexId>(out, edge.first);
        write_pod<VertexId>(out, edge.second);
        write_pod<SgId>(out, sg_id);
    }
}

std::unordered_map<Edge, SgId, PairHash> read_edge_to_subgraph_map(std::istream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::unordered_map<Edge, SgId, PairHash> values;
    values.reserve(static_cast<std::size_t>(count * 2U + 1U));
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto u = read_pod<VertexId>(in);
        const auto v = read_pod<VertexId>(in);
        const auto sg_id = read_pod<SgId>(in);
        values.emplace(Edge{u, v}, sg_id);
    }
    return values;
}

void write_local_subgraph_index(std::ostream& out, const LocalSubgraphIndex& index) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(index.local_ids.size()));
    for (const auto& [vertex, local_id] : index.local_ids) {
        write_pod<VertexId>(out, vertex);
        write_pod<std::uint32_t>(out, local_id);
    }
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(index.adjacency.size()));
    for (const auto& row : index.adjacency) {
        write_u32_weight_pairs(out, row);
    }
}

LocalSubgraphIndex read_local_subgraph_index(std::istream& in) {
    LocalSubgraphIndex index;
    const auto local_id_count = read_pod<std::uint64_t>(in);
    index.local_ids.reserve(static_cast<std::size_t>(local_id_count * 2U + 1U));
    for (std::uint64_t i = 0; i < local_id_count; ++i) {
        index.local_ids.emplace(read_pod<VertexId>(in), read_pod<std::uint32_t>(in));
    }
    const auto adjacency_count = read_pod<std::uint64_t>(in);
    index.adjacency.reserve(static_cast<std::size_t>(adjacency_count));
    for (std::uint64_t i = 0; i < adjacency_count; ++i) {
        index.adjacency.push_back(read_u32_weight_pairs(in));
    }
    return index;
}

void write_local_subgraph_indices(std::ostream& out, const std::vector<LocalSubgraphIndex>& values) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) {
        write_local_subgraph_index(out, value);
    }
}

std::vector<LocalSubgraphIndex> read_local_subgraph_indices(std::istream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::vector<LocalSubgraphIndex> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        values.push_back(read_local_subgraph_index(in));
    }
    return values;
}

void write_factorized_transfer_model(std::ostream& out, const FactorizedTransferSubgraphModel& model) {
    write_pod<SgId>(out, model.subgraph_id);
    write_pod<std::uint8_t>(out, model.feasible ? 1U : 0U);
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(model.num_borders));
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(model.explicit_directed_arcs));
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(model.factorized_directed_arcs));
    write_pod<double>(out, model.factorized_arc_ratio);
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(model.greedy_hub_count));
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(model.greedy_internal_hub_count));
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(model.greedy_border_hub_count));
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(model.entry_rows.size()));
    for (const auto& [entry_border, rows] : model.entry_rows) {
        write_pod<VertexId>(out, entry_border);
        write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(rows.size()));
        for (const auto& row : rows) {
            write_pod<VertexId>(out, row.hub);
            write_pod<EdgeWeight>(out, row.entry_to_hub_distance);
            write_vertex_weight_pairs(out, row.exits);
        }
    }
}

FactorizedTransferSubgraphModel read_factorized_transfer_model(std::istream& in) {
    FactorizedTransferSubgraphModel model;
    model.subgraph_id = read_pod<SgId>(in);
    model.feasible = read_pod<std::uint8_t>(in) != 0U;
    model.num_borders = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    model.explicit_directed_arcs = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    model.factorized_directed_arcs = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    model.factorized_arc_ratio = read_pod<double>(in);
    model.greedy_hub_count = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    model.greedy_internal_hub_count = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    model.greedy_border_hub_count = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    const auto entry_count = read_pod<std::uint64_t>(in);
    model.entry_rows.reserve(static_cast<std::size_t>(entry_count * 2U + 1U));
    for (std::uint64_t i = 0; i < entry_count; ++i) {
        const auto entry_border = read_pod<VertexId>(in);
        const auto row_count = read_pod<std::uint64_t>(in);
        std::vector<FactorizedTransferHubRow> rows;
        rows.reserve(static_cast<std::size_t>(row_count));
        for (std::uint64_t j = 0; j < row_count; ++j) {
            FactorizedTransferHubRow row;
            row.hub = read_pod<VertexId>(in);
            row.entry_to_hub_distance = read_pod<EdgeWeight>(in);
            row.exits = read_vertex_weight_pairs(in);
            rows.push_back(std::move(row));
        }
        model.entry_rows.emplace(entry_border, std::move(rows));
    }
    return model;
}

void write_factorized_transfer_models(std::ostream& out, const std::vector<FactorizedTransferSubgraphModel>& values) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) {
        write_factorized_transfer_model(out, value);
    }
}

std::vector<FactorizedTransferSubgraphModel> read_factorized_transfer_models(std::istream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::vector<FactorizedTransferSubgraphModel> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        values.push_back(read_factorized_transfer_model(in));
    }
    return values;
}

std::filesystem::path temp_save_path(const std::filesystem::path& path) {
    return path.string() + ".tmp";
}

void commit_atomic_file(const std::filesystem::path& tmp_path, const std::filesystem::path& final_path) {
#ifdef _WIN32
    if (!MoveFileExW(
            tmp_path.wstring().c_str(),
            final_path.wstring().c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error("failed to atomically replace index cache file: " + final_path.string());
    }
#else
    std::filesystem::rename(tmp_path, final_path);
#endif
}

template <typename WriterFn>
void write_atomic_index_blob(const std::filesystem::path& path, WriterFn&& writer) {
    std::filesystem::create_directories(path.parent_path());
    const auto tmp_path = temp_save_path(path);
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to open index cache blob for write: " + tmp_path.string());
        }
        writer(out);
        out.flush();
        if (!out) {
            throw std::runtime_error("failed to flush index cache blob: " + tmp_path.string());
        }
    }
    commit_atomic_file(tmp_path, path);
}

EdgeWeight edge_weight_any(const Graph& graph, const Edge& edge) {
    if (const auto forward = graph.get_weight(edge.first, edge.second); forward.has_value()) {
        return *forward;
    }
    if (const auto reverse = graph.get_weight(edge.second, edge.first); reverse.has_value()) {
        return *reverse;
    }
    throw std::runtime_error("edge not found");
}

std::optional<EdgeWeight> edge_weight_any_or_null(const Graph& graph, const Edge& edge) {
    if (const auto forward = graph.get_weight(edge.first, edge.second); forward.has_value()) {
        return *forward;
    }
    if (const auto reverse = graph.get_weight(edge.second, edge.first); reverse.has_value()) {
        return *reverse;
    }
    return std::nullopt;
}

DistanceMap local_query_distances(const Subgraph& sg, const QueryPoint& query, EdgeWeight max_radius) {
    const auto edge = ordered_edge(query.edge.first, query.edge.second);
    const auto weight = edge_weight_any(sg.graph, edge);
    if (query.offset >= weight) {
        throw std::runtime_error("query offset must be smaller than edge weight");
    }
    return dijkstra(
        sg.graph,
        {
            {edge.first, query.offset},
            {edge.second, static_cast<EdgeWeight>(weight - query.offset)},
        },
        max_radius
    );
}

EdgeWeight lookup_local_distance(const LocalDijkstraResult& result, VertexId vertex) {
    if (result.index == nullptr) {
        return kInfWeight;
    }
    const auto it = result.index->local_ids.find(vertex);
    if (it == result.index->local_ids.end()) {
        return kInfWeight;
    }
    const auto local_id = static_cast<std::size_t>(it->second);
    if (local_id >= result.dist.size()) {
        return kInfWeight;
    }
    return result.dist[local_id];
}

LocalDijkstraResult local_dijkstra_compact(
    const LocalSubgraphIndex& local_index,
    const std::vector<std::pair<VertexId, EdgeWeight>>& seeds,
    EdgeWeight max_radius
) {
    using QueueItem = std::pair<EdgeWeight, std::uint32_t>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;

    LocalDijkstraResult result;
    result.index = &local_index;
    result.dist.assign(local_index.adjacency.size(), kInfWeight);

    for (const auto& [seed, value] : seeds) {
        const auto it = local_index.local_ids.find(seed);
        if (it == local_index.local_ids.end()) {
            continue;
        }
        const auto local_id = static_cast<std::size_t>(it->second);
        if (value < result.dist[local_id]) {
            if (result.dist[local_id] == kInfWeight) {
                ++result.reached_count;
            }
            result.dist[local_id] = value;
            pq.push({value, it->second});
        }
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();

        const auto local_u = static_cast<std::size_t>(u);
        if (local_u >= result.dist.size() || current != result.dist[local_u]) {
            continue;
        }
        if (current > max_radius) {
            continue;
        }

        for (const auto& [v, w] : local_index.adjacency[local_u]) {
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next > max_radius) {
                continue;
            }
            const auto local_v = static_cast<std::size_t>(v);
            if (next < result.dist[local_v]) {
                if (result.dist[local_v] == kInfWeight) {
                    ++result.reached_count;
                }
                result.dist[local_v] = next;
                pq.push({next, v});
            }
        }
    }

    return result;
}

LocalDijkstraResult local_query_distances_compact(
    const Subgraph& sg,
    const LocalSubgraphIndex& local_index,
    const QueryPoint& query,
    EdgeWeight max_radius
) {
    const auto edge = ordered_edge(query.edge.first, query.edge.second);
    const auto weight = edge_weight_any(sg.graph, edge);
    if (query.offset >= weight) {
        throw std::runtime_error("query offset must be smaller than edge weight");
    }
    return local_dijkstra_compact(
        local_index,
        {
            {edge.first, query.offset},
            {edge.second, static_cast<EdgeWeight>(weight - query.offset)},
        },
        max_radius
    );
}

EdgeWeight add_or_inf(EdgeWeight lhs, EdgeWeight rhs) {
    if (lhs == kInfWeight || rhs == kInfWeight || lhs > kInfWeight - rhs) {
        return kInfWeight;
    }
    return static_cast<EdgeWeight>(lhs + rhs);
}

EdgeWeight local_source_vertex_distance_from_table(
    const Subgraph& sg,
    VertexId source,
    VertexId target
) {
    return sg.distance.get_or_inf(source, target);
}

EdgeWeight local_query_vertex_distance_from_table(
    const Subgraph& sg,
    const Edge& query_edge,
    EdgeWeight query_offset,
    EdgeWeight query_edge_weight,
    VertexId target
) {
    EdgeWeight best = kInfWeight;
    const auto left_to_target = sg.distance.get_or_inf(query_edge.first, target);
    best = std::min(best, add_or_inf(query_offset, left_to_target));

    const auto right_seed = static_cast<EdgeWeight>(query_edge_weight - query_offset);
    const auto right_to_target = sg.distance.get_or_inf(query_edge.second, target);
    best = std::min(best, add_or_inf(right_seed, right_to_target));
    return best;
}

std::vector<std::pair<VertexId, EdgeWeight>> boundary_query_seeds(
    const Subgraph& initial_subgraph,
    const DistanceMap& local_dist
) {
    std::vector<std::pair<VertexId, EdgeWeight>> seeds;
    seeds.reserve(initial_subgraph.bound_vertices.size());
    for (const auto b : initial_subgraph.bound_vertices) {
        const auto it = local_dist.find(b);
        if (it != local_dist.end()) {
            seeds.push_back({b, it->second});
        }
    }
    return seeds;
}

std::vector<std::pair<VertexId, EdgeWeight>> boundary_query_seeds_from_source_table(
    const Subgraph& initial_subgraph,
    VertexId source
) {
    std::vector<std::pair<VertexId, EdgeWeight>> seeds;
    seeds.reserve(initial_subgraph.bound_vertices.size());
    for (const auto b : initial_subgraph.bound_vertices) {
        const auto distance = local_source_vertex_distance_from_table(initial_subgraph, source, b);
        if (distance != kInfWeight) {
            seeds.push_back({b, distance});
        }
    }
    return seeds;
}

std::vector<std::pair<VertexId, EdgeWeight>> boundary_query_seeds_from_edge_table(
    const Subgraph& initial_subgraph,
    const Edge& query_edge,
    EdgeWeight query_offset,
    EdgeWeight query_edge_weight
) {
    std::vector<std::pair<VertexId, EdgeWeight>> seeds;
    seeds.reserve(initial_subgraph.bound_vertices.size());
    for (const auto b : initial_subgraph.bound_vertices) {
        const auto distance = local_query_vertex_distance_from_table(
            initial_subgraph,
            query_edge,
            query_offset,
            query_edge_weight,
            b
        );
        if (distance != kInfWeight) {
            seeds.push_back({b, distance});
        }
    }
    return seeds;
}

std::vector<std::pair<VertexId, EdgeWeight>> boundary_query_seeds(
    const Subgraph& initial_subgraph,
    const LocalDijkstraResult& local_dist
) {
    std::vector<std::pair<VertexId, EdgeWeight>> seeds;
    seeds.reserve(initial_subgraph.bound_vertices.size());
    for (const auto b : initial_subgraph.bound_vertices) {
        const auto distance = lookup_local_distance(local_dist, b);
        if (distance != kInfWeight) {
            seeds.push_back({b, distance});
        }
    }
    return seeds;
}

std::vector<std::pair<VertexId, EdgeWeight>> boundary_row_from_local_index(
    const LocalSubgraphIndex& local_index,
    VertexId source,
    const std::vector<VertexId>& boundaries
) {
    const auto local_dist = local_dijkstra_compact(local_index, {{source, 0}}, kInfWeight);
    std::vector<std::pair<VertexId, EdgeWeight>> row;
    row.reserve(boundaries.size());
    for (const auto other : boundaries) {
        if (source == other) {
            continue;
        }
        const auto distance = lookup_local_distance(local_dist, other);
        if (distance != kInfWeight) {
            row.push_back({other, distance});
        }
    }
    std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
            return lhs.second < rhs.second;
        }
        return lhs.first < rhs.first;
    });
    return row;
}

DistanceMap boundary_query_distances(
    const Graph& skeleton,
    const Subgraph& initial_subgraph,
    const DistanceMap& local_dist,
    EdgeWeight max_radius
) {
    return dijkstra(skeleton, boundary_query_seeds(initial_subgraph, local_dist), max_radius);
}

struct DijkstraTrace {
    DistanceMap dist;
    std::vector<std::pair<VertexId, EdgeWeight>> settled;
    std::size_t relax_attempts{0};
    std::size_t successful_relaxes{0};
    std::size_t pq_pushes{0};
    std::size_t num_rows_truncated{0};
    std::size_t num_exits_skipped_by_truncation{0};
};

struct CliqueExitCandidate {
    VertexId border{kInvalidVertex};
    EdgeWeight local_distance{kInfWeight};
    EdgeWeight candidate_distance{kInfWeight};
};

struct MaterializedCliqueRow {
    std::vector<CliqueExitCandidate> exits;
    bool used_factorized{false};
    std::size_t hubs_used{0};
};

struct KnnFrontierItem {
    enum class Kind {
        Boundary,
        RowEmit,
    };

    EdgeWeight distance{kInfWeight};
    Kind kind{Kind::Boundary};
    VertexId vertex{kInvalidVertex};
    std::size_t row_id{0};
};

struct KnnFrontierGreater {
    bool operator()(const KnnFrontierItem& lhs, const KnnFrontierItem& rhs) const {
        if (lhs.distance != rhs.distance) {
            return lhs.distance > rhs.distance;
        }
        if (lhs.kind != rhs.kind) {
            return static_cast<int>(lhs.kind) > static_cast<int>(rhs.kind);
        }
        if (lhs.vertex != rhs.vertex) {
            return lhs.vertex > rhs.vertex;
        }
        return lhs.row_id > rhs.row_id;
    }
};

const std::vector<std::pair<VertexId, EdgeWeight>>& empty_sorted_row() {
    static const std::vector<std::pair<VertexId, EdgeWeight>> empty;
    return empty;
}

const std::vector<std::pair<VertexId, EdgeWeight>>& lookup_sorted_skeleton_row(
    VertexId border,
    const std::vector<std::int32_t>& row_index,
    const std::vector<std::vector<std::pair<VertexId, EdgeWeight>>>& rows
) {
    if (border >= row_index.size()) {
        return empty_sorted_row();
    }
    const auto idx = row_index[border];
    if (idx < 0) {
        return empty_sorted_row();
    }
    return rows[static_cast<std::size_t>(idx)];
}

const std::vector<std::pair<VertexId, EdgeWeight>>& lookup_subgraph_clique_row(
    VertexId entry_border,
    const std::unordered_map<VertexId, std::vector<std::pair<VertexId, EdgeWeight>>>& rows
) {
    const auto it = rows.find(entry_border);
    return (it == rows.end()) ? empty_sorted_row() : it->second;
}

std::vector<CliqueExitCandidate> materialize_clique_exit_candidates(
    const std::vector<std::pair<VertexId, EdgeWeight>>& local_row,
    EdgeWeight entry_distance,
    EdgeWeight max_distance = kInfWeight
) {
    std::vector<CliqueExitCandidate> exits;
    exits.reserve(local_row.size());
    for (const auto& [other_b, local] : local_row) {
        if (entry_distance == kInfWeight || local == kInfWeight || entry_distance > kInfWeight - local) {
            continue;
        }
        const auto candidate_distance = static_cast<EdgeWeight>(entry_distance + local);
        if (max_distance != kInfWeight && candidate_distance > max_distance) {
            continue;
        }
        exits.push_back(CliqueExitCandidate{
            other_b,
            local,
            candidate_distance,
        });
    }
    return exits;
}

MaterializedCliqueRow materialize_factorized_exit_candidates(
    const FactorizedTransferSubgraphModel& model,
    VertexId entry_border,
    EdgeWeight entry_distance,
    EdgeWeight max_distance = kInfWeight
) {
    MaterializedCliqueRow row;
    const auto it = model.entry_rows.find(entry_border);
    if (it == model.entry_rows.end()) {
        return row;
    }
    row.used_factorized = true;
    row.hubs_used = it->second.size();
    for (const auto& hub_row : it->second) {
        if (entry_distance == kInfWeight ||
            hub_row.entry_to_hub_distance == kInfWeight ||
            entry_distance > kInfWeight - hub_row.entry_to_hub_distance) {
            continue;
        }
        const auto base = static_cast<EdgeWeight>(entry_distance + hub_row.entry_to_hub_distance);
        for (const auto& [exit_border, hub_to_exit] : hub_row.exits) {
            if (hub_to_exit == kInfWeight || base > kInfWeight - hub_to_exit) {
                continue;
            }
            const auto candidate_distance = static_cast<EdgeWeight>(base + hub_to_exit);
            if (max_distance != kInfWeight && candidate_distance > max_distance) {
                continue;
            }
            row.exits.push_back(CliqueExitCandidate{
                exit_border,
                static_cast<EdgeWeight>(hub_row.entry_to_hub_distance + hub_to_exit),
                candidate_distance,
            });
        }
    }
    return row;
}

MaterializedCliqueRow materialize_subgraph_exit_candidates(
    VertexId entry_border,
    EdgeWeight entry_distance,
    SgId sg_id,
    const std::vector<std::unordered_map<VertexId, std::vector<std::pair<VertexId, EdgeWeight>>>>& explicit_rows,
    const std::vector<FactorizedTransferSubgraphModel>& factorized_models,
    const std::vector<std::uint8_t>& factorized_enabled,
    bool use_factorized_transfer,
    EdgeWeight max_distance = kInfWeight
) {
    if (use_factorized_transfer &&
        sg_id < factorized_enabled.size() &&
        factorized_enabled[sg_id] != 0U) {
        auto row = materialize_factorized_exit_candidates(
            factorized_models[sg_id],
            entry_border,
            entry_distance,
            max_distance
        );
        if (row.used_factorized) {
            return row;
        }
    }
    MaterializedCliqueRow row;
    row.exits = materialize_clique_exit_candidates(
        lookup_subgraph_clique_row(entry_border, explicit_rows[sg_id]),
        entry_distance,
        max_distance
    );
    return row;
}

struct DirectEmitStats {
    bool used_factorized{false};
    std::size_t hubs_used{0};
    std::size_t exits_emitted{0};
};

template <typename EmitFn>
DirectEmitStats for_each_factorized_subgraph_exit_candidate(
    VertexId entry_border,
    EdgeWeight entry_distance,
    SgId sg_id,
    const std::vector<FactorizedTransferSubgraphModel>& factorized_models,
    const std::vector<std::uint8_t>& factorized_enabled,
    EdgeWeight max_distance,
    EmitFn&& emit
) {
    DirectEmitStats stats;
    if (entry_distance == kInfWeight ||
        sg_id >= factorized_enabled.size() ||
        factorized_enabled[sg_id] == 0U) {
        return stats;
    }

    const auto& model = factorized_models[sg_id];
    const auto it = model.entry_rows.find(entry_border);
    if (it == model.entry_rows.end()) {
        return stats;
    }

    stats.used_factorized = true;
    stats.hubs_used = it->second.size();
    for (const auto& hub_row : it->second) {
        if (hub_row.entry_to_hub_distance == kInfWeight ||
            entry_distance > kInfWeight - hub_row.entry_to_hub_distance) {
            continue;
        }
        const auto base = static_cast<EdgeWeight>(entry_distance + hub_row.entry_to_hub_distance);
        if (max_distance != kInfWeight && base > max_distance) {
            break;
        }
        for (const auto& [exit_border, hub_to_exit] : hub_row.exits) {
            if (hub_to_exit == kInfWeight || base > kInfWeight - hub_to_exit) {
                continue;
            }
            const auto candidate_distance = static_cast<EdgeWeight>(base + hub_to_exit);
            if (max_distance != kInfWeight && candidate_distance > max_distance) {
                break;
            }
            emit(exit_border, candidate_distance);
            ++stats.exits_emitted;
        }
    }
    return stats;
}

std::vector<std::size_t> build_parent_assignment(
    const std::vector<std::vector<SgId>>& adjacency,
    std::size_t max_children
) {
    const std::size_t n = adjacency.size();
    std::vector<std::size_t> parent_of(n, std::numeric_limits<std::size_t>::max());
    if (max_children == 0 || n == 0) {
        return parent_of;
    }

    std::size_t next_parent = 0;
    std::vector<SgId> queue;
    queue.reserve(max_children * 2U + 8U);

    for (SgId seed = 0; seed < n; ++seed) {
        if (parent_of[seed] != std::numeric_limits<std::size_t>::max()) {
            continue;
        }
        parent_of[seed] = next_parent;
        queue.clear();
        queue.push_back(seed);
        std::size_t head = 0;
        std::size_t assigned = 1;

        while (head < queue.size() && assigned < max_children) {
            const auto current = queue[head++];
            for (const auto neighbor : adjacency[current]) {
                if (parent_of[neighbor] != std::numeric_limits<std::size_t>::max()) {
                    continue;
                }
                parent_of[neighbor] = next_parent;
                queue.push_back(neighbor);
                ++assigned;
                if (assigned >= max_children) {
                    break;
                }
            }
        }

        ++next_parent;
    }

    return parent_of;
}

DijkstraTrace dijkstra_trace(
    const Graph& graph,
    const std::vector<std::pair<VertexId, EdgeWeight>>& seeds,
    EdgeWeight max_radius,
    const std::function<void(VertexId, EdgeWeight, std::size_t, const DistanceMap&)>& on_settle = {}
) {
    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    DijkstraTrace trace;

    for (const auto& [seed, value] : seeds) {
        auto it = trace.dist.find(seed);
        if (it == trace.dist.end() || value < it->second) {
            trace.dist[seed] = value;
            pq.push({value, seed});
        }
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();

        const auto best_it = trace.dist.find(u);
        if (best_it == trace.dist.end() || current != best_it->second) {
            continue;
        }
        if (current > max_radius) {
            continue;
        }

        trace.settled.push_back({u, current});
        if (on_settle) {
            on_settle(u, current, trace.settled.size(), trace.dist);
        }
        for (const auto& [v, w] : graph.neighbors(u)) {
            ++trace.relax_attempts;
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next > max_radius) {
                continue;
            }
            auto it = trace.dist.find(v);
            if (it == trace.dist.end() || next < it->second) {
                trace.dist[v] = next;
                ++trace.successful_relaxes;
                ++trace.pq_pushes;
                pq.push({next, v});
            }
        }
    }

    return trace;
}

DijkstraTrace dijkstra_trace_sorted_rows(
    const std::vector<std::int32_t>& sorted_row_index,
    const std::vector<std::vector<std::pair<VertexId, EdgeWeight>>>& sorted_rows,
    const std::vector<std::pair<VertexId, EdgeWeight>>& seeds,
    EdgeWeight max_radius,
    bool enable_row_truncation,
    const std::function<void(VertexId, EdgeWeight, std::size_t, const DistanceMap&)>& on_settle = {}
) {
    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    DijkstraTrace trace;

    for (const auto& [seed, value] : seeds) {
        auto it = trace.dist.find(seed);
        if (it == trace.dist.end() || value < it->second) {
            trace.dist[seed] = value;
            pq.push({value, seed});
        }
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();

        const auto best_it = trace.dist.find(u);
        if (best_it == trace.dist.end() || current != best_it->second) {
            continue;
        }
        if (current > max_radius) {
            continue;
        }

        trace.settled.push_back({u, current});
        if (on_settle) {
            on_settle(u, current, trace.settled.size(), trace.dist);
        }

        const auto& row = lookup_sorted_skeleton_row(u, sorted_row_index, sorted_rows);
        for (std::size_t i = 0; i < row.size(); ++i) {
            const auto [v, w] = row[i];
            ++trace.relax_attempts;
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next > max_radius) {
                if (enable_row_truncation) {
                    ++trace.num_rows_truncated;
                    trace.num_exits_skipped_by_truncation += row.size() - i;
                    break;
                }
                continue;
            }
            auto it = trace.dist.find(v);
            if (it == trace.dist.end() || next < it->second) {
                trace.dist[v] = next;
                ++trace.successful_relaxes;
                ++trace.pq_pushes;
                pq.push({next, v});
            }
        }
    }

    return trace;
}

DijkstraTrace dijkstra_trace_subgraph_rows(
    const std::vector<std::vector<SgId>>& inverted_index_fast,
    const std::vector<std::unordered_map<VertexId, std::vector<std::pair<VertexId, EdgeWeight>>>>& explicit_rows,
    const std::vector<FactorizedTransferSubgraphModel>& factorized_models,
    const std::vector<std::uint8_t>& factorized_enabled,
    const std::vector<std::pair<VertexId, EdgeWeight>>& seeds,
    EdgeWeight max_radius,
    bool use_factorized_transfer,
    std::size_t* factorized_rows_used = nullptr,
    std::size_t* factorized_hubs_used = nullptr,
    std::size_t* factorized_exits_emitted = nullptr,
    const std::function<void(VertexId, EdgeWeight, std::size_t, const DistanceMap&)>& on_settle = {}
) {
    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    DijkstraTrace trace;

    for (const auto& [seed, value] : seeds) {
        auto it = trace.dist.find(seed);
        if (it == trace.dist.end() || value < it->second) {
            trace.dist[seed] = value;
            pq.push({value, seed});
        }
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();

        const auto best_it = trace.dist.find(u);
        if (best_it == trace.dist.end() || current != best_it->second) {
            continue;
        }
        if (current > max_radius) {
            continue;
        }

        trace.settled.push_back({u, current});
        if (on_settle) {
            on_settle(u, current, trace.settled.size(), trace.dist);
        }

        if (u >= inverted_index_fast.size()) {
            continue;
        }
        for (const auto sg_id : inverted_index_fast[u]) {
            const auto row = materialize_subgraph_exit_candidates(
                u,
                current,
                sg_id,
                explicit_rows,
                factorized_models,
                factorized_enabled,
                use_factorized_transfer
            );
            if (row.used_factorized) {
                if (factorized_rows_used != nullptr) {
                    ++(*factorized_rows_used);
                }
                if (factorized_hubs_used != nullptr) {
                    *factorized_hubs_used += row.hubs_used;
                }
                if (factorized_exits_emitted != nullptr) {
                    *factorized_exits_emitted += row.exits.size();
                }
            }
            for (const auto& exit : row.exits) {
                ++trace.relax_attempts;
                const auto next = exit.candidate_distance;
                if (next > max_radius) {
                    continue;
                }
                auto [it, inserted] = trace.dist.emplace(exit.border, next);
                if (inserted || next < it->second) {
                    if (!inserted) {
                        it->second = next;
                    }
                    pq.push({next, exit.border});
                    ++trace.successful_relaxes;
                    ++trace.pq_pushes;
                }
            }
        }
    }

    return trace;
}

bool is_fully_covered(
    const Subgraph& sg,
    const DistanceMap& boundary_distances,
    EdgeWeight radius,
    FcRule rule
) {
    if (rule == FcRule::UpperBoundCandidate) {
        double ub = std::numeric_limits<double>::infinity();
        for (const auto& [b, rb] : sg.rb_map) {
            const auto it = boundary_distances.find(b);
            if (it == boundary_distances.end()) {
                continue;
            }
            ub = std::min(ub, static_cast<double>(it->second) + rb.to_double());
        }
        return ub <= static_cast<double>(radius);
    }

    bool all_covered = true;
    bool paper_guard = false;
    for (const auto b : sg.bound_vertices) {
        const auto it = boundary_distances.find(b);
        if (it == boundary_distances.end() || it->second > radius) {
            all_covered = false;
            break;
        }
        const auto rb_it = sg.rb_map.find(b);
        if (rb_it != sg.rb_map.end()) {
            const auto total = rb_it->second.to_double() + static_cast<double>(it->second);
            if (total <= static_cast<double>(radius)) {
                paper_guard = true;
            }
        }
    }
    if (!all_covered) {
        return false;
    }
    if (rule == FcRule::AllBordersVisited) {
        return true;
    }
    return paper_guard;
}

bool is_fully_covered(
    const Subgraph& sg,
    const std::vector<EdgeWeight>& boundary_distances,
    const std::vector<std::uint32_t>& boundary_stamps,
    std::uint32_t boundary_epoch,
    EdgeWeight radius,
    FcRule rule
) {
    const auto boundary_distance = [&](VertexId v) -> EdgeWeight {
        return (v < boundary_stamps.size() && boundary_stamps[v] == boundary_epoch)
            ? boundary_distances[v]
            : kInfWeight;
    };
    if (rule == FcRule::UpperBoundCandidate) {
        double ub = std::numeric_limits<double>::infinity();
        for (const auto& [b, rb] : sg.rb_map) {
            const auto distance = boundary_distance(b);
            if (distance == kInfWeight) {
                continue;
            }
            ub = std::min(ub, static_cast<double>(distance) + rb.to_double());
        }
        return ub <= static_cast<double>(radius);
    }

    bool all_covered = true;
    bool paper_guard = false;
    for (const auto b : sg.bound_vertices) {
        const auto distance = boundary_distance(b);
        if (distance == kInfWeight || distance > radius) {
            all_covered = false;
            break;
        }
        const auto rb_it = sg.rb_map.find(b);
        if (rb_it != sg.rb_map.end()) {
            const auto total = rb_it->second.to_double() + static_cast<double>(distance);
            if (total <= static_cast<double>(radius)) {
                paper_guard = true;
            }
        }
    }
    if (!all_covered) {
        return false;
    }
    if (rule == FcRule::AllBordersVisited) {
        return true;
    }
    return paper_guard;
}

EdgeWeight object_distance_from_seed(
    const Subgraph& sg,
    const DistanceMap& seed_distances,
    const MovingObject& object
) {
    const auto left_it = seed_distances.find(object.edge.first);
    const auto right_it = seed_distances.find(object.edge.second);
    EdgeWeight best = kInfWeight;
    if (left_it != seed_distances.end() && left_it->second <= kInfWeight - object.offset) {
        best = std::min(best, static_cast<EdgeWeight>(left_it->second + object.offset));
    }
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    if (right_it != seed_distances.end() && right_it->second <= kInfWeight - right_cost) {
        best = std::min(best, static_cast<EdgeWeight>(right_it->second + right_cost));
    }
    return best;
}

EdgeWeight object_distance_from_seed(
    const LocalDijkstraResult& seed_distances,
    const MovingObject& object
) {
    EdgeWeight best = kInfWeight;
    const auto left = lookup_local_distance(seed_distances, object.edge.first);
    if (left != kInfWeight && left <= kInfWeight - object.offset) {
        best = std::min(best, static_cast<EdgeWeight>(left + object.offset));
    }
    const auto right = lookup_local_distance(seed_distances, object.edge.second);
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    if (right != kInfWeight && right <= kInfWeight - right_cost) {
        best = std::min(best, static_cast<EdgeWeight>(right + right_cost));
    }
    return best;
}

EdgeWeight object_distance_from_source_table(
    const Subgraph& sg,
    VertexId source,
    const MovingObject& object
) {
    EdgeWeight best = kInfWeight;
    const auto left = local_source_vertex_distance_from_table(sg, source, object.edge.first);
    best = std::min(best, add_or_inf(left, object.offset));

    const auto right = local_source_vertex_distance_from_table(sg, source, object.edge.second);
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    best = std::min(best, add_or_inf(right, right_cost));
    return best;
}

EdgeWeight object_distance_from_query_edge_table(
    const Subgraph& sg,
    const Edge& query_edge,
    EdgeWeight query_offset,
    EdgeWeight query_edge_weight,
    const MovingObject& object
) {
    EdgeWeight best = kInfWeight;
    const auto left = local_query_vertex_distance_from_table(
        sg,
        query_edge,
        query_offset,
        query_edge_weight,
        object.edge.first
    );
    best = std::min(best, add_or_inf(left, object.offset));

    const auto right = local_query_vertex_distance_from_table(
        sg,
        query_edge,
        query_offset,
        query_edge_weight,
        object.edge.second
    );
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    best = std::min(best, add_or_inf(right, right_cost));
    return best;
}

EdgeWeight object_distance_from_boundary_costs(
    const std::vector<std::pair<VertexId, EdgeWeight>>& border_costs,
    VertexId boundary,
    EdgeWeight boundary_distance
) {
    for (const auto& [candidate_boundary, border_cost] : border_costs) {
        if (candidate_boundary != boundary) {
            continue;
        }
        if (boundary_distance <= kInfWeight - border_cost) {
            return static_cast<EdgeWeight>(boundary_distance + border_cost);
        }
        break;
    }
    return kInfWeight;
}

EdgeWeight object_lower_bound_from_subgraph(
    const Subgraph& sg,
    EdgeWeight subgraph_lb,
    const MovingObject& object
) {
    if (subgraph_lb == kInfWeight) {
        return kInfWeight;
    }

    const auto lookup_md = [&](VertexId v) -> EdgeWeight {
        if (sg.bound_vertices.contains(v)) {
            return 0;
        }
        const auto it = sg.internal_to_nearest_border_dist.find(v);
        return (it == sg.internal_to_nearest_border_dist.end()) ? kInfWeight : it->second;
    };

    const auto md_left = lookup_md(object.edge.first);
    const auto md_right = lookup_md(object.edge.second);
    EdgeWeight suffix = kInfWeight;
    if (md_left != kInfWeight && md_left <= kInfWeight - object.offset) {
        suffix = std::min(suffix, static_cast<EdgeWeight>(md_left + object.offset));
    }
    const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
    if (md_right != kInfWeight && md_right <= kInfWeight - right_cost) {
        suffix = std::min(suffix, static_cast<EdgeWeight>(md_right + right_cost));
    }
    if (suffix == kInfWeight || subgraph_lb > kInfWeight - suffix) {
        return kInfWeight;
    }
    return static_cast<EdgeWeight>(subgraph_lb + suffix);
}

EdgeWeight subgraph_lower_bound(const Subgraph& sg, const DistanceMap& skeleton_distances) {
    EdgeWeight best = kInfWeight;
    for (const auto b : sg.bound_vertices) {
        const auto it = skeleton_distances.find(b);
        if (it != skeleton_distances.end()) {
            best = std::min(best, it->second);
        }
    }
    return best;
}

EdgeWeight subgraph_lower_bound(const Subgraph& sg, const std::vector<EdgeWeight>& skeleton_distances) {
    EdgeWeight best = kInfWeight;
    for (const auto b : sg.bound_vertices) {
        if (b < skeleton_distances.size()) {
            best = std::min(best, skeleton_distances[b]);
        }
    }
    return best;
}

std::optional<VertexId> query_vertex_if_exact_vertex(const QueryPoint& query, const Graph& graph) {
    const auto edge = ordered_edge(query.edge.first, query.edge.second);
    const auto weight = edge_weight_any_or_null(graph, edge);
    if (!weight.has_value()) {
        return std::nullopt;
    }
    if (query.offset == 0) {
        return edge.first;
    }
    if (query.offset == *weight) {
        return edge.second;
    }
    return std::nullopt;
}

template <typename Heap>
void push_bounded_candidate(Heap& heap, std::size_t limit, const typename Heap::value_type& value) {
    if (limit == 0) {
        return;
    }
    if (heap.size() < limit) {
        heap.push(value);
        return;
    }
    if (value < heap.top()) {
        heap.pop();
        heap.push(value);
    }
}

bool fully_visited_edge(EdgeWeight du, EdgeWeight dv, EdgeWeight weight, EdgeWeight radius) {
    if (du <= radius && dv <= radius) {
        return true;
    }
    if (du == kInfWeight || dv == kInfWeight) {
        return false;
    }
    const auto lhs =
        static_cast<std::uint64_t>(du) + static_cast<std::uint64_t>(dv) +
        static_cast<std::uint64_t>(weight);
    const auto rhs = 2ULL * static_cast<std::uint64_t>(radius);
    return lhs <= rhs;
}

template <typename DistanceLookup, typename MarkSeen>
void collect_partial_subgraph_objects(
    const std::vector<PopulatedEdgeObjects>& populated_edges,
    DistanceLookup&& lookup_distance,
    EdgeWeight radius,
    const IndexedMovingObjectSet& objects,
    MarkSeen&& mark_seen,
    RangeQueryResult& result
) {
    for (const auto& populated : populated_edges) {
        const auto& edge = populated.edge;
        const auto weight = populated.edge_weight;
        const auto du = lookup_distance(edge.first);
        const auto dv = lookup_distance(edge.second);
        if (du == kInfWeight && dv == kInfWeight) {
            continue;
        }

        if (fully_visited_edge(du, dv, weight, radius)) {
            for (const auto obj_id : populated.object_ids) {
                if (mark_seen(obj_id)) {
                    result.object_ids.push_back(obj_id);
                    ++result.auto_included_objects;
                }
            }
            continue;
        }

        for (const auto obj_id : populated.object_ids) {
            const auto& object = objects[obj_id];
            ++result.exact_checked_objects;

            EdgeWeight best = kInfWeight;
            if (du != kInfWeight && du <= kInfWeight - object.offset) {
                best = std::min(best, static_cast<EdgeWeight>(du + object.offset));
            }
            const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
            if (dv != kInfWeight && dv <= kInfWeight - right_cost) {
                best = std::min(best, static_cast<EdgeWeight>(dv + right_cost));
            }

            if (best <= radius && mark_seen(obj_id)) {
                result.object_ids.push_back(object.unique_id);
            }
        }
    }
}

template <typename DistanceProvider>
KnnQueryResult compute_knn(
    const IndexedMovingObjectSet& objects,
    std::size_t k,
    DistanceProvider provider
) {
    std::vector<KnnItem> items;
    items.reserve(objects.size());
    for (const auto& object : objects.objects()) {
        const auto distance = provider(object);
        if (distance == kInfWeight) {
            continue;
        }
        items.push_back({object.unique_id, distance});
    }

    std::sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.distance != rhs.distance) {
            return lhs.distance < rhs.distance;
        }
        return lhs.id < rhs.id;
    });
    if (items.size() > k) {
        items.resize(k);
    }
    return KnnQueryResult{std::move(items)};
}

}  // namespace

std::string SkeletonIndex::make_cache_key(
    const std::string& partition_key,
    bool factorized_transfer,
    double max_arc_ratio,
    std::size_t min_border_count
) {
    auto hash = fnv1a64_seed();
    std::ostringstream oss;
    oss << "bag-index-cache-v1"
        << "|partition=" << partition_key
        << "|factorized_transfer=" << (factorized_transfer ? 1 : 0)
        << "|factorized_arc_threshold=" << max_arc_ratio
        << "|factorized_border_threshold=" << min_border_count;
    hash = fnv1a64_append(hash, oss.str());
    return hex_u64(hash);
}

void SkeletonIndex::initialize_runtime_scratch() {
    knn_best_subgraph_lb_scratch_.assign(subgraphs_.size(), kInfWeight);
    knn_best_subgraph_ud_scratch_.assign(subgraphs_.size(), kInfWeight);
    knn_best_subgraph_lb_stamp_.assign(subgraphs_.size(), 0U);
    knn_best_subgraph_ud_stamp_.assign(subgraphs_.size(), 0U);
    knn_visited_subgraph_stamp_.assign(subgraphs_.size(), 0U);
    knn_admitted_subgraph_stamp_.assign(subgraphs_.size(), 0U);
    knn_pruned_subgraph_stamp_.assign(subgraphs_.size(), 0U);
    knn_tightened_subgraph_stamp_.assign(subgraphs_.size(), 0U);
    knn_sg_touch_count_scratch_.assign(subgraphs_.size(), 0U);
    knn_sg_touch_stamp_.assign(subgraphs_.size(), 0U);
    skeleton_scratch_dist_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, kInfWeight);
    skeleton_scratch_stamp_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
    range_subgraph_touch_count_scratch_.assign(subgraphs_.size(), 0U);
    range_subgraph_touch_stamp_scratch_.assign(subgraphs_.size(), 0U);
}

void SkeletonIndex::rebuild_cached_runtime_views() {
    if (subgraph_boundaries_.size() != subgraphs_.size() ||
        subgraph_clique_row_storage_.size() != subgraphs_.size() ||
        subgraph_local_indices_.size() != subgraphs_.size() ||
        subgraph_adjacency_.size() != subgraphs_.size()) {
        throw std::runtime_error("index cache payload does not match subgraph count");
    }
    subgraph_clique_rows_.assign(subgraphs_.size(), {});
    inverted_index_fast_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, {});
    inverted_boundary_index_fast_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, {});
    for (const auto& sg : subgraphs_) {
        if (sg.id >= subgraph_boundaries_.size() || sg.id >= subgraph_clique_row_storage_.size()) {
            throw std::runtime_error("index cache subgraph id out of range");
        }
        const auto& boundaries = subgraph_boundaries_[sg.id];
        const auto& storage = subgraph_clique_row_storage_[sg.id];
        if (boundaries.size() != storage.size()) {
            throw std::runtime_error("index cache clique row storage mismatch");
        }
        auto& clique_rows = subgraph_clique_rows_[sg.id];
        clique_rows.reserve(boundaries.size());
        for (std::size_t row_id = 0; row_id < boundaries.size(); ++row_id) {
            const auto boundary = boundaries[row_id];
            clique_rows.emplace(boundary, storage[row_id]);
            if (boundary < inverted_index_fast_.size()) {
                inverted_index_fast_[boundary].push_back(sg.id);
                inverted_boundary_index_fast_[boundary].push_back(
                    InvertedSubgraphBoundaryEntry{sg.id, static_cast<std::uint32_t>(row_id)}
                );
            }
        }
    }
}

SkeletonIndex SkeletonIndex::build(const Graph& global, std::vector<Subgraph> subgraphs) {
    SkeletonIndex index;
    index.global_ = &global;
    index.subgraphs_ = std::move(subgraphs);
    std::unordered_map<VertexId, std::vector<SgId>> inverted_map;
    for (const auto& sg : index.subgraphs_) {
        for (const auto v : sg.graph.vertices()) {
            index.max_vertex_id_ = std::max(index.max_vertex_id_, v);
        }
    }
    index.vertex_to_subgraph_fast_.assign(
        static_cast<std::size_t>(index.max_vertex_id_) + 1U,
        std::numeric_limits<SgId>::max()
    );
    index.inverted_boundary_index_fast_.resize(static_cast<std::size_t>(index.max_vertex_id_) + 1U);
    index.subgraph_boundaries_.resize(index.subgraphs_.size());
    index.subgraph_clique_rows_.resize(index.subgraphs_.size());
    index.subgraph_clique_row_storage_.resize(index.subgraphs_.size());
    index.subgraph_local_indices_.resize(index.subgraphs_.size());

    for (auto& sg : index.subgraphs_) {
        auto& boundaries = index.subgraph_boundaries_[sg.id];
        boundaries.assign(sg.bound_vertices.begin(), sg.bound_vertices.end());
        std::sort(boundaries.begin(), boundaries.end());
        auto& local_index = index.subgraph_local_indices_[sg.id];
        const auto vertices = sg.graph.vertices();
        local_index.local_ids.reserve(vertices.size() * 2U + 1U);
        local_index.adjacency.assign(vertices.size(), {});
        for (std::size_t local_id = 0; local_id < vertices.size(); ++local_id) {
            local_index.local_ids.emplace(vertices[local_id], static_cast<std::uint32_t>(local_id));
        }
        for (std::size_t local_id = 0; local_id < vertices.size(); ++local_id) {
            const auto vertex = vertices[local_id];
            const auto& neighbors = sg.graph.neighbors(vertex);
            auto& out = local_index.adjacency[local_id];
            out.reserve(neighbors.size());
            for (const auto& [neighbor, weight] : neighbors) {
                const auto neighbor_it = local_index.local_ids.find(neighbor);
                if (neighbor_it != local_index.local_ids.end()) {
                    out.push_back({neighbor_it->second, weight});
                }
            }
        }
        auto& clique_rows = index.subgraph_clique_rows_[sg.id];
        auto& clique_row_storage = index.subgraph_clique_row_storage_[sg.id];
        clique_rows.reserve(boundaries.size());
        clique_row_storage.reserve(boundaries.size());
        const bool have_precomputed_local_dist = !sg.distance.empty();
        for (const auto b : boundaries) {
            auto row = have_precomputed_local_dist
                ? [&] {
                    std::vector<std::pair<VertexId, EdgeWeight>> precomputed;
                    precomputed.reserve(boundaries.size());
                    for (const auto other : boundaries) {
                        if (b == other) {
                            continue;
                        }
                        const auto distance = sg.distance.get_or_inf(b, other);
                        if (distance != kInfWeight) {
                            precomputed.push_back({other, distance});
                        }
                    }
                    std::sort(precomputed.begin(), precomputed.end(), [](const auto& lhs, const auto& rhs) {
                        if (lhs.second != rhs.second) {
                            return lhs.second < rhs.second;
                        }
                        return lhs.first < rhs.first;
                    });
                    return precomputed;
                }()
                : boundary_row_from_local_index(local_index, b, boundaries);
            clique_rows.emplace(b, row);
            clique_row_storage.push_back(std::move(row));
        }
        for (const auto v : sg.internal_vertices) {
            index.vertex_to_subgraph_fast_[v] = sg.id;
        }
        for (const auto& [edge, weight] : sg.graph.undirected_edges()) {
            (void)weight;
            index.edge_to_subgraph_[ordered_edge(edge.first, edge.second)] = sg.id;
        }

        std::uint32_t clique_row_id = 0U;
        for (const auto b : boundaries) {
            index.skeleton_.insert(b);
            inverted_map[b].push_back(sg.id);
            if (b < index.inverted_boundary_index_fast_.size()) {
                index.inverted_boundary_index_fast_[b].push_back(InvertedSubgraphBoundaryEntry{sg.id, clique_row_id});
            }
            for (const auto& [other, distance] : clique_row_storage[clique_row_id]) {
                if (distance != kInfWeight) {
                    index.skeleton_.set_min_undirected_edge(b, other, distance);
                }
            }
            ++clique_row_id;
        }
    }

    index.inverted_index_fast_.resize(static_cast<std::size_t>(index.max_vertex_id_) + 1U);
    for (const auto& [vertex, ids] : inverted_map) {
        index.inverted_index_fast_[vertex] = ids;
    }

    std::vector<std::unordered_set<SgId>> adjacency_sets(index.subgraphs_.size());
    for (const auto& [border, ids] : inverted_map) {
        (void)border;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            for (std::size_t j = i + 1; j < ids.size(); ++j) {
                adjacency_sets[ids[i]].insert(ids[j]);
                adjacency_sets[ids[j]].insert(ids[i]);
            }
        }
    }
    index.subgraph_adjacency_.resize(index.subgraphs_.size());
    for (std::size_t sg_id = 0; sg_id < adjacency_sets.size(); ++sg_id) {
        auto& out = index.subgraph_adjacency_[sg_id];
        out.assign(adjacency_sets[sg_id].begin(), adjacency_sets[sg_id].end());
        std::sort(out.begin(), out.end());
    }

    index.sorted_skeleton_row_index_.assign(
        static_cast<std::size_t>(index.max_vertex_id_) + 1U,
        -1
    );
    index.sorted_skeleton_row_storage_.reserve(index.skeleton_.size());
    for (const auto border : index.skeleton_.vertices()) {
        std::vector<std::pair<VertexId, EdgeWeight>> row;
        const auto& neighbors = index.skeleton_.neighbors(border);
        row.reserve(neighbors.size());
        for (const auto& [other, weight] : neighbors) {
            row.push_back({other, weight});
        }
        std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.second != rhs.second) {
                return lhs.second < rhs.second;
            }
            return lhs.first < rhs.first;
        });
        index.sorted_skeleton_row_index_[border] =
            static_cast<std::int32_t>(index.sorted_skeleton_row_storage_.size());
        index.sorted_skeleton_row_storage_.push_back(std::move(row));
    }

    index.initialize_runtime_scratch();
    return index;
}

bool SkeletonIndex::load_cache(
    const std::filesystem::path& path,
    const std::string& expected_key,
    const Graph& global,
    const std::vector<Subgraph>& subgraphs,
    SkeletonIndex& out
) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    const auto magic = read_pod<std::uint64_t>(in);
    const auto version = read_pod<std::uint32_t>(in);
    if (magic != kIndexCacheMagic || version != kIndexCacheVersion) {
        throw std::runtime_error("index cache header mismatch: " + path.string());
    }
    const auto key = read_string(in);
    if (key != expected_key) {
        throw std::runtime_error("index cache key mismatch: " + path.string());
    }
    const auto subgraph_count = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    if (subgraph_count != subgraphs.size()) {
        throw std::runtime_error("index cache subgraph count mismatch: " + path.string());
    }
    SkeletonIndex index;
    index.global_ = &global;
    index.subgraphs_ = subgraphs;
    index.max_vertex_id_ = read_pod<VertexId>(in);
    index.skeleton_ = read_graph(in);
    index.vertex_to_subgraph_fast_ = read_pod_vector<SgId>(in);
    index.edge_to_subgraph_ = read_edge_to_subgraph_map(in);
    index.sorted_skeleton_row_index_ = read_pod_vector<std::int32_t>(in);
    index.sorted_skeleton_row_storage_ = read_nested_vertex_weight_rows(in);
    index.subgraph_boundaries_ = read_vector_of_vertex_lists(in);
    const auto clique_storage_outer = read_pod<std::uint64_t>(in);
    index.subgraph_clique_row_storage_.reserve(static_cast<std::size_t>(clique_storage_outer));
    for (std::uint64_t i = 0; i < clique_storage_outer; ++i) {
        index.subgraph_clique_row_storage_.push_back(read_nested_vertex_weight_rows(in));
    }
    index.subgraph_local_indices_ = read_local_subgraph_indices(in);
    index.subgraph_adjacency_ = read_vector_of_sgid_lists(in);
    index.factorized_transfer_enabled_ = read_pod_vector<std::uint8_t>(in);
    index.factorized_transfer_models_ = read_factorized_transfer_models(in);
    index.rebuild_cached_runtime_views();
    index.initialize_runtime_scratch();
    out = std::move(index);
    return true;
}

void SkeletonIndex::save_cache(
    const std::filesystem::path& path,
    const std::string& key
) const {
    write_atomic_index_blob(path, [&](std::ostream& out) {
        write_pod<std::uint64_t>(out, kIndexCacheMagic);
        write_pod<std::uint32_t>(out, kIndexCacheVersion);
        write_string(out, key);
        write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(subgraphs_.size()));
        write_pod<VertexId>(out, max_vertex_id_);
        write_graph(out, skeleton_);
        write_pod_vector(out, vertex_to_subgraph_fast_);
        write_edge_to_subgraph_map(out, edge_to_subgraph_);
        write_pod_vector(out, sorted_skeleton_row_index_);
        write_nested_vertex_weight_rows(out, sorted_skeleton_row_storage_);
        write_vector_of_vertex_lists(out, subgraph_boundaries_);
        write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(subgraph_clique_row_storage_.size()));
        for (const auto& subgraph_rows : subgraph_clique_row_storage_) {
            write_nested_vertex_weight_rows(out, subgraph_rows);
        }
        write_local_subgraph_indices(out, subgraph_local_indices_);
        write_vector_of_sgid_lists(out, subgraph_adjacency_);
        write_pod_vector(out, factorized_transfer_enabled_);
        write_factorized_transfer_models(out, factorized_transfer_models_);
    });
}

void SkeletonIndex::configure_factorized_transfer(
    double max_arc_ratio,
    std::size_t min_border_count
) {
    factorized_transfer_models_.assign(subgraphs_.size(), FactorizedTransferSubgraphModel{});
    factorized_transfer_enabled_.assign(subgraphs_.size(), 0U);
    for (const auto& sg : subgraphs_) {
        if (sg.id >= factorized_transfer_models_.size()) {
            continue;
        }
        if (sg.bound_vertices.size() < min_border_count) {
            continue;
        }
        auto model = build_factorized_transfer_model(sg);
        if (!model.feasible || model.factorized_arc_ratio > max_arc_ratio) {
            factorized_transfer_models_[sg.id] = std::move(model);
            continue;
        }
        factorized_transfer_enabled_[sg.id] = 1U;
        factorized_transfer_models_[sg.id] = std::move(model);
    }
}

void SkeletonIndex::release_subgraph_distances() {
    for (auto& sg : subgraphs_) {
        sg.distance = DistanceTable{};
    }
}

const Graph& SkeletonIndex::skeleton() const {
    return skeleton_;
}

const std::vector<Subgraph>& SkeletonIndex::subgraphs() const {
    return subgraphs_;
}

const std::unordered_map<Edge, SgId, PairHash>& SkeletonIndex::edge_to_subgraph() const {
    return edge_to_subgraph_;
}

const std::vector<std::unordered_map<VertexId, std::vector<std::pair<VertexId, EdgeWeight>>>>&
SkeletonIndex::subgraph_clique_rows() const {
    return subgraph_clique_rows_;
}

const Subgraph& SkeletonIndex::require_initial_subgraph(const Edge& edge) const {
    const auto normalized = ordered_edge(edge.first, edge.second);
    const auto it = edge_to_subgraph_.find(normalized);
    if (it == edge_to_subgraph_.end()) {
        throw std::runtime_error("query edge not found in any subgraph");
    }
    return subgraphs_.at(it->second);
}

RangeQueryResult SkeletonIndex::range_query(
    const QueryPoint& query,
    EdgeWeight radius,
    const IndexedMovingObjectSet& objects,
    FcRule rule,
    std::size_t query_id,
    std::vector<BorderExposureEvent>* exposure_events,
    std::vector<CliqueRowShadowRecord>* row_shadow_records,
    bool enable_row_truncation,
    bool use_factorized_transfer,
    bool allow_whole_subgraph_acceptance
) const {
    RangeQueryResult result;
    std::unordered_map<std::pair<SgId, VertexId>, std::size_t, PairHash> exposure_index;
    struct PendingRowShadow {
        std::size_t index{0};
        std::vector<std::pair<VertexId, EdgeWeight>> pushed_exits;
    };
    const bool need_exposure_state = exposure_events != nullptr;
    const bool need_row_shadow = row_shadow_records != nullptr;
    const bool fast_dense_path = !need_exposure_state && !need_row_shadow && !use_factorized_transfer;
    std::vector<std::vector<VertexId>> sg_seed_borders =
        need_exposure_state ? std::vector<std::vector<VertexId>>(subgraphs_.size()) : std::vector<std::vector<VertexId>>{};
    std::vector<std::uint8_t> sg_fully_covered =
        need_exposure_state ? std::vector<std::uint8_t>(subgraphs_.size(), 0U) : std::vector<std::uint8_t>{};
    std::vector<std::size_t> sg_touch_events =
        need_exposure_state ? std::vector<std::size_t>(subgraphs_.size(), 0U) : std::vector<std::size_t>{};
    std::vector<std::uint8_t> shadow_seen_subgraphs =
        need_row_shadow ? std::vector<std::uint8_t>(subgraphs_.size(), 0U) : std::vector<std::uint8_t>{};
    std::vector<PendingRowShadow> pending_rows;
    std::size_t factorized_rows_used = 0;
    std::size_t factorized_hubs_used = 0;
    std::size_t factorized_exits_emitted = 0;
    if (range_object_seen_stamp_.size() != objects.size()) {
        range_object_seen_stamp_.assign(objects.size(), 0U);
        range_object_seen_epoch_ = 0U;
    }
    if (range_object_seen_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(range_object_seen_stamp_.begin(), range_object_seen_stamp_.end(), 0U);
        range_object_seen_epoch_ = 0U;
    }
    ++range_object_seen_epoch_;
    const auto object_seen_epoch = range_object_seen_epoch_;
    const auto mark_seen_object = [&](ObjId obj_id) -> bool {
        const auto index = static_cast<std::size_t>(obj_id);
        if (index >= range_object_seen_stamp_.size()) {
            return false;
        }
        if (range_object_seen_stamp_[index] == object_seen_epoch) {
            return false;
        }
        range_object_seen_stamp_[index] = object_seen_epoch;
        return true;
    };

    const auto& initial = require_initial_subgraph(query.edge);
    const auto& initial_local_index = subgraph_local_indices_.at(initial.id);
    const auto initial_local = local_query_distances_compact(initial, initial_local_index, query, radius);
    const auto initial_seeds = boundary_query_seeds(initial, initial_local);
    DijkstraTrace boundary_trace;
    DistanceMap boundary_distances;
    std::vector<VertexId> fast_touched_boundaries;
    std::vector<SgId> fast_touched_subgraphs;
    std::uint32_t fast_boundary_epoch = 0U;
    std::uint32_t fast_subgraph_epoch = 0U;
    if (use_factorized_transfer && exposure_events == nullptr && row_shadow_records == nullptr) {
        boundary_trace = dijkstra_trace_subgraph_rows(
            inverted_index_fast_,
            subgraph_clique_rows_,
            factorized_transfer_models_,
            factorized_transfer_enabled_,
            initial_seeds,
            radius,
            true,
            &factorized_rows_used,
            &factorized_hubs_used,
            &factorized_exits_emitted
        );
        boundary_distances = boundary_trace.dist;
    } else if (fast_dense_path) {
        using QueueItem = std::pair<EdgeWeight, VertexId>;
        std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
        if (skeleton_scratch_dist_.size() != static_cast<std::size_t>(max_vertex_id_) + 1U) {
            skeleton_scratch_dist_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
            skeleton_scratch_stamp_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
            skeleton_scratch_epoch_ = 0U;
        }
        if (skeleton_scratch_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(skeleton_scratch_stamp_.begin(), skeleton_scratch_stamp_.end(), 0U);
            skeleton_scratch_epoch_ = 0U;
        }
        ++skeleton_scratch_epoch_;
        fast_boundary_epoch = skeleton_scratch_epoch_;
        fast_touched_boundaries.reserve(initial_seeds.size() * 8U + 64U);
        const auto boundary_distance = [&](VertexId v) -> EdgeWeight {
            return (v < skeleton_scratch_stamp_.size() && skeleton_scratch_stamp_[v] == fast_boundary_epoch)
                ? skeleton_scratch_dist_[v]
                : kInfWeight;
        };
        const auto update_boundary = [&](VertexId v, EdgeWeight distance) {
            if (skeleton_scratch_stamp_[v] == fast_boundary_epoch) {
                if (distance >= skeleton_scratch_dist_[v]) {
                    return false;
                }
                skeleton_scratch_dist_[v] = distance;
            } else {
                skeleton_scratch_stamp_[v] = fast_boundary_epoch;
                skeleton_scratch_dist_[v] = distance;
                fast_touched_boundaries.push_back(v);
            }
            pq.push({distance, v});
            return true;
        };
        for (const auto& [seed, value] : initial_seeds) {
            update_boundary(seed, value);
        }
        while (!pq.empty()) {
            const auto [current, u] = pq.top();
            pq.pop();
            if (current != boundary_distance(u)) {
                continue;
            }
            if (current > radius) {
                continue;
            }
            const auto& row = lookup_sorted_skeleton_row(u, sorted_skeleton_row_index_, sorted_skeleton_row_storage_);
            for (std::size_t i = 0; i < row.size(); ++i) {
                const auto [v, w] = row[i];
                ++boundary_trace.relax_attempts;
                if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                    continue;
                }
                const auto next = static_cast<EdgeWeight>(current + w);
                if (next > radius) {
                    if (enable_row_truncation) {
                        ++boundary_trace.num_rows_truncated;
                        boundary_trace.num_exits_skipped_by_truncation += row.size() - i;
                        break;
                    }
                    continue;
                }
                if (update_boundary(v, next)) {
                    ++boundary_trace.successful_relaxes;
                    ++boundary_trace.pq_pushes;
                }
            }
        }
        if (range_subgraph_touch_count_scratch_.size() != subgraphs_.size()) {
            range_subgraph_touch_count_scratch_.assign(subgraphs_.size(), 0U);
            range_subgraph_touch_stamp_scratch_.assign(subgraphs_.size(), 0U);
            range_subgraph_touch_epoch_ = 0U;
        }
        if (range_subgraph_touch_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(range_subgraph_touch_stamp_scratch_.begin(), range_subgraph_touch_stamp_scratch_.end(), 0U);
            range_subgraph_touch_epoch_ = 0U;
        }
        ++range_subgraph_touch_epoch_;
        fast_subgraph_epoch = range_subgraph_touch_epoch_;
        fast_touched_subgraphs.reserve(fast_touched_boundaries.size() * 2U + 8U);
        for (const auto vertex : fast_touched_boundaries) {
            if (vertex >= inverted_index_fast_.size()) {
                continue;
            }
            for (const auto sg_id : inverted_index_fast_[vertex]) {
                if (range_subgraph_touch_stamp_scratch_[sg_id] == fast_subgraph_epoch) {
                    ++range_subgraph_touch_count_scratch_[sg_id];
                    ++result.num_redundant_subgraph_reentries;
                } else {
                    range_subgraph_touch_stamp_scratch_[sg_id] = fast_subgraph_epoch;
                    range_subgraph_touch_count_scratch_[sg_id] = 1U;
                    fast_touched_subgraphs.push_back(sg_id);
                }
            }
        }
    } else if (!use_factorized_transfer) {
        boundary_trace = dijkstra_trace_sorted_rows(
            sorted_skeleton_row_index_,
            sorted_skeleton_row_storage_,
            initial_seeds,
            radius,
            enable_row_truncation,
            [&](VertexId border, EdgeWeight distance, std::size_t visit_order, const DistanceMap& current_dist) {
                if (row_shadow_records == nullptr) {
                    return;
                }
                if (border >= inverted_index_fast_.size()) {
                    return;
                }
                for (const auto sg_id : inverted_index_fast_[border]) {
                    if (shadow_seen_subgraphs[sg_id] != 0U) {
                        continue;
                    }
                    shadow_seen_subgraphs[sg_id] = 1U;
                    std::vector<std::pair<VertexId, EdgeWeight>> exits;
                    exits = lookup_subgraph_clique_row(border, subgraph_clique_rows_[sg_id]);

                    CliqueRowShadowRecord record;
                    record.query_id = query_id;
                    record.query_type = QueryType::Range;
                    record.subgraph_id = sg_id;
                    record.entry_border = border;
                    record.entry_visit_order = visit_order;
                    record.entry_distance = distance;
                    record.threshold_at_entry = radius;
                    record.num_exits_total = exits.size();
                    record.useful_prefix_len = exits.size();

                    PendingRowShadow pending;
                    pending.index = row_shadow_records->size();

                    for (std::size_t i = 0; i < exits.size(); ++i) {
                        const auto [exit_border, local] = exits[i];
                        if (distance == kInfWeight || local == kInfWeight || distance > kInfWeight - local) {
                            ++record.num_exits_threshold_dead;
                            record.useful_prefix_len = std::min(record.useful_prefix_len, i);
                            continue;
                        }
                        const auto cand = static_cast<EdgeWeight>(distance + local);
                        const auto current_it = current_dist.find(exit_border);
                        const auto current_best =
                            (current_it == current_dist.end()) ? kInfWeight : current_it->second;
                        const bool dist_dominated = cand >= current_best;
                        const bool threshold_dead = cand > radius;
                        if (dist_dominated) {
                            ++record.num_exits_dist_dominated;
                        }
                        if (threshold_dead) {
                            ++record.num_exits_threshold_dead;
                            record.useful_prefix_len = std::min(record.useful_prefix_len, i);
                        }
                        if (!dist_dominated) {
                            ++record.num_pq_pushes_from_this_row;
                            if (i >= record.useful_prefix_len) {
                                ++record.num_pq_pushes_beyond_useful_prefix;
                            }
                            pending.pushed_exits.push_back({exit_border, cand});
                        }
                    }

                    row_shadow_records->push_back(record);
                    pending_rows.push_back(std::move(pending));
                }
            }
        );
        boundary_distances = boundary_trace.dist;
    } else {
        boundary_trace = dijkstra_trace_subgraph_rows(
            inverted_index_fast_,
            subgraph_clique_rows_,
            factorized_transfer_models_,
            factorized_transfer_enabled_,
            initial_seeds,
            radius,
            true,
            &factorized_rows_used,
            &factorized_hubs_used,
            &factorized_exits_emitted
        );
        boundary_distances = boundary_trace.dist;
    }
    result.initial_local_vertices = initial_local.reached_count;
    result.boundary_vertices_reached = fast_dense_path ? fast_touched_boundaries.size() : boundary_distances.size();
    result.num_clique_relax_attempts = boundary_trace.relax_attempts;
    result.num_successful_clique_relaxes = boundary_trace.successful_relaxes;
    result.num_pq_pushes_from_clique = boundary_trace.pq_pushes;
    result.num_rows_truncated = boundary_trace.num_rows_truncated;
    result.num_exits_skipped_by_truncation = boundary_trace.num_exits_skipped_by_truncation;
    result.factorized_rows_used = factorized_rows_used;
    result.factorized_hubs_used = factorized_hubs_used;
    result.factorized_exits_emitted = factorized_exits_emitted;

    auto ensure_exposure = [&](SgId sg_id, VertexId border, std::size_t visit_order, EdgeWeight distance) {
        if (exposure_events == nullptr) {
            return;
        }
        const auto key = std::make_pair(sg_id, border);
        if (const auto it = exposure_index.find(key); it != exposure_index.end()) {
            auto& event = exposure_events->at(it->second);
            event.visit_order = std::min(event.visit_order, visit_order);
            event.distance_from_query = std::min(event.distance_from_query, distance);
            return;
        }
        exposure_index[key] = exposure_events->size();
        exposure_events->push_back(BorderExposureEvent{
            query_id,
            QueryType::Range,
            sg_id,
            border,
            visit_order,
            false,
            false,
            false,
            false,
            false,
            distance,
            0,
        });
    };

    if (exposure_events != nullptr) {
        for (std::size_t order = 0; order < boundary_trace.settled.size(); ++order) {
            const auto [border, distance] = boundary_trace.settled[order];
            if (border >= inverted_index_fast_.size()) {
                continue;
            }
            for (const auto sg_id : inverted_index_fast_[border]) {
                ensure_exposure(sg_id, border, order + 1U, distance);
            }
        }
    }

    std::vector<SgId> hit_subgraphs;
    if (fast_dense_path) {
        hit_subgraphs.reserve(fast_touched_subgraphs.size());
        for (const auto sg_id : fast_touched_subgraphs) {
            if (sg_id != initial.id) {
                hit_subgraphs.push_back(sg_id);
            }
        }
    } else {
        std::vector<std::size_t> hit_counts(subgraphs_.size(), 0U);
        hit_subgraphs.reserve(boundary_distances.size());
        for (const auto& [vertex, distance] : boundary_distances) {
            if (distance > radius) {
                continue;
            }
            if (vertex >= inverted_index_fast_.size()) {
                continue;
            }
            for (const auto sg_id : inverted_index_fast_[vertex]) {
                ++sg_touch_events[sg_id];
                if (sg_touch_events[sg_id] > 1U) {
                    ++result.num_redundant_subgraph_reentries;
                }
                if (sg_id != initial.id) {
                    if (hit_counts[sg_id] == 0U) {
                        hit_subgraphs.push_back(sg_id);
                    }
                    ++hit_counts[sg_id];
                }
            }
        }
    }

    const bool initial_fc = fast_dense_path
        ? is_fully_covered(initial, skeleton_scratch_dist_, skeleton_scratch_stamp_, fast_boundary_epoch, radius, rule)
        : is_fully_covered(initial, boundary_distances, radius, rule);
    if (need_exposure_state) {
        sg_fully_covered[initial.id] = initial_fc ? 1U : 0U;
        for (const auto& [b, distance] : initial_seeds) {
            (void)distance;
            sg_seed_borders[initial.id].push_back(b);
        }
    }
    if (initial_fc) {
        ++result.fc_subgraphs;
        if (allow_whole_subgraph_acceptance) {
            for (const auto obj_id : objects.objects_in(initial.id)) {
                if (mark_seen_object(obj_id)) {
                    result.object_ids.push_back(obj_id);
                    ++result.auto_included_objects;
                }
            }
        } else {
            collect_partial_subgraph_objects(
                objects.populated_edges_in(initial.id),
                [&](VertexId v) { return lookup_local_distance(initial_local, v); },
                radius,
                objects,
                mark_seen_object,
                result
            );
        }
    } else {
        ++result.pc_subgraphs;
        collect_partial_subgraph_objects(
            objects.populated_edges_in(initial.id),
            [&](VertexId v) { return lookup_local_distance(initial_local, v); },
            radius,
            objects,
            mark_seen_object,
            result
        );
    }

    for (const auto sg_id : hit_subgraphs) {
        const auto hit_count = fast_dense_path
            ? range_subgraph_touch_count_scratch_[sg_id]
            : 1U;
        if (fast_dense_path && hit_count == 0) {
            continue;
        }
        const auto& sg = subgraphs_.at(sg_id);
        const bool fully_covered = fast_dense_path
            ? is_fully_covered(sg, skeleton_scratch_dist_, skeleton_scratch_stamp_, fast_boundary_epoch, radius, rule)
            : is_fully_covered(sg, boundary_distances, radius, rule);
        if (need_exposure_state) {
            sg_fully_covered[sg.id] = fully_covered ? 1U : 0U;
        }
        if (fully_covered) {
            ++result.fc_subgraphs;
            if (need_exposure_state) {
                for (const auto b : subgraph_boundaries_[sg.id]) {
                    const auto it = boundary_distances.find(b);
                    if (it != boundary_distances.end() && it->second <= radius) {
                        sg_seed_borders[sg.id].push_back(b);
                    }
                }
            }
            if (allow_whole_subgraph_acceptance) {
                for (const auto obj_id : objects.objects_in(sg.id)) {
                    if (mark_seen_object(obj_id)) {
                        result.object_ids.push_back(obj_id);
                        ++result.auto_included_objects;
                    }
                }
                continue;
            }
        }

        if (!fully_covered) {
            ++result.pc_subgraphs;
        }
        std::vector<std::pair<VertexId, EdgeWeight>> seeds;
        seeds.reserve(subgraph_boundaries_[sg.id].size());
        if (fast_dense_path) {
            for (const auto b : subgraph_boundaries_[sg.id]) {
                if (b < skeleton_scratch_stamp_.size() && skeleton_scratch_stamp_[b] == fast_boundary_epoch) {
                    const auto distance = skeleton_scratch_dist_[b];
                    if (distance <= radius) {
                        seeds.push_back({b, distance});
                    }
                }
            }
        } else {
            for (const auto b : subgraph_boundaries_[sg.id]) {
                const auto it = boundary_distances.find(b);
                if (it != boundary_distances.end() && it->second <= radius) {
                    seeds.push_back({b, it->second});
                    if (need_exposure_state) {
                        sg_seed_borders[sg.id].push_back(b);
                    }
                }
            }
        }
        if (seeds.empty()) {
            continue;
        }
        const auto dist = local_dijkstra_compact(subgraph_local_indices_[sg.id], seeds, radius);
        collect_partial_subgraph_objects(
            objects.populated_edges_in(sg.id),
            [&](VertexId v) { return lookup_local_distance(dist, v); },
            radius,
            objects,
            mark_seen_object,
            result
        );
    }

    result.touched_subgraphs = fast_dense_path ? fast_touched_subgraphs.size() : hit_subgraphs.size() + 1U;
    if (need_exposure_state) {
        std::unordered_map<SgId, std::size_t> overlap_by_subgraph;
        for (const auto obj_id : result.object_ids) {
            ++overlap_by_subgraph[objects.object_subgraph(obj_id)];
        }
        for (SgId sg_id = 0; sg_id < sg_seed_borders.size(); ++sg_id) {
            const auto& borders = sg_seed_borders[sg_id];
            if (borders.empty()) {
                continue;
            }
            const auto overlap = overlap_by_subgraph.contains(sg_id) ? overlap_by_subgraph.at(sg_id) : 0U;
            const bool triggered_full_cover = sg_fully_covered[sg_id] != 0U;
            for (const auto border : borders) {
                const auto key = std::make_pair(sg_id, border);
                const auto it = exposure_index.find(key);
                if (it == exposure_index.end()) {
                    continue;
                }
                auto& event = exposure_events->at(it->second);
                event.triggered_full_cover = triggered_full_cover;
                event.eventual_result_overlap = overlap;
                event.was_needed_for_final_answer = overlap != 0U;
                event.was_only_used_for_traversal = (overlap == 0U && !triggered_full_cover);
            }
        }
    }
    if (fast_dense_path) {
        for (const auto sg_id : fast_touched_subgraphs) {
            const auto touches = range_subgraph_touch_count_scratch_[sg_id];
            if (touches > 1U) {
                result.num_useful_subgraph_reentries += touches - 1U;
            }
        }
    } else {
        for (SgId sg_id = 0; sg_id < sg_touch_events.size(); ++sg_id) {
            const auto touches = sg_touch_events[sg_id];
            if (touches <= 1U) {
                continue;
            }
            const auto has_overlap = !sg_seed_borders[sg_id].empty();
            const auto is_fc = sg_fully_covered[sg_id] != 0U;
            if (has_overlap || is_fc) {
                result.num_useful_subgraph_reentries += touches - 1U;
            }
        }
    }
    if (need_row_shadow) {
        std::unordered_set<VertexId> settled_vertices;
        settled_vertices.reserve(boundary_trace.settled.size() * 2U + 1U);
        for (const auto& [vertex, distance] : boundary_trace.settled) {
            (void)distance;
            settled_vertices.insert(vertex);
        }
        for (const auto& pending : pending_rows) {
            auto& record = row_shadow_records->at(pending.index);
            for (const auto& [exit_border, cand] : pending.pushed_exits) {
                const auto it = boundary_distances.find(exit_border);
                if (it != boundary_distances.end() && it->second == cand && settled_vertices.contains(exit_border)) {
                    ++record.num_useful_pushes_from_this_row;
                    ++record.num_exits_useful;
                }
            }
        }
    }
    std::sort(result.object_ids.begin(), result.object_ids.end());
    return result;
}

std::vector<ObjId> SkeletonIndex::exact_range_query(
    const QueryPoint& query,
    EdgeWeight radius,
    const IndexedMovingObjectSet& objects
) const {
    const auto seeds = std::vector<std::pair<VertexId, EdgeWeight>>{
        {query.edge.first, query.offset},
        {query.edge.second, static_cast<EdgeWeight>(edge_weight_any(*global_, query.edge) - query.offset)},
    };
    const auto dist = dijkstra(*global_, seeds, radius);
    std::vector<ObjId> result;
    for (const auto& object : objects.objects()) {
        EdgeWeight best = kInfWeight;
        if (const auto it = dist.find(object.edge.first); it != dist.end() &&
            it->second <= kInfWeight - object.offset) {
            best = std::min(best, static_cast<EdgeWeight>(it->second + object.offset));
        }
        const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
        if (const auto it = dist.find(object.edge.second); it != dist.end() &&
            it->second <= kInfWeight - right_cost) {
            best = std::min(best, static_cast<EdgeWeight>(it->second + right_cost));
        }
        if (best <= radius) {
            result.push_back(object.unique_id);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

KnnQueryResult SkeletonIndex::knn_query(
    const QueryPoint& query,
    std::size_t k,
    const IndexedMovingObjectSet& objects,
    std::size_t query_id,
    std::vector<BorderExposureEvent>* exposure_events,
    std::vector<CliqueRowShadowRecord>* row_shadow_records,
    bool streamed_clique,
    bool safe_coverage_shadow,
    std::size_t parent_shadow_size,
    bool use_factorized_transfer,
    bool knn_direct_row_path,
    bool subgraph_admit,
    bool subgraph_admit_exact_cap,
    bool subgraph_admit_density_fallback
) const {
    if (k == 0 || objects.size() == 0) {
        return KnnQueryResult{};
    }

    using Clock = std::chrono::steady_clock;
    const auto t_started = Clock::now();
    const double object_density =
        (global_ == nullptr || global_->size() == 0)
            ? 0.0
            : static_cast<double>(objects.size()) / static_cast<double>(global_->size());
    const std::size_t occupied_subgraphs =
        (subgraph_admit && subgraph_admit_density_fallback) ? objects.occupied_subgraphs() : 0U;
    const double occupied_subgraph_ratio =
        subgraphs_.empty() ? 0.0 : static_cast<double>(occupied_subgraphs) / static_cast<double>(subgraphs_.size());
    const double avg_objects_per_occupied_subgraph =
        (subgraph_admit && subgraph_admit_density_fallback) ? objects.avg_objects_per_occupied_subgraph() : 0.0;
    const std::size_t max_objects_in_subgraph =
        (subgraph_admit && subgraph_admit_density_fallback) ? objects.max_objects_in_subgraph() : 0U;
    bool subgraph_admit_auto_disabled = false;
    if (subgraph_admit && subgraph_admit_density_fallback) {
        const bool strongly_clustered =
            occupied_subgraph_ratio <= 0.05 ||
            avg_objects_per_occupied_subgraph >= 32.0 ||
            max_objects_in_subgraph >= 128U;
        const bool strongly_scattered =
            occupied_subgraph_ratio >= 0.15 &&
            avg_objects_per_occupied_subgraph <= 8.0;
        const bool should_disable =
            strongly_scattered ||
            (object_density >= 0.10 && !strongly_clustered) ||
            (object_density >= 0.025 && occupied_subgraph_ratio >= 0.25 && avg_objects_per_occupied_subgraph <= 4.0);
        if (should_disable) {
            subgraph_admit = false;
            subgraph_admit_auto_disabled = true;
        }
    }

    const auto maybe_query_vertex = query_vertex_if_exact_vertex(query, *global_);
    const Subgraph* initial_ptr = nullptr;
    std::optional<VertexId> query_vertex;
    if (maybe_query_vertex.has_value()) {
        if (*maybe_query_vertex < vertex_to_subgraph_fast_.size() &&
            vertex_to_subgraph_fast_[*maybe_query_vertex] != std::numeric_limits<SgId>::max()) {
            initial_ptr = &subgraphs_.at(vertex_to_subgraph_fast_[*maybe_query_vertex]);
            query_vertex = *maybe_query_vertex;
        }
    }
    if (initial_ptr == nullptr) {
        initial_ptr = &require_initial_subgraph(query.edge);
    }
    const auto& initial = *initial_ptr;

    std::optional<LocalDijkstraResult> initial_local;
    std::vector<std::pair<VertexId, EdgeWeight>> initial_boundary_seeds;
    const bool initial_table_fast_path = !initial.distance.empty();
    Edge initial_query_edge{};
    EdgeWeight initial_query_edge_weight = 0;
    if (initial_table_fast_path) {
        if (query_vertex.has_value()) {
            initial_boundary_seeds = boundary_query_seeds_from_source_table(initial, *query_vertex);
        } else {
            initial_query_edge = ordered_edge(query.edge.first, query.edge.second);
            initial_query_edge_weight = edge_weight_any(initial.graph, initial_query_edge);
            if (query.offset >= initial_query_edge_weight) {
                throw std::runtime_error("query offset must be smaller than edge weight");
            }
            initial_boundary_seeds = boundary_query_seeds_from_edge_table(
                initial,
                initial_query_edge,
                query.offset,
                initial_query_edge_weight
            );
        }
    } else if (query_vertex.has_value()) {
        initial_local = local_dijkstra_compact(
            subgraph_local_indices_[initial.id],
            {{*query_vertex, 0}},
            kInfWeight
        );
        initial_boundary_seeds = boundary_query_seeds(initial, *initial_local);
    } else {
        initial_local = local_query_distances_compact(
            initial,
            subgraph_local_indices_[initial.id],
            query,
            kInfWeight
        );
        initial_boundary_seeds = boundary_query_seeds(initial, *initial_local);
    }

    const auto initial_exact_distance = [&](const MovingObject& object) -> EdgeWeight {
        if (initial_table_fast_path) {
            if (query_vertex.has_value()) {
                return object_distance_from_source_table(initial, *query_vertex, object);
            }
            return object_distance_from_query_edge_table(
                initial,
                initial_query_edge,
                query.offset,
                initial_query_edge_weight,
                object
            );
        }
        return object_distance_from_seed(*initial_local, object);
    };
    using CandidateItem = std::tuple<EdgeWeight, ObjId, SgId>;
    using ExactItem = std::tuple<EdgeWeight, ObjId>;
    using CoverageItem = std::tuple<EdgeWeight, ObjId, SgId>;
    struct SubgraphSetState {
        bool present{false};
        std::size_t object_count{0};
        std::size_t candidate_count{0};
        EdgeWeight ld{kInfWeight};
        EdgeWeight ud{kInfWeight};
    };

    std::vector<KnnFrontierItem> frontier_storage;
    frontier_storage.reserve(std::max<std::size_t>(64U, initial_boundary_seeds.size() * 16U + 64U));
    std::priority_queue<KnnFrontierItem, std::vector<KnnFrontierItem>, KnnFrontierGreater> frontier_heap(
        KnnFrontierGreater{},
        std::move(frontier_storage)
    );
    std::vector<CandidateItem> threshold_storage;
    threshold_storage.reserve(k + 1U);
    std::priority_queue<CandidateItem, std::vector<CandidateItem>, std::less<CandidateItem>> threshold_heap(
        std::less<CandidateItem>{},
        std::move(threshold_storage)
    );
    std::vector<ExactItem> exact_storage;
    exact_storage.reserve(k + 1U);
    std::priority_queue<ExactItem, std::vector<ExactItem>, std::less<ExactItem>> exact_heap(
        std::less<ExactItem>{},
        std::move(exact_storage)
    );
    std::vector<CoverageItem> safe_coverage_storage;
    if (safe_coverage_shadow) {
        safe_coverage_storage.reserve(k + 1U);
    }
    std::priority_queue<CoverageItem, std::vector<CoverageItem>, std::less<CoverageItem>> safe_coverage_heap(
        std::less<CoverageItem>{},
        std::move(safe_coverage_storage)
    );
    std::vector<CandidateItem> final_candidates;
    std::vector<std::pair<ObjId, EdgeWeight>> root_exact_candidates;
    if (subgraph_admit) {
        root_exact_candidates.reserve(k + 1U);
    }
    std::unordered_map<ObjId, EdgeWeight> admitted_exact_cache;
    if (subgraph_admit_exact_cap) {
        admitted_exact_cache.reserve(k * 2U + 1U);
    }
    if (skeleton_scratch_dist_.size() != static_cast<std::size_t>(max_vertex_id_) + 1U) {
        skeleton_scratch_dist_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_stamp_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    if (skeleton_scratch_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(skeleton_scratch_stamp_.begin(), skeleton_scratch_stamp_.end(), 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    ++skeleton_scratch_epoch_;
    const auto skeleton_epoch = skeleton_scratch_epoch_;
    std::vector<VertexId> touched_boundaries;
    touched_boundaries.reserve(initial_boundary_seeds.size() * 8U + 64U);
    const auto subgraph_count = subgraphs_.size();
    if (knn_best_subgraph_lb_scratch_.size() != subgraph_count) {
        knn_best_subgraph_lb_scratch_.resize(subgraph_count);
        knn_best_subgraph_ud_scratch_.resize(subgraph_count);
        knn_sg_touch_count_scratch_.resize(subgraph_count);
        knn_best_subgraph_lb_stamp_.assign(subgraph_count, 0U);
        knn_best_subgraph_ud_stamp_.assign(subgraph_count, 0U);
        knn_visited_subgraph_stamp_.assign(subgraph_count, 0U);
        knn_admitted_subgraph_stamp_.assign(subgraph_count, 0U);
        knn_pruned_subgraph_stamp_.assign(subgraph_count, 0U);
        knn_tightened_subgraph_stamp_.assign(subgraph_count, 0U);
        knn_sg_touch_stamp_.assign(subgraph_count, 0U);
        knn_subgraph_epoch_ = 0U;
    }
    if (knn_subgraph_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(knn_best_subgraph_lb_stamp_.begin(), knn_best_subgraph_lb_stamp_.end(), 0U);
        std::fill(knn_best_subgraph_ud_stamp_.begin(), knn_best_subgraph_ud_stamp_.end(), 0U);
        std::fill(knn_visited_subgraph_stamp_.begin(), knn_visited_subgraph_stamp_.end(), 0U);
        std::fill(knn_admitted_subgraph_stamp_.begin(), knn_admitted_subgraph_stamp_.end(), 0U);
        std::fill(knn_pruned_subgraph_stamp_.begin(), knn_pruned_subgraph_stamp_.end(), 0U);
        std::fill(knn_tightened_subgraph_stamp_.begin(), knn_tightened_subgraph_stamp_.end(), 0U);
        std::fill(knn_sg_touch_stamp_.begin(), knn_sg_touch_stamp_.end(), 0U);
        knn_subgraph_epoch_ = 0U;
    }
    ++knn_subgraph_epoch_;
    const auto knn_epoch = knn_subgraph_epoch_;

    const auto get_best_lb = [&](SgId sg_id) -> EdgeWeight {
        return knn_best_subgraph_lb_stamp_[sg_id] == knn_epoch
            ? knn_best_subgraph_lb_scratch_[sg_id]
            : kInfWeight;
    };
    const auto set_best_lb = [&](SgId sg_id, EdgeWeight value) {
        knn_best_subgraph_lb_stamp_[sg_id] = knn_epoch;
        knn_best_subgraph_lb_scratch_[sg_id] = value;
    };
    const auto get_best_ud = [&](SgId sg_id) -> EdgeWeight {
        return knn_best_subgraph_ud_stamp_[sg_id] == knn_epoch
            ? knn_best_subgraph_ud_scratch_[sg_id]
            : kInfWeight;
    };
    const auto set_best_ud = [&](SgId sg_id, EdgeWeight value) {
        knn_best_subgraph_ud_stamp_[sg_id] = knn_epoch;
        knn_best_subgraph_ud_scratch_[sg_id] = value;
    };
    const auto is_visited = [&](SgId sg_id) -> bool {
        return knn_visited_subgraph_stamp_[sg_id] == knn_epoch;
    };
    const auto mark_visited = [&](SgId sg_id) {
        knn_visited_subgraph_stamp_[sg_id] = knn_epoch;
    };
    const auto is_admitted = [&](SgId sg_id) -> bool {
        return knn_admitted_subgraph_stamp_[sg_id] == knn_epoch;
    };
    const auto mark_admitted = [&](SgId sg_id) {
        knn_admitted_subgraph_stamp_[sg_id] = knn_epoch;
    };
    const auto clear_admitted = [&](SgId sg_id) {
        knn_admitted_subgraph_stamp_[sg_id] = 0U;
    };
    const auto mark_pruned = [&](SgId sg_id) {
        knn_pruned_subgraph_stamp_[sg_id] = knn_epoch;
    };
    const auto is_pruned = [&](SgId sg_id) -> bool {
        return knn_pruned_subgraph_stamp_[sg_id] == knn_epoch;
    };
    const auto mark_tightened = [&](SgId sg_id) {
        knn_tightened_subgraph_stamp_[sg_id] = knn_epoch;
    };
    const auto is_tightened = [&](SgId sg_id) -> bool {
        return knn_tightened_subgraph_stamp_[sg_id] == knn_epoch;
    };
    const auto increment_touch_count = [&](SgId sg_id) -> std::size_t {
        if (knn_sg_touch_stamp_[sg_id] != knn_epoch) {
            knn_sg_touch_stamp_[sg_id] = knn_epoch;
            knn_sg_touch_count_scratch_[sg_id] = 0U;
        }
        return ++knn_sg_touch_count_scratch_[sg_id];
    };
    const auto get_touch_count = [&](SgId sg_id) -> std::size_t {
        return knn_sg_touch_stamp_[sg_id] == knn_epoch
            ? knn_sg_touch_count_scratch_[sg_id]
            : 0U;
    };
    std::vector<SubgraphSetState> subgraph_set_states;
    subgraph_set_states.resize(subgraph_count);
    std::vector<SgId> visited_subgraph_list;
    std::vector<SgId> candidate_subgraph_list;
    std::vector<SgId> admitted_subgraph_list_live;
    visited_subgraph_list.reserve(initial_boundary_seeds.size() * 4U + 16U);
    candidate_subgraph_list.reserve(initial_boundary_seeds.size() * 4U + 16U);
    admitted_subgraph_list_live.reserve(initial_boundary_seeds.size() * 2U + 8U);
    std::unordered_map<std::pair<SgId, VertexId>, std::size_t, PairHash> exposure_index;
    std::unordered_map<SgId, std::size_t> trigger_event_index;
    std::size_t num_clique_relax_attempts = 0;
    std::size_t num_successful_clique_relaxes = 0;
    std::size_t num_pq_pushes_from_clique = 0;
    std::size_t num_redundant_subgraph_reentries = 0;
    std::size_t num_useful_subgraph_reentries = 0;
    std::vector<VertexId> settled_boundaries;
    if (row_shadow_records != nullptr) {
        settled_boundaries.reserve(initial_boundary_seeds.size() * 8U + 64U);
    }
    struct PendingRowShadow {
        std::size_t index{0};
        std::vector<std::pair<VertexId, EdgeWeight>> pushed_exits;
    };
    struct StreamedCliqueRow {
        std::vector<std::pair<VertexId, EdgeWeight>> emissions;
        std::size_t next_index{0};
    };
    std::vector<PendingRowShadow> pending_rows;
    std::vector<StreamedCliqueRow> streamed_rows;
    streamed_rows.reserve(initial_boundary_seeds.size() + 16U);

    long long pq_us = 0;
    long long membership_us = 0;
    long long subgraph_bookkeeping_us = 0;
    long long clique_emit_us = 0;
    std::size_t boundary_pq_pushes = 0;
    std::size_t boundary_pq_pops = 0;
    std::size_t row_state_pushes = 0;
    std::size_t row_state_pops = 0;
    std::size_t streamed_rows_started = 0;
    std::size_t streamed_row_emissions = 0;
    std::size_t streamed_rows_stopped_by_threshold = 0;
    std::size_t streamed_exits_skipped_by_threshold = 0;
    std::size_t peak_frontier_queue_size = 0;
    std::size_t safe_coverage_candidate_count = 0;
    std::size_t first_k_candidates_boundary_visit_order = 0;
    EdgeWeight first_finite_safe_coverage_radius = kInfWeight;
    std::size_t first_finite_safe_coverage_boundary_visit_order = 0;
    EdgeWeight safe_coverage_radius = kInfWeight;
    EdgeWeight subgraph_safe_coverage_exact_cap = kInfWeight;
    std::size_t safe_coverage_updates = 0;
    EdgeWeight subgraph_safe_coverage_radius = kInfWeight;
    bool subgraph_safe_coverage_established = false;
    std::size_t object_set_total_objects = 0;
    std::size_t admitted_subgraphs_live = 0;
    std::size_t admitted_objects_live = 0;
    std::size_t admission_refresh_calls = 0;
    std::size_t admission_refresh_successes = 0;
    std::size_t admission_ud_updates = 0;
    std::size_t admission_gate_heap_fail = 0;
    std::size_t admission_gate_mass_fail = 0;
    std::size_t admission_gate_avg_fail = 0;
    std::size_t admission_gate_dt_fail = 0;
    std::size_t admission_gate_no_admit_fail = 0;
    std::size_t admission_gate_object_fail = 0;
    std::size_t finite_ud_candidate_subgraphs = 0;
    long long best_admission_margin = 0;
    constexpr std::size_t kMinAdmittedSubgraphObjects = 1U;
    bool subgraph_admission_refresh_pending = false;
    EdgeWeight first_finite_tau = kInfWeight;
    std::size_t first_finite_tau_boundary_visit_order = 0;
    EdgeWeight last_observed_tau = kInfWeight;
    std::size_t tau_updates = 0;
    std::size_t would_stop_rows_by_safe_coverage = 0;
    std::size_t would_skip_exits_by_safe_coverage = 0;
    std::size_t would_save_pq_pushes_by_safe_coverage = 0;
    std::size_t factorized_rows_used = 0;
    std::size_t factorized_hubs_used = 0;
    std::size_t factorized_exits_emitted = 0;
    EdgeWeight root_kth_exact = kInfWeight;

    const auto boundary_distance = [&](VertexId v) -> EdgeWeight {
        return (v < skeleton_scratch_stamp_.size() && skeleton_scratch_stamp_[v] == skeleton_epoch)
            ? skeleton_scratch_dist_[v]
            : kInfWeight;
    };

    mark_visited(initial.id);
    visited_subgraph_list.push_back(initial.id);
    std::size_t visited_subgraph_count = 1;
    std::size_t candidates_considered = 0;
    std::size_t boundary_visit_order = 0;

    auto ensure_exposure = [&](SgId sg_id, VertexId border, EdgeWeight distance) -> std::size_t {
        if (exposure_events == nullptr) {
            return static_cast<std::size_t>(-1);
        }
        const auto key = std::make_pair(sg_id, border);
        if (const auto it = exposure_index.find(key); it != exposure_index.end()) {
            auto& event = exposure_events->at(it->second);
            event.distance_from_query = std::min(event.distance_from_query, distance);
            return it->second;
        }
        const auto index = exposure_events->size();
        exposure_index[key] = index;
        exposure_events->push_back(BorderExposureEvent{
            query_id,
            QueryType::Knn,
            sg_id,
            border,
            boundary_visit_order,
            false,
            false,
            false,
            false,
            false,
            distance,
            0,
        });
        return index;
    };

    const auto observe_tau = [&](EdgeWeight tau, std::size_t order) {
        if (tau == last_observed_tau) {
            return;
        }
        ++tau_updates;
        last_observed_tau = tau;
        if (tau != kInfWeight && first_finite_tau == kInfWeight) {
            first_finite_tau = tau;
            first_finite_tau_boundary_visit_order = order;
        }
    };

    const auto observe_safe_coverage_candidate = [&](EdgeWeight upper_bound, ObjId obj_id, SgId sg_id, std::size_t order) {
        if (!safe_coverage_shadow || upper_bound == kInfWeight) {
            return;
        }
        ++safe_coverage_candidate_count;
        push_bounded_candidate(safe_coverage_heap, k, CoverageItem{upper_bound, obj_id, sg_id});
        if (safe_coverage_heap.size() < k) {
            return;
        }
        if (first_k_candidates_boundary_visit_order == 0) {
            first_k_candidates_boundary_visit_order = order;
        }
        const auto new_radius = std::get<0>(safe_coverage_heap.top());
        if (new_radius == safe_coverage_radius) {
            return;
        }
        safe_coverage_radius = new_radius;
        ++safe_coverage_updates;
        if (first_finite_safe_coverage_radius == kInfWeight) {
            first_finite_safe_coverage_radius = new_radius;
            first_finite_safe_coverage_boundary_visit_order = order;
        }
    };

    const auto clear_subgraph_admission = [&]() {
        if (!subgraph_safe_coverage_established && admitted_subgraph_list_live.empty()) {
            return;
        }
        admitted_subgraphs_live = 0;
        admitted_objects_live = 0;
        subgraph_safe_coverage_established = false;
        subgraph_safe_coverage_radius = kInfWeight;
        subgraph_safe_coverage_exact_cap = kInfWeight;
        admitted_exact_cache.clear();
        for (const auto sg_id : admitted_subgraph_list_live) {
            clear_admitted(sg_id);
        }
        admitted_subgraph_list_live.clear();
    };

    const auto refresh_subgraph_admission = [&]() {
        if (!subgraph_admit) {
            return;
        }
        ++admission_refresh_calls;
        subgraph_admission_refresh_pending = false;
        if (candidate_subgraph_list.empty()) {
            ++admission_gate_heap_fail;
            clear_subgraph_admission();
            return;
        }
        if (object_set_total_objects < k) {
            ++admission_gate_mass_fail;
            clear_subgraph_admission();
            return;
        }
        EdgeWeight dt = 0;
        bool dt_ready = false;
        for (const auto sg_id : candidate_subgraph_list) {
            const auto& state = subgraph_set_states[sg_id];
            if (!state.present || state.ld == kInfWeight) {
                continue;
            }
            if (!dt_ready || state.ld > dt) {
                dt = state.ld;
                dt_ready = true;
            }
        }
        if (!dt_ready) {
            ++admission_gate_dt_fail;
            clear_subgraph_admission();
            return;
        }

        admitted_subgraphs_live = 0;
        admitted_objects_live = 0;
        subgraph_safe_coverage_exact_cap = kInfWeight;
        admitted_exact_cache.clear();
        for (const auto sg_id : admitted_subgraph_list_live) {
            clear_admitted(sg_id);
        }
        admitted_subgraph_list_live.clear();
        EdgeWeight du = 0;
        bool have_remaining = false;
        finite_ud_candidate_subgraphs = 0;
        bool best_margin_ready = false;
        for (const auto sg_id : candidate_subgraph_list) {
            const auto& state = subgraph_set_states[sg_id];
            if (!state.present || state.ud == kInfWeight) {
                clear_admitted(sg_id);
                continue;
            }
            ++finite_ud_candidate_subgraphs;
            const auto margin = static_cast<long long>(state.ud) - static_cast<long long>(dt);
            if (!best_margin_ready || margin < best_admission_margin) {
                best_admission_margin = margin;
                best_margin_ready = true;
            }
            if (state.ud <= dt && state.object_count >= kMinAdmittedSubgraphObjects) {
                mark_admitted(sg_id);
                admitted_subgraph_list_live.push_back(sg_id);
                ++admitted_subgraphs_live;
                admitted_objects_live += state.object_count;
            } else {
                clear_admitted(sg_id);
                du = std::max(du, state.ud);
                have_remaining = true;
            }
        }
        if (!best_margin_ready) {
            best_admission_margin = 0;
        }
        if (admitted_subgraphs_live == 0U) {
            ++admission_gate_no_admit_fail;
        }
        if (admitted_objects_live < k && admitted_subgraphs_live != 0U) {
            ++admission_gate_object_fail;
        }
        if (finite_ud_candidate_subgraphs == 0U) {
            clear_subgraph_admission();
            return;
        }
        subgraph_safe_coverage_established = true;
        subgraph_safe_coverage_radius = have_remaining ? du : 0U;
        ++admission_refresh_successes;
    };

    for (const auto obj_id : objects.objects_in(initial.id)) {
        const auto& object = objects[obj_id];
        const auto exact = initial_exact_distance(object);
        if (exact != kInfWeight) {
            ++candidates_considered;
            if (subgraph_admit) {
                root_exact_candidates.push_back({object.unique_id, exact});
            } else {
                push_bounded_candidate(threshold_heap, k, CandidateItem{exact, object.unique_id, initial.id});
            }
            observe_safe_coverage_candidate(exact, object.unique_id, initial.id, 0);
        }
    }
    if (subgraph_admit) {
        object_set_total_objects = root_exact_candidates.size();
        if (root_exact_candidates.size() >= k) {
            std::vector<EdgeWeight> root_exact_distances;
            root_exact_distances.reserve(root_exact_candidates.size());
            for (const auto& [obj_id, exact] : root_exact_candidates) {
                (void)obj_id;
                root_exact_distances.push_back(exact);
            }
            std::nth_element(
                root_exact_distances.begin(),
                root_exact_distances.begin() + static_cast<std::ptrdiff_t>(k - 1),
                root_exact_distances.end()
            );
            root_kth_exact = root_exact_distances[k - 1];
        }
    }

    const auto current_knn_threshold = [&]() -> EdgeWeight {
        if (subgraph_admit) {
            if (subgraph_safe_coverage_established) {
                return std::min(subgraph_safe_coverage_radius, subgraph_safe_coverage_exact_cap);
            }
            return root_kth_exact;
        }
        return (threshold_heap.size() >= k) ? std::get<0>(threshold_heap.top()) : kInfWeight;
    };

    const auto maybe_update_subgraph_exact_cap = [&]() {
        if (!subgraph_admit || !subgraph_admit_exact_cap || !subgraph_safe_coverage_established) {
            return;
        }
        if (subgraph_safe_coverage_exact_cap != kInfWeight) {
            return;
        }
        std::priority_queue<ExactItem> cap_heap;
        for (const auto& [obj_id, exact] : root_exact_candidates) {
            push_bounded_candidate(cap_heap, k, ExactItem{exact, obj_id});
        }
        for (const auto sg_id : admitted_subgraph_list_live) {
            for (const auto obj_id : objects.objects_in(sg_id)) {
                EdgeWeight exact = kInfWeight;
                for (const auto& [b, border_cost] : objects.knn_border_costs(obj_id)) {
                    const auto border_dist = boundary_distance(b);
                    if (border_dist == kInfWeight || border_dist > kInfWeight - border_cost) {
                        continue;
                    }
                    exact = std::min(exact, static_cast<EdgeWeight>(border_dist + border_cost));
                }
                if (exact != kInfWeight) {
                    admitted_exact_cache[obj_id] = exact;
                    push_bounded_candidate(cap_heap, k, ExactItem{exact, obj_id});
                }
            }
        }
        if (cap_heap.size() >= k) {
            subgraph_safe_coverage_exact_cap = std::get<0>(cap_heap.top());
        }
    };

    observe_tau(current_knn_threshold(), 0);

    const auto frontier_push = [&](const KnnFrontierItem& item) {
        if constexpr (kEnableKnnFineTiming) {
            const auto t0 = Clock::now();
            frontier_heap.push(item);
            pq_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
        } else {
            frontier_heap.push(item);
        }
        if (item.kind == KnnFrontierItem::Kind::Boundary) {
            ++boundary_pq_pushes;
        } else {
            ++row_state_pushes;
        }
        peak_frontier_queue_size = std::max(peak_frontier_queue_size, frontier_heap.size());
    };

    const auto frontier_pop = [&]() {
        KnnFrontierItem item;
        if constexpr (kEnableKnnFineTiming) {
            const auto t0 = Clock::now();
            item = frontier_heap.top();
            frontier_heap.pop();
            pq_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
        } else {
            item = frontier_heap.top();
            frontier_heap.pop();
        }
        if (item.kind == KnnFrontierItem::Kind::Boundary) {
            ++boundary_pq_pops;
        } else {
            ++row_state_pops;
        }
        return item;
    };

    const auto try_schedule_next_row_emission =
        [&](std::size_t row_id, bool count_start) -> bool {
            auto& row = streamed_rows[row_id];
            if (row.next_index >= row.emissions.size()) {
                return false;
            }
            const auto current_threshold = current_knn_threshold();
            const auto& [next_border, next_distance] = row.emissions[row.next_index];
            if (current_threshold != kInfWeight && next_distance > current_threshold) {
                ++streamed_rows_stopped_by_threshold;
                streamed_exits_skipped_by_threshold += row.emissions.size() - row.next_index;
                row.next_index = row.emissions.size();
                return false;
            }
            if (count_start) {
                ++streamed_rows_started;
            }
            ++row.next_index;
            frontier_push(KnnFrontierItem{
                next_distance,
                KnnFrontierItem::Kind::RowEmit,
                next_border,
                row_id,
            });
            return true;
        };

    const auto update_boundary_state = [&](VertexId b, EdgeWeight distance, bool from_clique) {
        if constexpr (kEnableKnnFineTiming) {
            const auto t0_membership = Clock::now();
            if (skeleton_scratch_stamp_[b] == skeleton_epoch) {
                auto& best_distance = skeleton_scratch_dist_[b];
                if (distance >= best_distance) {
                    membership_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_membership).count();
                    return false;
                }
                best_distance = distance;
            } else {
                skeleton_scratch_stamp_[b] = skeleton_epoch;
                skeleton_scratch_dist_[b] = distance;
                touched_boundaries.push_back(b);
            }
            membership_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_membership).count();
        } else {
            if (skeleton_scratch_stamp_[b] == skeleton_epoch) {
                auto& best_distance = skeleton_scratch_dist_[b];
                if (distance >= best_distance) {
                    return false;
                }
                best_distance = distance;
            } else {
                skeleton_scratch_stamp_[b] = skeleton_epoch;
                skeleton_scratch_dist_[b] = distance;
                touched_boundaries.push_back(b);
            }
        }

        frontier_push(KnnFrontierItem{distance, KnnFrontierItem::Kind::Boundary, b, 0});
        if (from_clique) {
            ++num_successful_clique_relaxes;
            ++num_pq_pushes_from_clique;
        }

        const auto t0_bookkeeping = kEnableKnnFineTiming ? Clock::now() : Clock::time_point{};
        if (b < inverted_index_fast_.size()) {
            for (const auto sg_id : inverted_index_fast_[b]) {
                const auto sg_best = get_best_lb(sg_id);
                if (distance < sg_best) {
                    set_best_lb(sg_id, distance);
                }
                if (subgraph_admit) {
                    const auto& sg = subgraphs_.at(sg_id);
                    if (const auto rb_it = sg.rb_map.find(b); rb_it != sg.rb_map.end()) {
                        const auto whole = rb_it->second.whole;
            if (distance <= kInfWeight - whole) {
                const auto candidate_ud = static_cast<EdgeWeight>(
                    distance + whole + (rb_it->second.half ? 1U : 0U)
                );
                    const auto sg_ud = get_best_ud(sg_id);
                    if (candidate_ud < sg_ud) {
                        set_best_ud(sg_id, candidate_ud);
                        ++admission_ud_updates;
                        auto& state = subgraph_set_states[sg_id];
                        if (state.present && candidate_ud < state.ud) {
                            state.ud = candidate_ud;
                            if (!subgraph_safe_coverage_established && object_set_total_objects >= k) {
                                subgraph_admission_refresh_pending = true;
                            }
                        }
                    }
                }
                    }
                }
            }
        }
        if constexpr (kEnableKnnFineTiming) {
            subgraph_bookkeeping_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_bookkeeping).count();
        }
        return true;
    };

    for (const auto& [b, distance] : initial_boundary_seeds) {
        update_boundary_state(b, distance, false);
    }
    const auto t_after_init = Clock::now();

    while (!frontier_heap.empty()) {
        if (subgraph_admission_refresh_pending) {
            refresh_subgraph_admission();
            maybe_update_subgraph_exact_cap();
        }
        const auto item = frontier_pop();
        const auto current_threshold = current_knn_threshold();
        if (current_threshold != kInfWeight && item.distance > current_threshold) {
            break;
        }

        if (item.kind == KnnFrontierItem::Kind::RowEmit) {
            ++streamed_row_emissions;
            const auto t0_emit = kEnableKnnFineTiming ? Clock::now() : Clock::time_point{};
            auto& row = streamed_rows[item.row_id];
            update_boundary_state(item.vertex, item.distance, true);
            try_schedule_next_row_emission(item.row_id, false);
            if constexpr (kEnableKnnFineTiming) {
                clique_emit_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_emit).count();
            }
            continue;
        }

        const auto dist_vb = item.distance;
        const auto v_b = item.vertex;
        ++boundary_visit_order;

        const auto t0_membership = kEnableKnnFineTiming ? Clock::now() : Clock::time_point{};
        const bool stale = v_b >= skeleton_scratch_stamp_.size() || boundary_distance(v_b) != dist_vb;
        const bool missing_inverted =
            v_b >= inverted_boundary_index_fast_.size() || inverted_boundary_index_fast_[v_b].empty();
        if constexpr (kEnableKnnFineTiming) {
            membership_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_membership).count();
        }
        if (stale) {
            continue;
        }

        if (row_shadow_records != nullptr) {
            settled_boundaries.push_back(v_b);
        }
        if (missing_inverted) {
            continue;
        }

        for (const auto& boundary_entry : inverted_boundary_index_fast_[v_b]) {
            const auto sg_id = boundary_entry.subgraph_id;
            const auto clique_row_id = static_cast<std::size_t>(boundary_entry.clique_row_id);
            EdgeWeight sg_lb = kInfWeight;
            std::size_t event_idx = static_cast<std::size_t>(-1);
            bool first_visit = false;
            {
                const auto t0 = kEnableKnnFineTiming ? Clock::now() : Clock::time_point{};
                const auto touch_count = increment_touch_count(sg_id);
                if (touch_count > 1U) {
                    ++num_redundant_subgraph_reentries;
                }
                event_idx = ensure_exposure(sg_id, v_b, dist_vb);
                first_visit = !is_visited(sg_id);
                if (first_visit) {
                    mark_visited(sg_id);
                    visited_subgraph_list.push_back(sg_id);
                    ++visited_subgraph_count;
                    if (event_idx != static_cast<std::size_t>(-1)) {
                        trigger_event_index.try_emplace(sg_id, event_idx);
                    }
                }
                sg_lb = get_best_lb(sg_id);
                if constexpr (kEnableKnnFineTiming) {
                    membership_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
                }
            }
            const auto& sg = subgraphs_.at(sg_id);
            if (!first_visit) {
                continue;
            }
            if (sg_lb == kInfWeight) {
                continue;
            }

            auto threshold = current_knn_threshold();
            const auto threshold_before = threshold;
            bool pruned = false;
            const bool can_direct_emit =
                knn_direct_row_path &&
                !use_factorized_transfer &&
                !streamed_clique &&
                row_shadow_records == nullptr &&
                !safe_coverage_shadow;
            const bool sg_factorized =
                use_factorized_transfer &&
                sg_id < factorized_transfer_enabled_.size() &&
                factorized_transfer_enabled_[sg_id] != 0U;
            const bool direct_factorized_emit = false;
            const bool direct_explicit_emit = can_direct_emit && !sg_factorized;
            MaterializedCliqueRow materialized_row;
            std::vector<CliqueExitCandidate> exits;
            if (safe_coverage_shadow) {
                const auto t0 = kEnableKnnFineTiming ? Clock::now() : Clock::time_point{};
                for (const auto obj_id : objects.objects_in(sg_id)) {
                    const auto& object = objects[obj_id];
                    const auto upper_bound =
                        object_distance_from_boundary_costs(
                            objects.knn_border_costs(object.unique_id),
                            v_b,
                            dist_vb
                        );
                    observe_safe_coverage_candidate(upper_bound, object.unique_id, sg_id, boundary_visit_order);
                }
                if constexpr (kEnableKnnFineTiming) {
                    subgraph_bookkeeping_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
                }
            }
            std::size_t subgraph_candidate_count = 0;
            const auto should_defer_subgraph_objects = subgraph_admit && sg_id != initial.id;
            if (!should_defer_subgraph_objects) {
                const auto t0 = kEnableKnnFineTiming ? Clock::now() : Clock::time_point{};
                for (const auto obj_id : objects.objects_in_knn_order(sg_id)) {
                    const auto& object = objects[obj_id];
                    const auto suffix = objects.knn_suffix(obj_id);
                    if (suffix == kInfWeight) {
                        continue;
                    }
                    ++candidates_considered;
                    if (threshold != kInfWeight) {
                        if (sg_lb > threshold) {
                            pruned = true;
                            break;
                        }
                        const auto budget = static_cast<EdgeWeight>(threshold - sg_lb);
                        if (suffix > budget) {
                            pruned = true;
                            break;
                        }
                    }
                    const auto lower_bound = static_cast<EdgeWeight>(sg_lb + suffix);
                    if (lower_bound != kInfWeight) {
                        push_bounded_candidate(threshold_heap, k, CandidateItem{lower_bound, object.unique_id, sg_id});
                        ++subgraph_candidate_count;
                        if (threshold_heap.size() >= k) {
                            threshold = std::get<0>(threshold_heap.top());
                        }
                    }
                }
                if constexpr (kEnableKnnFineTiming) {
                    subgraph_bookkeeping_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
                }
            }
            if (should_defer_subgraph_objects) {
                auto& state = subgraph_set_states[sg_id];
                if (!state.present) {
                    const auto object_count = objects.objects_in(sg_id).size();
                    if (object_count != 0U) {
                        state.present = true;
                        state.object_count = object_count;
                        state.candidate_count = subgraph_candidate_count;
                        state.ld = sg_lb;
                        state.ud = get_best_ud(sg_id);
                        candidate_subgraph_list.push_back(sg_id);
                        object_set_total_objects += state.object_count;
                        if (!subgraph_safe_coverage_established && object_set_total_objects >= k) {
                            subgraph_admission_refresh_pending = true;
                        }
                    }
                } else if (subgraph_candidate_count > state.candidate_count) {
                    state.candidate_count = subgraph_candidate_count;
                }
                const auto sg_ud = get_best_ud(sg_id);
                if (sg_ud < state.ud) {
                    state.ud = sg_ud;
                    if (!subgraph_safe_coverage_established && object_set_total_objects >= k) {
                        subgraph_admission_refresh_pending = true;
                    }
                }
                if (!subgraph_safe_coverage_established && subgraph_admission_refresh_pending) {
                    refresh_subgraph_admission();
                    maybe_update_subgraph_exact_cap();
                }
            }
            if (pruned) {
                mark_pruned(sg_id);
            }
            const auto threshold_after = current_knn_threshold();
            observe_tau(threshold_after, boundary_visit_order);
            if (threshold_after < threshold_before) {
                mark_tightened(sg_id);
            }
            if (!direct_explicit_emit && !direct_factorized_emit) {
                const auto t0 = kEnableKnnFineTiming ? Clock::now() : Clock::time_point{};
                materialized_row = materialize_subgraph_exit_candidates(
                    v_b,
                    dist_vb,
                    sg_id,
                    subgraph_clique_rows_,
                    factorized_transfer_models_,
                    factorized_transfer_enabled_,
                    use_factorized_transfer,
                    threshold_after
                );
                if (materialized_row.used_factorized) {
                    ++factorized_rows_used;
                    factorized_hubs_used += materialized_row.hubs_used;
                    factorized_exits_emitted += materialized_row.exits.size();
                }
                exits = std::move(materialized_row.exits);
                if constexpr (kEnableKnnFineTiming) {
                    clique_emit_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
                }
            }

            if (row_shadow_records != nullptr) {
                CliqueRowShadowRecord record;
                record.query_id = query_id;
                record.query_type = QueryType::Knn;
                record.subgraph_id = sg_id;
                record.entry_border = v_b;
                record.entry_visit_order = boundary_visit_order;
                record.entry_distance = dist_vb;
                record.threshold_at_entry = threshold_after;
                record.num_exits_total = exits.size();
                record.useful_prefix_len = exits.size();

                PendingRowShadow pending;
                pending.index = row_shadow_records->size();

                for (std::size_t i = 0; i < exits.size(); ++i) {
                    const auto& exit = exits[i];
                    const auto exit_border = exit.border;
                    const auto cand = exit.candidate_distance;
                    const auto current_best = boundary_distance(exit_border);
                    const bool dist_dominated = cand >= current_best;
                    const bool threshold_dead = threshold_after != kInfWeight && cand >= threshold_after;
                    if (dist_dominated) {
                        ++record.num_exits_dist_dominated;
                    }
                    if (threshold_dead) {
                        ++record.num_exits_threshold_dead;
                        record.useful_prefix_len = std::min(record.useful_prefix_len, i);
                    }
                    if (!dist_dominated) {
                        ++record.num_pq_pushes_from_this_row;
                        if (i >= record.useful_prefix_len) {
                            ++record.num_pq_pushes_beyond_useful_prefix;
                        }
                        pending.pushed_exits.push_back({exit_border, cand});
                    }
                }

                row_shadow_records->push_back(record);
                pending_rows.push_back(std::move(pending));
            }

            if (safe_coverage_shadow && safe_coverage_radius != kInfWeight) {
                bool row_stopped = false;
                std::size_t row_skipped = 0;
                std::size_t row_saved_pushes = 0;
                for (std::size_t i = 0; i < exits.size(); ++i) {
                    const auto& exit = exits[i];
                    if (exit.candidate_distance <= safe_coverage_radius) {
                        continue;
                    }
                    row_stopped = true;
                    row_skipped = exits.size() - i;
                    for (std::size_t j = i; j < exits.size(); ++j) {
                        const auto current_best = boundary_distance(exits[j].border);
                        if (exits[j].candidate_distance < current_best) {
                            ++row_saved_pushes;
                        }
                    }
                    break;
                }
                if (row_stopped) {
                    ++would_stop_rows_by_safe_coverage;
                    would_skip_exits_by_safe_coverage += row_skipped;
                    would_save_pq_pushes_by_safe_coverage += row_saved_pushes;
                }
            }

            const auto t0_emit = kEnableKnnFineTiming ? Clock::now() : Clock::time_point{};
            if (direct_explicit_emit) {
                const auto emit_threshold = threshold_after;
                const std::vector<std::pair<VertexId, EdgeWeight>>* row_ptr = nullptr;
                if (sg_id < subgraph_clique_row_storage_.size() &&
                    clique_row_id < subgraph_clique_row_storage_[sg_id].size()) {
                    row_ptr = &subgraph_clique_row_storage_[sg_id][clique_row_id];
                }
                const auto& row =
                    (row_ptr != nullptr) ? *row_ptr : lookup_subgraph_clique_row(v_b, subgraph_clique_rows_[sg_id]);
                for (const auto& [exit_border, local] : row) {
                    if (dist_vb == kInfWeight || local == kInfWeight || dist_vb > kInfWeight - local) {
                        continue;
                    }
                    const auto candidate_distance = static_cast<EdgeWeight>(dist_vb + local);
                    if (emit_threshold != kInfWeight && candidate_distance > emit_threshold) {
                        break;
                    }
                    ++num_clique_relax_attempts;
                    update_boundary_state(exit_border, candidate_distance, true);
                }
            } else if (direct_factorized_emit) {
                const auto emit_threshold = threshold_after;
                const auto direct_stats = for_each_factorized_subgraph_exit_candidate(
                    v_b,
                    dist_vb,
                    sg_id,
                    factorized_transfer_models_,
                    factorized_transfer_enabled_,
                    emit_threshold,
                    [&](VertexId exit_border, EdgeWeight candidate_distance) {
                        ++num_clique_relax_attempts;
                        update_boundary_state(exit_border, candidate_distance, true);
                    }
                );
                if (direct_stats.used_factorized) {
                    ++factorized_rows_used;
                    factorized_hubs_used += direct_stats.hubs_used;
                    factorized_exits_emitted += direct_stats.exits_emitted;
                }
            } else if (streamed_clique) {
                if (!exits.empty()) {
                    StreamedCliqueRow row;
                    row.emissions.reserve(exits.size());
                    for (const auto& exit : exits) {
                        ++num_clique_relax_attempts;
                        row.emissions.push_back({exit.border, exit.candidate_distance});
                    }
                    const auto row_id = streamed_rows.size();
                    streamed_rows.push_back(std::move(row));
                    try_schedule_next_row_emission(row_id, true);
                }
            } else {
                for (const auto& exit : exits) {
                    ++num_clique_relax_attempts;
                    update_boundary_state(exit.border, exit.candidate_distance, true);
                }
            }
            if constexpr (kEnableKnnFineTiming) {
                clique_emit_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0_emit).count();
            }
        }
    }
    const auto t_after_explore = Clock::now();
    const auto final_tau_after_explore = current_knn_threshold();

    EdgeWeight kth_exact = kInfWeight;
    std::size_t exact_evaluated = 0;
    std::size_t exact_from_admitted = 0;
    std::size_t exact_from_unresolved = 0;
    std::size_t exact_reused_from_admitted_cache = 0;
    const auto admitted_subgraphs = admitted_subgraphs_live;
    const auto admitted_objects = admitted_objects_live;

    const auto evaluate_exact_object =
        [&](ObjId obj_id, SgId sg_id) -> std::optional<EdgeWeight> {
            EdgeWeight exact = kInfWeight;
            if (sg_id == initial.id) {
                const auto& object = objects[obj_id];
                exact = initial_exact_distance(object);
            } else {
                const auto sg_lb = get_best_lb(sg_id);
                EdgeWeight border_cost_budget = kInfWeight;
                if (exact_heap.size() >= k && kth_exact != kInfWeight && sg_lb != kInfWeight && sg_lb < kth_exact) {
                    border_cost_budget = static_cast<EdgeWeight>(kth_exact - sg_lb);
                }
                for (const auto& [b, border_cost] : objects.knn_border_costs(obj_id)) {
                    if (border_cost_budget != kInfWeight && border_cost > border_cost_budget) {
                        break;
                    }
                    const auto border_distance = boundary_distance(b);
                    if (border_distance == kInfWeight || border_distance > kInfWeight - border_cost) {
                        continue;
                    }
                    exact = std::min(exact, static_cast<EdgeWeight>(border_distance + border_cost));
                }
            }
            if (exact == kInfWeight) {
                return std::nullopt;
            }
            return exact;
        };

    const auto push_exact_item = [&](ObjId obj_id, EdgeWeight exact) {
        push_bounded_candidate(exact_heap, k, ExactItem{exact, obj_id});
        if (exact_heap.size() == k) {
            kth_exact = std::get<0>(exact_heap.top());
        }
    };

    if (subgraph_admit) {
        for (const auto& [obj_id, exact] : root_exact_candidates) {
            push_exact_item(obj_id, exact);
        }
        for (const auto sg_id : admitted_subgraph_list_live) {
            for (const auto obj_id : objects.objects_in(sg_id)) {
                if (const auto it = admitted_exact_cache.find(obj_id); it != admitted_exact_cache.end()) {
                    ++exact_reused_from_admitted_cache;
                    push_exact_item(obj_id, it->second);
                    continue;
                }
                ++exact_evaluated;
                ++exact_from_admitted;
                const auto exact = evaluate_exact_object(obj_id, sg_id);
                if (!exact.has_value()) {
                    continue;
                }
                push_exact_item(obj_id, *exact);
            }
        }

        const auto unresolved_target =
            (admitted_objects_live >= k) ? 0U : (k - admitted_objects_live);
        for (const auto sg_id : candidate_subgraph_list) {
            if (unresolved_target == 0U) {
                break;
            }
            if (is_admitted(sg_id)) {
                continue;
            }
            const auto sg_lb = get_best_lb(sg_id);
            if (sg_lb == kInfWeight) {
                continue;
            }
            auto threshold = kth_exact;
            bool pruned = false;
            for (const auto obj_id : objects.objects_in_knn_order(sg_id)) {
                const auto suffix = objects.knn_suffix(obj_id);
                if (suffix == kInfWeight) {
                    continue;
                }
                ++candidates_considered;
                if (threshold != kInfWeight) {
                    if (sg_lb > threshold) {
                        pruned = true;
                        break;
                    }
                    const auto budget = static_cast<EdgeWeight>(threshold - sg_lb);
                    if (suffix > budget) {
                        pruned = true;
                        break;
                    }
                }
                const auto lower_bound = static_cast<EdgeWeight>(sg_lb + suffix);
                if (lower_bound != kInfWeight) {
                    push_bounded_candidate(threshold_heap, unresolved_target, CandidateItem{lower_bound, obj_id, sg_id});
                    if (threshold_heap.size() >= unresolved_target) {
                        threshold = std::get<0>(threshold_heap.top());
                    }
                }
            }
            if (pruned) {
                mark_pruned(sg_id);
            }
        }
    }

    final_candidates.reserve(threshold_heap.size());
    while (!threshold_heap.empty()) {
        final_candidates.push_back(threshold_heap.top());
        threshold_heap.pop();
    }
    std::sort(final_candidates.begin(), final_candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (std::get<0>(lhs) != std::get<0>(rhs)) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        }
        return std::get<1>(lhs) < std::get<1>(rhs);
    });

    for (const auto& [lower_bound, obj_id, sg_id] : final_candidates) {
        if (exact_heap.size() >= k && lower_bound > kth_exact) {
            break;
        }
        ++exact_evaluated;
        ++exact_from_unresolved;

        const auto exact = evaluate_exact_object(obj_id, sg_id);
        if (!exact.has_value()) {
            continue;
        }

        push_bounded_candidate(exact_heap, k, ExactItem{*exact, obj_id});
        if (exact_heap.size() == k) {
            kth_exact = std::get<0>(exact_heap.top());
        }
    }

    std::vector<KnnItem> items;
    items.reserve(exact_heap.size());
    while (!exact_heap.empty()) {
        const auto [distance, obj_id] = exact_heap.top();
        exact_heap.pop();
        items.push_back({obj_id, distance});
    }
    std::sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.distance != rhs.distance) {
            return lhs.distance < rhs.distance;
        }
        return lhs.id < rhs.id;
    });
    if (exposure_events != nullptr) {
        std::unordered_map<SgId, std::size_t> overlap_by_subgraph;
        for (const auto& item : items) {
            ++overlap_by_subgraph[objects.object_subgraph(item.id)];
        }
        for (const auto& [key, idx] : exposure_index) {
            const auto sg_id = key.first;
            auto& event = exposure_events->at(idx);
            const auto overlap = overlap_by_subgraph.contains(sg_id) ? overlap_by_subgraph.at(sg_id) : 0U;
            event.eventual_result_overlap = overlap;
            event.was_needed_for_final_answer = overlap != 0U;
            event.was_only_used_for_traversal = overlap == 0U && !is_tightened(sg_id);
        }
        for (const auto& [sg_id, idx] : trigger_event_index) {
            auto& event = exposure_events->at(idx);
            event.tightened_upper_bound = is_tightened(sg_id);
            event.pruned_subgraph = is_pruned(sg_id);
        }
    }
    for (const auto sg_id : visited_subgraph_list) {
        const auto touches = get_touch_count(sg_id);
        if (touches <= 1U) {
            continue;
        }
        if (is_tightened(sg_id) || is_pruned(sg_id)) {
            num_useful_subgraph_reentries += touches - 1U;
        }
    }
    if (row_shadow_records != nullptr) {
        std::unordered_set<VertexId> settled_vertices;
        settled_vertices.reserve(settled_boundaries.size() * 2U + 1U);
        for (const auto vertex : settled_boundaries) {
            settled_vertices.insert(vertex);
        }
        for (const auto& pending : pending_rows) {
            auto& record = row_shadow_records->at(pending.index);
            for (const auto& [exit_border, cand] : pending.pushed_exits) {
                if (boundary_distance(exit_border) == cand && settled_vertices.contains(exit_border)) {
                    ++record.num_useful_pushes_from_this_row;
                    ++record.num_exits_useful;
                }
            }
        }
    }
    const auto t_after_finalize = Clock::now();
    KnnQueryResult result;
    result.items = std::move(items);
    result.visited_boundaries = touched_boundaries.size();
    result.visited_subgraphs = visited_subgraph_count;
    result.candidates_considered = candidates_considered;
    result.final_candidates = final_candidates.size();
    result.exact_evaluated = exact_evaluated;
    result.exact_from_admitted = exact_from_admitted;
    result.exact_from_unresolved = exact_from_unresolved;
    result.exact_reused_from_admitted_cache = exact_reused_from_admitted_cache;
    result.admitted_subgraphs = admitted_subgraphs;
    result.admitted_objects = admitted_objects;
    result.candidate_subgraphs = candidate_subgraph_list.size();
    result.admission_refresh_calls = admission_refresh_calls;
    result.admission_refresh_successes = admission_refresh_successes;
    result.admission_ud_updates = admission_ud_updates;
    result.admission_gate_heap_fail = admission_gate_heap_fail;
    result.admission_gate_mass_fail = admission_gate_mass_fail;
    result.admission_gate_avg_fail = admission_gate_avg_fail;
    result.admission_gate_dt_fail = admission_gate_dt_fail;
    result.admission_gate_no_admit_fail = admission_gate_no_admit_fail;
    result.admission_gate_object_fail = admission_gate_object_fail;
    result.finite_ud_candidate_subgraphs = finite_ud_candidate_subgraphs;
    result.best_admission_margin = best_admission_margin;
    result.vertex_fast_path = query_vertex.has_value();
    result.subgraph_admit = subgraph_admit;
    result.subgraph_admit_auto_disabled = subgraph_admit_auto_disabled;
    result.object_density = object_density;
    result.occupied_subgraphs = occupied_subgraphs;
    result.occupied_subgraph_ratio = occupied_subgraph_ratio;
    result.avg_objects_per_occupied_subgraph = avg_objects_per_occupied_subgraph;
    result.max_objects_in_subgraph = max_objects_in_subgraph;
    result.init_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_init - t_started).count();
    result.explore_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_explore - t_after_init).count();
    result.finalize_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_finalize - t_after_explore).count();
    result.pq_us = pq_us;
    result.membership_us = membership_us;
    result.subgraph_bookkeeping_us = subgraph_bookkeeping_us;
    result.clique_emit_us = clique_emit_us;
    result.num_clique_relax_attempts = num_clique_relax_attempts;
    result.num_successful_clique_relaxes = num_successful_clique_relaxes;
    result.num_pq_pushes_from_clique = num_pq_pushes_from_clique;
    result.num_redundant_subgraph_reentries = num_redundant_subgraph_reentries;
    result.num_useful_subgraph_reentries = num_useful_subgraph_reentries;
    result.boundary_pq_pushes = boundary_pq_pushes;
    result.boundary_pq_pops = boundary_pq_pops;
    result.row_state_pushes = row_state_pushes;
    result.row_state_pops = row_state_pops;
    result.streamed_rows_started = streamed_rows_started;
    result.streamed_row_emissions = streamed_row_emissions;
    result.streamed_rows_stopped_by_threshold = streamed_rows_stopped_by_threshold;
    result.streamed_exits_skipped_by_threshold = streamed_exits_skipped_by_threshold;
    result.peak_frontier_queue_size = peak_frontier_queue_size;
    result.safe_coverage_candidate_count = safe_coverage_candidate_count;
    result.first_k_candidates_boundary_visit_order = first_k_candidates_boundary_visit_order;
    result.root_kth_exact = root_kth_exact;
    result.first_finite_safe_coverage_radius = first_finite_safe_coverage_radius;
    result.first_finite_safe_coverage_boundary_visit_order = first_finite_safe_coverage_boundary_visit_order;
    result.final_safe_coverage_radius =
        (subgraph_admit && subgraph_safe_coverage_established) ? subgraph_safe_coverage_radius : safe_coverage_radius;
    result.safe_coverage_updates =
        (subgraph_admit && subgraph_safe_coverage_established) ? 1U : safe_coverage_updates;
    result.first_finite_tau = first_finite_tau;
    result.first_finite_tau_boundary_visit_order = first_finite_tau_boundary_visit_order;
    result.final_tau = final_tau_after_explore;
    result.tau_updates = tau_updates;
    result.would_stop_rows_by_safe_coverage = would_stop_rows_by_safe_coverage;
    result.would_skip_exits_by_safe_coverage = would_skip_exits_by_safe_coverage;
    result.would_save_pq_pushes_by_safe_coverage = would_save_pq_pushes_by_safe_coverage;
    result.final_kth_exact = kth_exact;
    result.parent_region_size = parent_shadow_size;
    if (parent_shadow_size > 0 && !subgraph_adjacency_.empty()) {
        const auto parent_of = build_parent_assignment(subgraph_adjacency_, parent_shadow_size);
        std::size_t parent_count = 0;
        for (const auto parent_id : parent_of) {
            if (parent_id != std::numeric_limits<std::size_t>::max()) {
                parent_count = std::max(parent_count, parent_id + 1U);
            }
        }
        if (parent_count != 0) {
            std::vector<std::size_t> parent_group_sizes(parent_count, 0U);
            std::vector<std::size_t> children_descended(parent_count, 0U);
            for (std::size_t sg_id = 0; sg_id < parent_of.size(); ++sg_id) {
                if (parent_of[sg_id] != std::numeric_limits<std::size_t>::max()) {
                    ++parent_group_sizes[parent_of[sg_id]];
                }
            }
            for (const auto sg_id : visited_subgraph_list) {
                const auto parent_id = parent_of[sg_id];
                if (parent_id != std::numeric_limits<std::size_t>::max()) {
                    ++children_descended[parent_id];
                }
            }
            std::size_t touched_parents = 0;
            std::size_t total_children_descended = 0;
            std::size_t total_children_capacity = 0;
            std::size_t max_children_descended = 0;
            std::size_t single_child_parents = 0;
            for (std::size_t parent_id = 0; parent_id < parent_count; ++parent_id) {
                const auto descended = children_descended[parent_id];
                if (descended == 0) {
                    continue;
                }
                ++touched_parents;
                total_children_descended += descended;
                total_children_capacity += parent_group_sizes[parent_id];
                max_children_descended = std::max(max_children_descended, descended);
                if (descended == 1) {
                    ++single_child_parents;
                }
            }
            result.parent_regions_touched = touched_parents;
            result.max_children_descended_in_parent = max_children_descended;
            result.single_child_parent_regions = single_child_parents;
            result.avg_children_descended_per_parent =
                touched_parents == 0 ? 0.0 : static_cast<double>(total_children_descended) / static_cast<double>(touched_parents);
            result.parent_descent_ratio =
                total_children_capacity == 0 ? 0.0 : static_cast<double>(total_children_descended) / static_cast<double>(total_children_capacity);
        }
    }
    result.factorized_rows_used = factorized_rows_used;
    result.factorized_hubs_used = factorized_hubs_used;
    result.factorized_exits_emitted = factorized_exits_emitted;
    result.streamed_clique = streamed_clique;
    return result;
}

KnnQueryResult SkeletonIndex::exact_knn_query(
    const QueryPoint& query,
    std::size_t k,
    const IndexedMovingObjectSet& objects
) const {
    const auto seeds = std::vector<std::pair<VertexId, EdgeWeight>>{
        {query.edge.first, query.offset},
        {query.edge.second, static_cast<EdgeWeight>(edge_weight_any(*global_, query.edge) - query.offset)},
    };
    const auto dist = dijkstra(*global_, seeds, kInfWeight);
    return compute_knn(objects, k, [&](const MovingObject& object) -> EdgeWeight {
        EdgeWeight best = kInfWeight;
        if (const auto it = dist.find(object.edge.first); it != dist.end() &&
            it->second <= kInfWeight - object.offset) {
            best = std::min(best, static_cast<EdgeWeight>(it->second + object.offset));
        }
        const auto right_cost = static_cast<EdgeWeight>(object.edge_weight - object.offset);
        if (const auto it = dist.find(object.edge.second); it != dist.end() &&
            it->second <= kInfWeight - right_cost) {
            best = std::min(best, static_cast<EdgeWeight>(it->second + right_cost));
        }
        return best;
    });
}

FrontierBoundaryStats SkeletonIndex::frontier_boundary_stats(
    const QueryPoint& query,
    EdgeWeight radius
) const {
    const auto& initial = require_initial_subgraph(query.edge);
    const auto initial_local = local_query_distances(initial, query, radius);
    const auto initial_seeds = boundary_query_seeds(initial, initial_local);

    if (skeleton_scratch_dist_.size() != static_cast<std::size_t>(max_vertex_id_) + 1U) {
        skeleton_scratch_dist_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_stamp_.assign(static_cast<std::size_t>(max_vertex_id_) + 1U, 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    if (skeleton_scratch_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(skeleton_scratch_stamp_.begin(), skeleton_scratch_stamp_.end(), 0U);
        skeleton_scratch_epoch_ = 0U;
    }
    ++skeleton_scratch_epoch_;
    const auto scratch_epoch = skeleton_scratch_epoch_;

    const auto boundary_distance = [&](VertexId v) -> EdgeWeight {
        return (v < skeleton_scratch_stamp_.size() && skeleton_scratch_stamp_[v] == scratch_epoch)
            ? skeleton_scratch_dist_[v]
            : kInfWeight;
    };

    using QueueItem = std::pair<EdgeWeight, VertexId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;
    std::vector<VertexId> touched_boundaries;
    touched_boundaries.reserve(initial_seeds.size() * 8U + 64U);

    const auto update_boundary = [&](VertexId v, EdgeWeight distance) {
        if (skeleton_scratch_stamp_[v] == scratch_epoch) {
            if (distance >= skeleton_scratch_dist_[v]) {
                return false;
            }
            skeleton_scratch_dist_[v] = distance;
        } else {
            skeleton_scratch_stamp_[v] = scratch_epoch;
            skeleton_scratch_dist_[v] = distance;
            touched_boundaries.push_back(v);
        }
        pq.push({distance, v});
        return true;
    };

    for (const auto& [seed, value] : initial_seeds) {
        update_boundary(seed, value);
    }

    while (!pq.empty()) {
        const auto [current, u] = pq.top();
        pq.pop();
        if (current != boundary_distance(u)) {
            continue;
        }
        if (current > radius) {
            continue;
        }
        const auto& row = lookup_sorted_skeleton_row(u, sorted_skeleton_row_index_, sorted_skeleton_row_storage_);
        for (const auto& [v, w] : row) {
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next > radius) {
                continue;
            }
            update_boundary(v, next);
        }
    }

    std::size_t frontier_boundaries = 0;
    std::size_t outward_edges = 0;
    for (const auto b : touched_boundaries) {
        const auto current = boundary_distance(b);
        if (current == kInfWeight || current > radius) {
            continue;
        }
        bool is_frontier = false;
        const auto& row = lookup_sorted_skeleton_row(b, sorted_skeleton_row_index_, sorted_skeleton_row_storage_);
        for (const auto& [v, w] : row) {
            if (current == kInfWeight || w == kInfWeight || current > kInfWeight - w) {
                continue;
            }
            const auto next = static_cast<EdgeWeight>(current + w);
            if (next <= radius) {
                continue;
            }
            if (boundary_distance(v) == kInfWeight) {
                is_frontier = true;
                ++outward_edges;
            }
        }
        if (is_frontier) {
            ++frontier_boundaries;
        }
    }

    FrontierBoundaryStats result;
    result.radius = radius;
    result.reached_boundaries = touched_boundaries.size();
    result.frontier_boundaries = frontier_boundaries;
    result.outward_edges = outward_edges;
    return result;
}

}  // namespace bag
