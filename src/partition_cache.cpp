#include "partition_cache.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#endif

namespace bag {

namespace {

constexpr std::uint64_t kPartitionCacheMagic = 0x4241475041525431ULL; // BAGPART1
constexpr std::uint32_t kPartitionCacheVersion = 1U;
constexpr std::uint64_t kPartitionCheckpointMagic = 0x42414750434B5031ULL; // BAGPCKP1
constexpr std::uint32_t kPartitionCheckpointVersion = 1U;

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
        throw std::runtime_error("failed to write partition cache payload");
    }
}

template <typename T>
T read_pod(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("failed to read partition cache payload");
    }
    return value;
}

void write_string(std::ostream& out, const std::string& value) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!out) {
        throw std::runtime_error("failed to write partition cache string");
    }
}

std::string read_string(std::istream& in) {
    const auto size = read_pod<std::uint64_t>(in);
    std::string value(size, '\0');
    in.read(value.data(), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("failed to read partition cache string");
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
        for (const auto& [v, weight] : row) {
            write_pod<VertexId>(out, u);
            write_pod<VertexId>(out, v);
            write_pod<EdgeWeight>(out, weight);
        }
    }
}

Graph read_graph(std::istream& in) {
    Graph graph;
    const auto vertex_count = read_pod<std::uint64_t>(in);
    graph.reserve_vertices(static_cast<std::size_t>(vertex_count));
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

void write_distance_table(std::ostream& out, const DistanceTable& table) {
    const auto& rows = table.rows();
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(rows.size()));
    for (const auto& [u, row] : rows) {
        write_pod<VertexId>(out, u);
        write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(row.size()));
        for (const auto& [v, d] : row) {
            write_pod<VertexId>(out, v);
            write_pod<EdgeWeight>(out, d);
        }
    }
}

DistanceTable read_distance_table(std::istream& in) {
    DistanceTable table;
    const auto row_count = read_pod<std::uint64_t>(in);
    table.reserve_rows(static_cast<std::size_t>(row_count));
    for (std::uint64_t i = 0; i < row_count; ++i) {
        const auto u = read_pod<VertexId>(in);
        const auto entry_count = read_pod<std::uint64_t>(in);
        table.reserve_row(u, static_cast<std::size_t>(entry_count));
        for (std::uint64_t j = 0; j < entry_count; ++j) {
            const auto v = read_pod<VertexId>(in);
            const auto d = read_pod<EdgeWeight>(in);
            table.set(u, v, d);
        }
    }
    return table;
}

void skip_distance_table(std::istream& in) {
    const auto row_count = read_pod<std::uint64_t>(in);
    for (std::uint64_t i = 0; i < row_count; ++i) {
        (void)read_pod<VertexId>(in);
        const auto entry_count = read_pod<std::uint64_t>(in);
        for (std::uint64_t j = 0; j < entry_count; ++j) {
            (void)read_pod<VertexId>(in);
            (void)read_pod<EdgeWeight>(in);
        }
    }
}

template <typename T>
void write_set(std::ostream& out, const std::unordered_set<T>& values) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) {
        write_pod<T>(out, value);
    }
}

template <typename T>
std::unordered_set<T> read_set(std::istream& in) {
    std::unordered_set<T> values;
    const auto count = read_pod<std::uint64_t>(in);
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        values.insert(read_pod<T>(in));
    }
    return values;
}

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& values) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) {
        write_pod<T>(out, value);
    }
}

template <typename T>
std::vector<T> read_vector(std::istream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::vector<T> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        values.push_back(read_pod<T>(in));
    }
    return values;
}

void write_rb_map(std::ostream& out, const std::unordered_map<VertexId, HalfWeight>& rb_map) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(rb_map.size()));
    for (const auto& [v, half] : rb_map) {
        write_pod<VertexId>(out, v);
        write_pod<EdgeWeight>(out, half.whole);
        write_pod<std::uint8_t>(out, half.half ? 1U : 0U);
    }
}

std::unordered_map<VertexId, HalfWeight> read_rb_map(std::istream& in) {
    std::unordered_map<VertexId, HalfWeight> rb_map;
    const auto count = read_pod<std::uint64_t>(in);
    rb_map.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto v = read_pod<VertexId>(in);
        const auto whole = read_pod<EdgeWeight>(in);
        const auto half = read_pod<std::uint8_t>(in);
        rb_map.emplace(v, HalfWeight{whole, half != 0U});
    }
    return rb_map;
}

void write_internal_border_map(std::ostream& out, const std::unordered_map<VertexId, EdgeWeight>& values) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& [v, d] : values) {
        write_pod<VertexId>(out, v);
        write_pod<EdgeWeight>(out, d);
    }
}

std::unordered_map<VertexId, EdgeWeight> read_internal_border_map(std::istream& in) {
    std::unordered_map<VertexId, EdgeWeight> values;
    const auto count = read_pod<std::uint64_t>(in);
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto v = read_pod<VertexId>(in);
        const auto d = read_pod<EdgeWeight>(in);
        values.emplace(v, d);
    }
    return values;
}

void write_subgraph(std::ostream& out, const Subgraph& sg) {
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(sg.id));
    write_pod<VertexId>(out, sg.seed_vertex);
    write_graph(out, sg.graph);
    write_distance_table(out, sg.distance);
    write_set(out, sg.bound_vertices);
    write_set(out, sg.internal_vertices);
    write_rb_map(out, sg.rb_map);
    write_internal_border_map(out, sg.internal_to_nearest_border_dist);
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(sg.insertion_order.size()));
    for (const auto v : sg.insertion_order) {
        write_pod<VertexId>(out, v);
    }
}

Subgraph read_subgraph(std::istream& in, bool skip_distance_on_read = false) {
    Subgraph sg;
    sg.id = static_cast<SgId>(read_pod<std::uint64_t>(in));
    sg.seed_vertex = read_pod<VertexId>(in);
    sg.graph = read_graph(in);
    if (skip_distance_on_read) {
        skip_distance_table(in);
        sg.distance = DistanceTable{};
    } else {
        sg.distance = read_distance_table(in);
    }
    sg.bound_vertices = read_set<VertexId>(in);
    sg.internal_vertices = read_set<VertexId>(in);
    sg.rb_map = read_rb_map(in);
    sg.internal_to_nearest_border_dist = read_internal_border_map(in);
    const auto insertion_count = read_pod<std::uint64_t>(in);
    sg.insertion_order.reserve(static_cast<std::size_t>(insertion_count));
    for (std::uint64_t i = 0; i < insertion_count; ++i) {
        sg.insertion_order.push_back(read_pod<VertexId>(in));
    }
    return sg;
}

void write_partition_runtime_stats(std::ostream& out, const PartitionRuntimeStats& stats) {
    write_pod<long long>(out, stats.core_partition_us);
    write_pod<long long>(out, stats.shortcut_repartition_us);
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(stats.subgraphs_before_shortcut));
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(stats.subgraphs_after_shortcut));
    write_pod<std::uint64_t>(out, stats.phase1_attempts);
    write_pod<std::uint64_t>(out, stats.phase1_commits);
    write_pod<std::uint64_t>(out, stats.phase1_rejects);
    write_pod<long long>(out, stats.phase1_clone_us);
    write_pod<long long>(out, stats.phase1_finalize_us);
    write_pod<long long>(out, stats.phase1_refresh_us);
    write_pod<long long>(out, stats.phase1_extend_distance_us);
    write_pod<long long>(out, stats.phase1_br_us);
    write_pod<long long>(out, stats.phase1_nearest_border_us);
    write_pod<std::uint64_t>(out, stats.phase2_attempts);
    write_pod<std::uint64_t>(out, stats.phase2_commits);
    write_pod<std::uint64_t>(out, stats.phase2_rejects);
    write_pod<long long>(out, stats.phase2_clone_us);
    write_pod<long long>(out, stats.phase2_finalize_us);
    write_pod<long long>(out, stats.phase2_refresh_us);
    write_pod<long long>(out, stats.phase2_apsp_us);
    write_pod<long long>(out, stats.phase2_distance_update_us);
    write_pod<long long>(out, stats.phase2_br_us);
    write_pod<long long>(out, stats.phase2_nearest_border_us);
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(stats.shortcut_stats.tiny_subgraphs));
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(stats.shortcut_stats.merged_components));
    write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(stats.shortcut_stats.merged_output_subgraphs));
    write_pod<std::uint8_t>(out, stats.shortcut_stats.skipped_by_tiny_budget ? 1U : 0U);
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
        throw std::runtime_error("failed to atomically replace file: " + final_path.string());
    }
#else
    std::filesystem::rename(tmp_path, final_path);
#endif
}

template <typename WriterFn>
void write_atomic_partition_blob(const std::filesystem::path& path, WriterFn&& writer) {
    std::filesystem::create_directories(path.parent_path());
    const auto tmp_path = temp_save_path(path);
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to open partition blob for write: " + tmp_path.string());
        }
        writer(out);
        out.flush();
        if (!out) {
            throw std::runtime_error("failed to flush partition blob: " + tmp_path.string());
        }
    }
    commit_atomic_file(tmp_path, path);
}

PartitionRuntimeStats read_partition_runtime_stats(std::istream& in) {
    PartitionRuntimeStats stats;
    stats.core_partition_us = read_pod<long long>(in);
    stats.shortcut_repartition_us = read_pod<long long>(in);
    stats.subgraphs_before_shortcut = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    stats.subgraphs_after_shortcut = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    stats.phase1_attempts = read_pod<std::uint64_t>(in);
    stats.phase1_commits = read_pod<std::uint64_t>(in);
    stats.phase1_rejects = read_pod<std::uint64_t>(in);
    stats.phase1_clone_us = read_pod<long long>(in);
    stats.phase1_finalize_us = read_pod<long long>(in);
    stats.phase1_refresh_us = read_pod<long long>(in);
    stats.phase1_extend_distance_us = read_pod<long long>(in);
    stats.phase1_br_us = read_pod<long long>(in);
    stats.phase1_nearest_border_us = read_pod<long long>(in);
    stats.phase2_attempts = read_pod<std::uint64_t>(in);
    stats.phase2_commits = read_pod<std::uint64_t>(in);
    stats.phase2_rejects = read_pod<std::uint64_t>(in);
    stats.phase2_clone_us = read_pod<long long>(in);
    stats.phase2_finalize_us = read_pod<long long>(in);
    stats.phase2_refresh_us = read_pod<long long>(in);
    stats.phase2_apsp_us = read_pod<long long>(in);
    stats.phase2_distance_update_us = read_pod<long long>(in);
    stats.phase2_br_us = read_pod<long long>(in);
    stats.phase2_nearest_border_us = read_pod<long long>(in);
    stats.shortcut_stats.tiny_subgraphs = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    stats.shortcut_stats.merged_components = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    stats.shortcut_stats.merged_output_subgraphs = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    stats.shortcut_stats.skipped_by_tiny_budget = read_pod<std::uint8_t>(in) != 0U;
    return stats;
}

}  // namespace

PartitionCacheMode parse_partition_cache_mode(const std::string& value) {
    if (value == "off") {
        return PartitionCacheMode::Off;
    }
    if (value == "read") {
        return PartitionCacheMode::Read;
    }
    if (value == "write") {
        return PartitionCacheMode::Write;
    }
    if (value == "auto") {
        return PartitionCacheMode::Auto;
    }
    if (value == "refresh") {
        return PartitionCacheMode::Refresh;
    }
    throw std::runtime_error("unsupported partition cache mode: " + value);
}

std::string partition_cache_mode_to_string(PartitionCacheMode mode) {
    switch (mode) {
    case PartitionCacheMode::Off:
        return "off";
    case PartitionCacheMode::Read:
        return "read";
    case PartitionCacheMode::Write:
        return "write";
    case PartitionCacheMode::Auto:
        return "auto";
    case PartitionCacheMode::Refresh:
        return "refresh";
    }
    throw std::runtime_error("unsupported partition cache mode enum");
}

PartitionCheckpointMode parse_partition_checkpoint_mode(const std::string& value) {
    if (value == "off") {
        return PartitionCheckpointMode::Off;
    }
    if (value == "write") {
        return PartitionCheckpointMode::Write;
    }
    if (value == "resume") {
        return PartitionCheckpointMode::Resume;
    }
    if (value == "auto") {
        return PartitionCheckpointMode::Auto;
    }
    throw std::runtime_error("unsupported partition checkpoint mode: " + value);
}

std::string partition_checkpoint_mode_to_string(PartitionCheckpointMode mode) {
    switch (mode) {
    case PartitionCheckpointMode::Off:
        return "off";
    case PartitionCheckpointMode::Write:
        return "write";
    case PartitionCheckpointMode::Resume:
        return "resume";
    case PartitionCheckpointMode::Auto:
        return "auto";
    }
    throw std::runtime_error("unsupported partition checkpoint mode enum");
}

std::string make_partition_cache_key(
    const std::string& graph_path,
    const Graph& graph,
    const PartitionOptions& options
) {
    auto graph_hash = fnv1a64_seed();
    const auto vertices = graph.vertices();
    for (const auto v : vertices) {
        graph_hash = fnv1a64_append(graph_hash, std::to_string(v));
        graph_hash = fnv1a64_append(graph_hash, "|");
    }
    auto directed = graph.directed_edges();
    std::sort(directed.begin(), directed.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first.first != rhs.first.first) {
            return lhs.first.first < rhs.first.first;
        }
        if (lhs.first.second != rhs.first.second) {
            return lhs.first.second < rhs.first.second;
        }
        return lhs.second < rhs.second;
    });
    for (const auto& [edge, weight] : directed) {
        graph_hash = fnv1a64_append(graph_hash, std::to_string(edge.first));
        graph_hash = fnv1a64_append(graph_hash, ">");
        graph_hash = fnv1a64_append(graph_hash, std::to_string(edge.second));
        graph_hash = fnv1a64_append(graph_hash, "@");
        graph_hash = fnv1a64_append(graph_hash, std::to_string(weight));
        graph_hash = fnv1a64_append(graph_hash, "|");
    }

    auto hash = fnv1a64_seed();
    std::ostringstream oss;
    oss
        << "bag-partition-cache-v1"
        << "|path=" << graph_path
        << "|vertices=" << graph.size()
        << "|edges=" << graph.edge_count()
        << "|graph_hash=" << hex_u64(graph_hash)
        << "|theta=" << options.theta
        << "|seed=" << options.partition_seed
        << "|paper_strict=" << (options.paper_strict_mode ? 1 : 0)
        << "|adaptive_z=" << (options.adaptive_z ? 1 : 0)
        << "|adaptive_alpha=" << options.adaptive_alpha
        << "|border_min=" << (options.border_minimization ? 1 : 0)
        << "|shortcut=" << (options.shortcut_repartition ? 1 : 0)
        << "|shortcut_small_upper_bound=" << options.shortcut_small_upper_bound
        << "|shortcut_k_neighbors=" << options.shortcut_k_neighbors
        << "|shortcut_radius_limit=" << options.shortcut_radius_limit
        << "|shortcut_max_tiny_subgraphs=" << options.shortcut_max_tiny_subgraphs
        << "|phase1_inplace=" << (options.phase1_inplace ? 1 : 0)
        << "|defer_nearest_border_fill=" << (options.defer_nearest_border_fill ? 1 : 0)
        << "|skip_phase1_br_check=" << (options.skip_phase1_br_check ? 1 : 0)
        << "|phase2_incremental_distance_update=" << (options.phase2_incremental_distance_update ? 1 : 0);
    const auto key_material = oss.str();
    hash = fnv1a64_append(hash, key_material);
    return hex_u64(hash);
}

std::filesystem::path partition_cache_file_path(
    const PartitionCacheConfig& config,
    const std::string& cache_key
) {
    return config.directory / (cache_key + ".bpart");
}

bool partition_cache_should_try_read(PartitionCacheMode mode) {
    return mode == PartitionCacheMode::Read || mode == PartitionCacheMode::Auto;
}

bool partition_cache_should_write(PartitionCacheMode mode) {
    return mode == PartitionCacheMode::Write ||
           mode == PartitionCacheMode::Auto ||
           mode == PartitionCacheMode::Refresh;
}

bool load_partition_cache(
    const std::filesystem::path& path,
    PartitionCacheEntry& entry
) {
    return load_partition_cache(path, entry, false);
}

bool load_partition_cache(
    const std::filesystem::path& path,
    PartitionCacheEntry& entry,
    bool skip_distance_on_read
) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    const auto magic = read_pod<std::uint64_t>(in);
    const auto version = read_pod<std::uint32_t>(in);
    if (magic != kPartitionCacheMagic || version != kPartitionCacheVersion) {
        throw std::runtime_error("partition cache header mismatch: " + path.string());
    }
    entry.key = read_string(in);
    entry.stats = read_partition_runtime_stats(in);
    const auto subgraph_count = read_pod<std::uint64_t>(in);
    entry.subgraphs.clear();
    entry.subgraphs.reserve(static_cast<std::size_t>(subgraph_count));
    for (std::uint64_t i = 0; i < subgraph_count; ++i) {
        entry.subgraphs.push_back(read_subgraph(in, skip_distance_on_read));
    }
    return true;
}

void save_partition_cache(
    const std::filesystem::path& path,
    const PartitionCacheEntry& entry
) {
    save_partition_cache(path, entry.key, entry.stats, entry.subgraphs);
}

void save_partition_cache(
    const std::filesystem::path& path,
    const std::string& key,
    const PartitionRuntimeStats& stats,
    const std::vector<Subgraph>& subgraphs
) {
    write_atomic_partition_blob(path, [&](std::ostream& out) {
        write_pod<std::uint64_t>(out, kPartitionCacheMagic);
        write_pod<std::uint32_t>(out, kPartitionCacheVersion);
        write_string(out, key);
        write_partition_runtime_stats(out, stats);
        write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(subgraphs.size()));
        for (const auto& sg : subgraphs) {
            write_subgraph(out, sg);
        }
    });
}

bool load_partition_checkpoint(
    const std::filesystem::path& path,
    PartitionCheckpointEntry& entry
) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    const auto magic = read_pod<std::uint64_t>(in);
    const auto version = read_pod<std::uint32_t>(in);
    if (magic != kPartitionCheckpointMagic || version != kPartitionCheckpointVersion) {
        throw std::runtime_error("partition checkpoint header mismatch: " + path.string());
    }
    entry.key = read_string(in);
    entry.stats = read_partition_runtime_stats(in);
    entry.pending_seed_vertices = read_vector<VertexId>(in);
    entry.no_progress_seeds = read_vector<VertexId>(in);
    const auto subgraph_count = read_pod<std::uint64_t>(in);
    entry.subgraphs.clear();
    entry.subgraphs.reserve(static_cast<std::size_t>(subgraph_count));
    for (std::uint64_t i = 0; i < subgraph_count; ++i) {
        entry.subgraphs.push_back(read_subgraph(in));
    }
    return true;
}

void save_partition_checkpoint(
    const std::filesystem::path& path,
    const PartitionCheckpointEntry& entry
) {
    save_partition_checkpoint(
        path,
        entry.key,
        entry.stats,
        entry.pending_seed_vertices,
        entry.no_progress_seeds,
        entry.subgraphs
    );
}

void save_partition_checkpoint(
    const std::filesystem::path& path,
    const std::string& key,
    const PartitionRuntimeStats& stats,
    const std::vector<VertexId>& pending_seed_vertices,
    const std::vector<VertexId>& no_progress_seeds,
    const std::vector<Subgraph>& subgraphs
) {
    write_atomic_partition_blob(path, [&](std::ostream& out) {
        write_pod<std::uint64_t>(out, kPartitionCheckpointMagic);
        write_pod<std::uint32_t>(out, kPartitionCheckpointVersion);
        write_string(out, key);
        write_partition_runtime_stats(out, stats);
        write_vector(out, pending_seed_vertices);
        write_vector(out, no_progress_seeds);
        write_pod<std::uint64_t>(out, static_cast<std::uint64_t>(subgraphs.size()));
        for (const auto& sg : subgraphs) {
            write_subgraph(out, sg);
        }
    });
}

}  // namespace bag
