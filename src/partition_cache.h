#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "partition.h"

namespace bag {

enum class PartitionCacheMode {
    Off,
    Read,
    Write,
    Auto,
    Refresh,
};

enum class PartitionCheckpointMode {
    Off,
    Write,
    Resume,
    Auto,
};

struct PartitionCacheConfig {
    PartitionCacheMode mode{PartitionCacheMode::Off};
    std::filesystem::path directory;
    bool skip_distance_on_read{false};
};

struct PartitionCacheState {
    bool enabled{false};
    bool hit{false};
    PartitionCacheMode mode{PartitionCacheMode::Off};
    std::string key;
    std::filesystem::path path;
    long long load_us{0};
    long long save_us{0};
    long long compute_us{0};
    bool deep_audit_requested{false};
    bool deep_audit_ok{false};
    long long deep_audit_us{0};
};

struct PartitionCacheEntry {
    std::string key;
    PartitionRuntimeStats stats;
    std::vector<Subgraph> subgraphs;
};

struct PartitionCheckpointEntry {
    std::string key;
    PartitionRuntimeStats stats;
    std::vector<VertexId> pending_seed_vertices;
    std::vector<VertexId> no_progress_seeds;
    std::vector<Subgraph> subgraphs;
};

PartitionCacheMode parse_partition_cache_mode(const std::string& value);
std::string partition_cache_mode_to_string(PartitionCacheMode mode);
PartitionCheckpointMode parse_partition_checkpoint_mode(const std::string& value);
std::string partition_checkpoint_mode_to_string(PartitionCheckpointMode mode);
std::string make_partition_cache_key(
    const std::string& graph_path,
    const Graph& graph,
    const PartitionOptions& options
);
std::filesystem::path partition_cache_file_path(
    const PartitionCacheConfig& config,
    const std::string& cache_key
);
bool partition_cache_should_try_read(PartitionCacheMode mode);
bool partition_cache_should_write(PartitionCacheMode mode);
bool load_partition_cache(
    const std::filesystem::path& path,
    PartitionCacheEntry& entry
);
bool load_partition_cache(
    const std::filesystem::path& path,
    PartitionCacheEntry& entry,
    bool skip_distance_on_read
);
void save_partition_cache(
    const std::filesystem::path& path,
    const PartitionCacheEntry& entry
);
void save_partition_cache(
    const std::filesystem::path& path,
    const std::string& key,
    const PartitionRuntimeStats& stats,
    const std::vector<Subgraph>& subgraphs
);
bool load_partition_checkpoint(
    const std::filesystem::path& path,
    PartitionCheckpointEntry& entry
);
void save_partition_checkpoint(
    const std::filesystem::path& path,
    const PartitionCheckpointEntry& entry
);
void save_partition_checkpoint(
    const std::filesystem::path& path,
    const std::string& key,
    const PartitionRuntimeStats& stats,
    const std::vector<VertexId>& pending_seed_vertices,
    const std::vector<VertexId>& no_progress_seeds,
    const std::vector<Subgraph>& subgraphs
);

}  // namespace bag
