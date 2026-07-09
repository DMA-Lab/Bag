#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "graph.h"
#include "object.h"

namespace bag {

struct Coordinate {
    double x{0.0};
    double y{0.0};
};

struct UpdateOnlyBenchOptions {
    std::size_t object_count{10000};
    std::size_t change_count{1000};
    std::size_t epochs{100};
    std::uint64_t object_seed{7};
    std::uint64_t move_seed{8};
    std::size_t grid_side{256};
    std::size_t active_query_count{0};
    std::uint64_t query_seed{17};
    EdgeWeight range_radius{50000};
    bool local_move{false};
};

struct UpdateOnlyBenchResult {
    std::string baseline;
    std::size_t object_count{0};
    std::size_t change_count{0};
    std::size_t epochs{0};
    std::size_t grid_side{0};
    std::size_t active_query_count{0};
    EdgeWeight range_radius{0};
    long long total_update_us{0};
    double avg_update_us{0.0};
};

std::unordered_map<VertexId, Coordinate> load_coordinates_from_file(const std::string& file);

UpdateOnlyBenchResult benchmark_arne_update_only(
    const Graph& graph,
    const UpdateOnlyBenchOptions& options
);

UpdateOnlyBenchResult benchmark_amovnet_update_only(
    const Graph& graph,
    const std::unordered_map<VertexId, Coordinate>& coords,
    const UpdateOnlyBenchOptions& options
);

UpdateOnlyBenchResult benchmark_amovnet_query_aware_update_only(
    const Graph& graph,
    const std::unordered_map<VertexId, Coordinate>& coords,
    const UpdateOnlyBenchOptions& options
);

UpdateOnlyBenchResult benchmark_amovnet_query_indexed_update_only(
    const Graph& graph,
    const std::unordered_map<VertexId, Coordinate>& coords,
    const UpdateOnlyBenchOptions& options
);

}  // namespace bag
