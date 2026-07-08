#pragma once

#include <filesystem>
#include <vector>

#include "graph.h"
#include "partition.h"

namespace bag {

struct MetisImportOptions {
    std::filesystem::path assignment_csv;
    bool rb_only_mode{true};
};

std::vector<Subgraph> build_subgraphs_from_metis_assignment(
    const Graph& global,
    const MetisImportOptions& options
);

}  // namespace bag
