#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_set>

#include "diagnostics.h"
#include "factorized_transfer.h"
#include "index.h"
#include "loader.h"
#include "metis_partition.h"
#include "object.h"
#include "partition.h"
#include "partition_cache.h"
#include "utils.h"

namespace {

using bag::EdgeWeight;
using bag::IndexedMovingObjectSet;
using bag::MovingObjectSet;
using bag::QueryPoint;
using bag::SkeletonIndex;
using bag::VfipPartition;

struct PartitionSummary {
    std::size_t subgraphs{0};
    std::size_t graph_vertices{0};
    std::size_t graph_edges{0};
    std::size_t skeleton_vertices{0};
    std::size_t skeleton_edges{0};
    std::size_t total_boundary_vertices{0};
    std::size_t total_internal_vertices{0};
    std::size_t max_boundary_vertices{0};
    double avg_boundary_per_subgraph{0.0};
    double skeleton_avg_degree{0.0};
    bool br_property_ok{true};
    std::size_t br_violating_subgraphs{0};
    std::size_t br_violating_boundaries{0};
};

struct HierarchyLiteSummary {
    std::size_t group_size{0};
    std::size_t parent_regions{0};
    double avg_children_per_parent{0.0};
    std::size_t max_children_per_parent{0};
    std::size_t upper_unique_boundary_vertices{0};
    std::size_t upper_total_boundary_vertices{0};
    std::size_t upper_parent_graph_edges{0};
    std::size_t upper_estimated_clique_edges{0};
    double upper_unique_boundary_ratio_vs_l1{0.0};
    double upper_total_boundary_ratio_vs_l1_total{0.0};
    double upper_estimated_clique_ratio_vs_l1{0.0};
};

struct ExactCoverageSummary {
    std::size_t result_subgraphs{0};
    std::size_t fully_admitted_subgraphs{0};
    std::size_t result_objects{0};
    std::size_t fully_admitted_objects{0};
};

bag::Graph build_skeleton_graph_from_subgraphs(const std::vector<bag::Subgraph>& subgraphs) {
    bag::Graph skeleton;
    for (const auto& sg : subgraphs) {
        std::vector<bag::VertexId> boundaries(sg.bound_vertices.begin(), sg.bound_vertices.end());
        std::sort(boundaries.begin(), boundaries.end());
        for (const auto b : boundaries) {
            skeleton.insert(b);
        }
        for (std::size_t i = 0; i < boundaries.size(); ++i) {
            for (std::size_t j = i + 1; j < boundaries.size(); ++j) {
                const auto distance = sg.distance.get_or_inf(boundaries[i], boundaries[j]);
                if (distance != bag::kInfWeight) {
                    skeleton.set_min_undirected_edge(boundaries[i], boundaries[j], distance);
                }
            }
        }
    }
    return skeleton;
}

PartitionSummary summarize_partition(
    const bag::Graph& graph,
    const bag::Graph& skeleton,
    const std::vector<bag::Subgraph>& subgraphs
) {
    PartitionSummary summary;
    summary.subgraphs = subgraphs.size();
    summary.graph_vertices = graph.size();
    summary.graph_edges = graph.edge_count();
    summary.skeleton_vertices = skeleton.size();
    summary.skeleton_edges = skeleton.undirected_edges().size();

    for (const auto& sg : subgraphs) {
        summary.total_boundary_vertices += sg.bound_vertices.size();
        summary.total_internal_vertices += sg.internal_vertices.size();
        summary.max_boundary_vertices = std::max(summary.max_boundary_vertices, sg.bound_vertices.size());

        bool subgraph_ok = true;
        const auto sg_edges = sg.graph.undirected_edges();
        for (const auto b : sg.bound_vertices) {
            bag::HalfWeight rb{0, false};
            for (const auto other : sg.bound_vertices) {
                rb = std::max(rb, bag::HalfWeight{sg.distance.get_or_inf(b, other), false});
            }

            bag::HalfWeight ib{0, false};
            for (const auto v : sg.internal_vertices) {
                ib = std::max(ib, bag::HalfWeight{sg.distance.get_or_inf(b, v), false});
            }
            for (const auto& [edge, w] : sg_edges) {
                const auto lhs = sg.distance.get_or_inf(b, edge.first);
                const auto rhs = sg.distance.get_or_inf(b, edge.second);
                if (lhs == bag::kInfWeight || rhs == bag::kInfWeight) {
                    continue;
                }
                ib = std::max(ib, bag::distal_point_distance(lhs, rhs, w));
            }

            if (rb < ib) {
                ++summary.br_violating_boundaries;
                subgraph_ok = false;
            }
        }
        if (!subgraph_ok) {
            ++summary.br_violating_subgraphs;
        }
    }

    summary.avg_boundary_per_subgraph =
        subgraphs.empty()
            ? 0.0
            : static_cast<double>(summary.total_boundary_vertices) / static_cast<double>(subgraphs.size());
    summary.skeleton_avg_degree =
        summary.skeleton_vertices == 0
            ? 0.0
            : (2.0 * static_cast<double>(summary.skeleton_edges)) / static_cast<double>(summary.skeleton_vertices);
    summary.br_property_ok = summary.br_violating_boundaries == 0;
    return summary;
}

std::vector<std::size_t> parse_size_list(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::vector<std::size_t>& fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    std::vector<std::size_t> result;
    std::stringstream ss(it->second);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            result.push_back(static_cast<std::size_t>(std::stoull(token)));
        }
    }
    return result.empty() ? fallback : result;
}

std::vector<std::vector<bag::SgId>> build_subgraph_adjacency(const std::vector<bag::Subgraph>& subgraphs) {
    std::unordered_map<bag::VertexId, std::vector<bag::SgId>> border_to_sgs;
    border_to_sgs.reserve(subgraphs.size() * 4U + 1U);
    for (const auto& sg : subgraphs) {
        for (const auto b : sg.bound_vertices) {
            border_to_sgs[b].push_back(sg.id);
        }
    }

    std::vector<std::unordered_set<bag::SgId>> adjacency_sets(subgraphs.size());
    for (const auto& [_, ids] : border_to_sgs) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            for (std::size_t j = i + 1; j < ids.size(); ++j) {
                adjacency_sets[ids[i]].insert(ids[j]);
                adjacency_sets[ids[j]].insert(ids[i]);
            }
        }
    }

    std::vector<std::vector<bag::SgId>> adjacency(subgraphs.size());
    for (std::size_t i = 0; i < adjacency_sets.size(); ++i) {
        adjacency[i].assign(adjacency_sets[i].begin(), adjacency_sets[i].end());
        std::sort(adjacency[i].begin(), adjacency[i].end());
    }
    return adjacency;
}

HierarchyLiteSummary analyze_hierarchy_lite(
    const std::vector<bag::Subgraph>& subgraphs,
    const PartitionSummary& level1_summary,
    std::size_t group_size
) {
    HierarchyLiteSummary summary;
    summary.group_size = group_size;
    if (subgraphs.empty() || group_size == 0) {
        return summary;
    }

    const auto adjacency = build_subgraph_adjacency(subgraphs);
    std::vector<int> parent_of(subgraphs.size(), -1);
    std::vector<std::vector<bag::SgId>> groups;
    groups.reserve((subgraphs.size() + group_size - 1U) / group_size);

    for (std::size_t seed = 0; seed < subgraphs.size(); ++seed) {
        if (parent_of[seed] != -1) {
            continue;
        }
        const int parent_id = static_cast<int>(groups.size());
        groups.push_back({});
        std::queue<bag::SgId> queue;
        queue.push(static_cast<bag::SgId>(seed));
        parent_of[seed] = parent_id;

        while (!queue.empty() && groups[parent_id].size() < group_size) {
            const auto current = queue.front();
            queue.pop();
            groups[parent_id].push_back(current);
            for (const auto neighbor : adjacency[current]) {
                if (parent_of[neighbor] == -1 &&
                    groups[parent_id].size() + queue.size() < group_size) {
                    parent_of[neighbor] = parent_id;
                    queue.push(neighbor);
                }
            }
        }
    }

    for (std::size_t i = 0; i < parent_of.size(); ++i) {
        if (parent_of[i] == -1) {
            const int parent_id = static_cast<int>(groups.size());
            groups.push_back({static_cast<bag::SgId>(i)});
            parent_of[i] = parent_id;
        }
    }

    summary.parent_regions = groups.size();
    std::size_t children_sum = 0;
    for (const auto& group : groups) {
        children_sum += group.size();
        summary.max_children_per_parent = std::max(summary.max_children_per_parent, group.size());
    }
    summary.avg_children_per_parent =
        groups.empty() ? 0.0 : static_cast<double>(children_sum) / static_cast<double>(groups.size());

    std::unordered_map<bag::VertexId, std::vector<bag::SgId>> border_to_sgs;
    border_to_sgs.reserve(subgraphs.size() * 4U + 1U);
    for (const auto& sg : subgraphs) {
        for (const auto b : sg.bound_vertices) {
            border_to_sgs[b].push_back(sg.id);
        }
    }

    std::vector<std::unordered_set<bag::VertexId>> parent_exposed_boundaries(groups.size());
    for (const auto& [border, ids] : border_to_sgs) {
        std::unordered_set<int> parents;
        for (const auto sg_id : ids) {
            parents.insert(parent_of[sg_id]);
        }
        if (parents.size() <= 1U) {
            continue;
        }
        ++summary.upper_unique_boundary_vertices;
        summary.upper_total_boundary_vertices += parents.size();
        for (const auto parent_id : parents) {
            parent_exposed_boundaries[static_cast<std::size_t>(parent_id)].insert(border);
        }
    }

    std::unordered_set<std::uint64_t> parent_edges;
    for (std::size_t u = 0; u < adjacency.size(); ++u) {
        for (const auto v : adjacency[u]) {
            const auto pu = static_cast<std::size_t>(parent_of[u]);
            const auto pv = static_cast<std::size_t>(parent_of[v]);
            if (pu == pv) {
                continue;
            }
            const auto edge = bag::ordered_edge(
                static_cast<bag::VertexId>(pu),
                static_cast<bag::VertexId>(pv)
            );
            parent_edges.insert(bag::pack_pair(edge.first, edge.second));
        }
    }
    summary.upper_parent_graph_edges = parent_edges.size();

    for (const auto& exposed : parent_exposed_boundaries) {
        const auto b = exposed.size();
        if (b >= 2U) {
            summary.upper_estimated_clique_edges += (b * (b - 1U)) / 2U;
        }
    }

    summary.upper_unique_boundary_ratio_vs_l1 =
        level1_summary.skeleton_vertices == 0
            ? 0.0
            : static_cast<double>(summary.upper_unique_boundary_vertices) /
                  static_cast<double>(level1_summary.skeleton_vertices);
    summary.upper_total_boundary_ratio_vs_l1_total =
        level1_summary.total_boundary_vertices == 0
            ? 0.0
            : static_cast<double>(summary.upper_total_boundary_vertices) /
                  static_cast<double>(level1_summary.total_boundary_vertices);

    std::size_t level1_clique_edges = 0;
    for (const auto& sg : subgraphs) {
        const auto b = sg.bound_vertices.size();
        if (b >= 2U) {
            level1_clique_edges += (b * (b - 1U)) / 2U;
        }
    }
    summary.upper_estimated_clique_ratio_vs_l1 =
        level1_clique_edges == 0
            ? 0.0
            : static_cast<double>(summary.upper_estimated_clique_edges) /
                  static_cast<double>(level1_clique_edges);

    return summary;
}

bag::FcRule parse_fc_rule(const std::unordered_map<std::string, std::string>& args) {
    const auto it = args.find("fc-rule");
    if (it == args.end() || it->second == "paper") {
        return bag::FcRule::PaperStrict;
    }
    if (it->second == "ub") {
        return bag::FcRule::UpperBoundCandidate;
    }
    if (it->second == "all") {
        return bag::FcRule::AllBordersVisited;
    }
    if (it->second == "none") {
        return bag::FcRule::Disabled;
    }
    throw std::runtime_error("unsupported --fc-rule, expected paper|ub|all|none");
}

std::string require_arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        throw std::runtime_error("missing required argument --" + key);
    }
    return it->second;
}

std::size_t require_usize(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key
) {
    return static_cast<std::size_t>(std::stoull(require_arg(args, key)));
}

std::size_t optional_usize(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    std::size_t fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    return static_cast<std::size_t>(std::stoull(it->second));
}

bag::VertexId require_vertex(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key
) {
    return static_cast<bag::VertexId>(std::stoul(require_arg(args, key)));
}

EdgeWeight require_weight(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key
) {
    return static_cast<EdgeWeight>(std::stoul(require_arg(args, key)));
}

bool optional_bool(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    bool fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    return it->second == "1" || it->second == "true";
}

double optional_double(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    double fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    return std::stod(it->second);
}

void set_default_arg(
    std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::string& value
) {
    if (!args.contains(key)) {
        args[key] = value;
    }
}

std::filesystem::path resolve_first_existing_path(
    const std::vector<std::filesystem::path>& candidates
) {
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("failed to resolve built-in default experiment paths from current working directory");
}

void apply_default_probe_profile(std::unordered_map<std::string, std::string>& args) {
    const auto graph_path = resolve_first_existing_path({
        "dataset/USA-road-d.NY.gr",
        "../dataset/USA-road-d.NY.gr",
        "../../dataset/USA-road-d.NY.gr",
    });
    const auto assignment_path = resolve_first_existing_path({
        "report/metis_ny_theta50.csv",
        "../report/metis_ny_theta50.csv",
        "../../report/metis_ny_theta50.csv",
    });

    set_default_arg(args, "path", graph_path.string());
    set_default_arg(args, "metis-assignment", assignment_path.string());
    set_default_arg(args, "theta", "50");
    set_default_arg(args, "query-u", "2");
    set_default_arg(args, "query-v", "13");
    set_default_arg(args, "offset", "0");
    set_default_arg(args, "objects", "1000");
    set_default_arg(args, "object-seed", "7");
    set_default_arg(args, "radius", "50000");
    set_default_arg(args, "k", "10");
    set_default_arg(args, "repeat", "5");
    set_default_arg(args, "fc-rule", "ub");
}

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  bag_cpp default\n"
        << "  bag_cpp partition --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--metis-assignment <file>]"
           " [--shortcut-repartition true|false]\n"
        << "  bag_cpp range --path <graph> --theta <z> [--partition-seed <v>] --query-u <u> --query-v <v> --offset <d>"
           " --radius <r> --objects <n> [--seed <s>] [--repeat <n>] [--verify true|false] [--fc-rule paper|ub|all|none]"
           " [--range-row-truncation true|false] [--factorized-transfer true|false] [--factorized-arc-threshold <x>] [--factorized-border-threshold <n>] [--gstar-shortcuts true|false] [--gstar-leaf-hops <n>] [--gstar-shortcuts-per-subgraph <n>] [--compare-baseline true|false]"
           " [--metis-assignment <file>]"
           " [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false]\n"
        << "  bag_cpp knn --path <graph> --theta <z> [--partition-seed <v>] --query-u <u> --query-v <v> --offset <d>"
           " --k <k> --objects <n> [--seed <s>] [--repeat <n>] [--verify true|false]"
           " [--knn-streamed-clique true|false] [--knn-safe-coverage-shadow true|false] [--knn-parent-shadow-size <n>] [--factorized-transfer true|false] [--factorized-arc-threshold <x>] [--factorized-border-threshold <n>] [--gstar-shortcuts true|false] [--gstar-leaf-hops <n>] [--gstar-shortcuts-per-subgraph <n>] [--compare-baseline true|false]"
           " [--metis-assignment <file>]"
           " [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false]\n"
        << "  bag_cpp probe --path <graph> --theta <z> [--partition-seed <v>] --query-u <u> --query-v <v> --offset <d>"
           " --radius <r> --k <k> --objects <n> [--seed <s>] [--repeat <n>] [--verify true|false]"
           " [--fc-rule paper|ub|all|none] [--range-row-truncation true|false]"
           " [--knn-streamed-clique true|false] [--knn-safe-coverage-shadow true|false] [--knn-parent-shadow-size <n>]"
           " [--factorized-transfer true|false] [--factorized-arc-threshold <x>] [--factorized-border-threshold <n>] [--gstar-shortcuts true|false] [--gstar-leaf-hops <n>] [--gstar-shortcuts-per-subgraph <n>]"
           " [--metis-assignment <file>] [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>]"
           " [--border-min true|false] [--shortcut-repartition true|false]\n"
        << "  bag_cpp frontier --path <graph> --theta <z> [--partition-seed <v>] --query-u <u> --query-v <v> --offset <d>"
           " [--radii <r1,r2,...>] [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>]"
           " [--border-min true|false] [--shortcut-repartition true|false]\n"
        << "  bag_cpp sample-queries --path <graph> --count <n> [--seed <s>]\n"
        << "  bag_cpp batch --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] [--metis-assignment <file>]"
           " --query-count <n> [--query-seed <s>] --objects <n> [--object-seed <s>]"
           " [--range-radius <r>] [--knn-k <k>] [--fc-rule paper|ub|all|none]"
           " [--factorized-transfer true|false] [--factorized-arc-threshold <x>]"
           " [--factorized-border-threshold <n>] [--gstar-shortcuts true|false] [--gstar-leaf-hops <n>] [--gstar-shortcuts-per-subgraph <n>]\n"
        << "  bag_cpp query-sweep --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] [--metis-assignment <file>]"
           " --query-count <n> [--query-seed <s>] --objects <n> [--object-seed <s>]"
           " --radii <r1,r2,...> --ks <k1,k2,...> [--fc-rule paper|ub|all|none]"
           " [--knn-pc-dijkstra-mode subgraph|global] [--verify true|false] [--output-csv <file>]\n"
        << "  bag_cpp coverage-sweep --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] [--metis-assignment <file>]"
           " --mode range|knn --query-count <n> [--query-seed <s>] --objects <n> [--object-seed <s>]"
           " [--object-layout random|spread-subgraphs] [--radii <r1,r2,...>] [--ks <k1,k2,...>]"
           " [--output-csv <file>]\n"
        << "  bag_cpp coverage-compare --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] --metis-assignment <file> --kahip-assignment <file>"
           " --mode range|knn --query-count <n> [--query-seed <s>] --objects <n> [--object-seed <s>]"
           " [--radii <r1,r2,...>] [--ks <k1,k2,...>] [--output-csv <file>]\n"
        << "  bag_cpp maint-bench --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
            " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
            " [--shortcut-repartition true|false] [--metis-assignment <file>] [--mode range|knn]"
            " [--objects <n>] [--change-count <n>] [--car-percent <x>] [--change-percent <x>]"
            " --query-per-update <n> [--epochs <n>] [--query-seed <s>] [--object-seed <s>]"
            " [--range-radius <r>] [--fc-rule paper|ub|all|none] [--knn-k <k>]"
            " [--object-layout random|spread-subgraphs]\n"
        << "  bag_cpp weight-audit --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
            " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
            " [--shortcut-repartition true|false] [--metis-assignment <file>]"
            " [--weight-change-fraction <x>] [--weight-scale <x>] [--weight-direction increase|decrease]"
            " [--weight-seed <s>]\n"
        << "  bag_cpp hierarchy-lite --path <graph> --theta <z> [--partition-seed <v>]"
            " [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>]"
            " [--border-min true|false] [--shortcut-repartition true|false]"
           " [--metis-assignment <file>] [--hier-group-sizes <g1,g2,...>]\n"
        << "  bag_cpp factorized-scan --path <graph> --theta <z> [--partition-seed <v>] [--paper-strict true|false]"
           " [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] [--output-csv <file>]\n"
        << "  bag_cpp diagnostics --path <graph> --theta <z> --output-dir <dir> [--partition-seed <v>]"
           " [--paper-strict true|false] [--adaptive-z true|false] [--alpha <x>] [--border-min true|false]"
           " [--shortcut-repartition true|false] [--query-count <n>] [--objects <n>] [--query-seed <s>]"
           " [--object-seed <s>] [--range-radius <r>] [--knn-k <k>] [--fc-rule paper|ub|all|none]"
           " [--exact-demotion-limit <n>] [--growth-trace true|false]\n";
}

QueryPoint load_query_point(const std::unordered_map<std::string, std::string>& args) {
    return QueryPoint{
        bag::ordered_edge(require_vertex(args, "query-u"), require_vertex(args, "query-v")),
        require_weight(args, "offset"),
    };
}

IndexedMovingObjectSet build_objects(
    const bag::Graph& graph,
    const SkeletonIndex& index,
    std::size_t object_count,
    std::uint64_t seed,
    const std::string& layout = "random"
) {
    if (layout == "spread-subgraphs") {
        struct EdgePlacement {
            bag::Edge edge{};
            EdgeWeight edge_weight{0};
            bag::SgId sg_id{0};
        };

        std::vector<EdgePlacement> placements;
        placements.reserve(index.subgraphs().size());
        const auto& edge_to_subgraph = index.edge_to_subgraph();
        for (const auto& sg : index.subgraphs()) {
            std::optional<std::pair<bag::Edge, EdgeWeight>> chosen;
            for (const auto& [edge, weight] : sg.graph.undirected_edges()) {
                if (weight <= 1) {
                    continue;
                }
                const auto ordered = bag::ordered_edge(edge.first, edge.second);
                const auto it = edge_to_subgraph.find(ordered);
                if (it == edge_to_subgraph.end() || it->second != sg.id) {
                    continue;
                }
                const bool internal_edge =
                    sg.internal_vertices.contains(ordered.first) &&
                    sg.internal_vertices.contains(ordered.second);
                if (internal_edge) {
                    chosen = std::make_pair(ordered, weight);
                    break;
                }
                if (!chosen.has_value()) {
                    chosen = std::make_pair(ordered, weight);
                }
            }
            if (chosen.has_value()) {
                placements.push_back(EdgePlacement{chosen->first, chosen->second, sg.id});
            }
        }

        if (placements.empty()) {
            throw std::runtime_error("no valid per-subgraph object placements found");
        }

        std::mt19937_64 rng(seed);
        std::shuffle(placements.begin(), placements.end(), rng);

        bag::MovingObjectSet set;
        bag::ObjId next_id = 0;
        std::size_t cursor = 0;
        while (next_id < object_count) {
            if (cursor == placements.size()) {
                std::shuffle(placements.begin(), placements.end(), rng);
                cursor = 0;
            }
            const auto& placement = placements[cursor++];
            std::uniform_int_distribution<EdgeWeight> offset_dist(1, placement.edge_weight - 1);
            set.push(bag::MovingObject{
                next_id++,
                placement.edge,
                offset_dist(rng),
                placement.edge_weight,
            });
        }
        return IndexedMovingObjectSet::from_partition(set.objects(), index.edge_to_subgraph(), index.subgraphs());
    }

    auto set = MovingObjectSet::random_uniform(graph, object_count, seed);
    return IndexedMovingObjectSet::from_partition(set.objects(), index.edge_to_subgraph(), index.subgraphs());
}

IndexedMovingObjectSet build_objects(
    const bag::Graph& graph,
    const SkeletonIndex& index,
    const std::unordered_map<std::string, std::string>& args
) {
    const auto seed_it = args.find("seed");
    const std::uint64_t seed =
        (seed_it == args.end()) ? 7ULL : static_cast<std::uint64_t>(std::stoull(seed_it->second));
    const auto layout_it = args.find("object-layout");
    const std::string layout = (layout_it == args.end()) ? "random" : layout_it->second;
    return build_objects(graph, index, require_usize(args, "objects"), seed, layout);
}

std::vector<QueryPoint> build_query_workload(
    const bag::Graph& graph,
    std::size_t query_count,
    std::uint64_t seed
) {
    std::vector<QueryPoint> queries;
    queries.reserve(query_count);
    const auto samples = MovingObjectSet::random_uniform(graph, query_count, seed);
    for (const auto& object : samples.objects()) {
        queries.push_back(QueryPoint{object.edge, object.offset});
    }
    return queries;
}

ExactCoverageSummary summarize_exact_subgraph_coverage(
    const std::vector<bag::ObjId>& result_ids,
    const IndexedMovingObjectSet& objects
) {
    ExactCoverageSummary summary;
    summary.result_objects = result_ids.size();
    if (result_ids.empty()) {
        return summary;
    }

    std::unordered_map<bag::SgId, std::size_t> subgraph_hits;
    subgraph_hits.reserve(result_ids.size());
    for (const auto obj_id : result_ids) {
        ++subgraph_hits[objects.object_subgraph(obj_id)];
    }

    summary.result_subgraphs = subgraph_hits.size();
    for (const auto& [sg_id, hit_count] : subgraph_hits) {
        if (hit_count == objects.objects_in(sg_id).size()) {
            ++summary.fully_admitted_subgraphs;
            summary.fully_admitted_objects += hit_count;
        }
    }
    return summary;
}

std::vector<bag::ObjId> extract_knn_ids(const bag::KnnQueryResult& result) {
    std::vector<bag::ObjId> ids;
    ids.reserve(result.items.size());
    for (const auto& item : result.items) {
        ids.push_back(item.id);
    }
    return ids;
}

std::vector<bag::MovingObject> build_object_vector(
    const bag::Graph& graph,
    const SkeletonIndex& index,
    std::size_t object_count,
    std::uint64_t seed,
    const std::string& layout = "random"
) {
    if (layout == "spread-subgraphs") {
        struct EdgePlacement {
            bag::Edge edge{};
            EdgeWeight edge_weight{0};
            bag::SgId sg_id{0};
        };

        std::vector<EdgePlacement> placements;
        placements.reserve(index.subgraphs().size());
        const auto& edge_to_subgraph = index.edge_to_subgraph();
        for (const auto& sg : index.subgraphs()) {
            std::optional<std::pair<bag::Edge, EdgeWeight>> chosen;
            for (const auto& [edge, weight] : sg.graph.undirected_edges()) {
                if (weight <= 1) {
                    continue;
                }
                const auto ordered = bag::ordered_edge(edge.first, edge.second);
                const auto it = edge_to_subgraph.find(ordered);
                if (it == edge_to_subgraph.end() || it->second != sg.id) {
                    continue;
                }
                const bool internal_edge =
                    sg.internal_vertices.contains(ordered.first) &&
                    sg.internal_vertices.contains(ordered.second);
                if (internal_edge) {
                    chosen = std::make_pair(ordered, weight);
                    break;
                }
                if (!chosen.has_value()) {
                    chosen = std::make_pair(ordered, weight);
                }
            }
            if (chosen.has_value()) {
                placements.push_back(EdgePlacement{chosen->first, chosen->second, sg.id});
            }
        }

        if (placements.empty()) {
            throw std::runtime_error("no valid per-subgraph object placements found");
        }

        std::mt19937_64 rng(seed);
        std::shuffle(placements.begin(), placements.end(), rng);

        std::vector<bag::MovingObject> objects;
        objects.reserve(object_count);
        bag::ObjId next_id = 0;
        std::size_t cursor = 0;
        while (next_id < object_count) {
            if (cursor == placements.size()) {
                std::shuffle(placements.begin(), placements.end(), rng);
                cursor = 0;
            }
            const auto& placement = placements[cursor++];
            std::uniform_int_distribution<EdgeWeight> offset_dist(1, placement.edge_weight - 1);
            objects.push_back(bag::MovingObject{
                next_id++,
                placement.edge,
                offset_dist(rng),
                placement.edge_weight,
            });
        }
        return objects;
    }

    auto set = MovingObjectSet::random_uniform(graph, object_count, seed);
    return set.objects();
}

std::vector<EdgeWeight> parse_weight_list(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::vector<EdgeWeight>& fallback
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    std::vector<EdgeWeight> result;
    std::stringstream ss(it->second);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            result.push_back(static_cast<EdgeWeight>(std::stoul(token)));
        }
    }
    return result.empty() ? fallback : result;
}

bag::Graph rebuild_weighted_graph(
    const bag::Graph& graph,
    double touched_fraction,
    double factor,
    bool increase_only,
    std::uint64_t seed
) {
    bag::Graph updated;
    for (const auto v : graph.vertices()) {
        updated.insert(v);
    }

    auto edges = graph.undirected_edges();
    std::mt19937_64 rng(seed);
    std::shuffle(edges.begin(), edges.end(), rng);
    const auto touched = std::min(
        edges.size(),
        static_cast<std::size_t>(std::floor(static_cast<double>(edges.size()) * touched_fraction))
    );

    std::unordered_set<std::uint64_t> changed;
    changed.reserve(touched * 2U + 1U);
    for (std::size_t i = 0; i < touched; ++i) {
        const auto ordered = bag::ordered_edge(edges[i].first.first, edges[i].first.second);
        changed.insert(bag::pack_pair(ordered.first, ordered.second));
    }

    for (const auto& [edge, weight] : edges) {
        const auto ordered = bag::ordered_edge(edge.first, edge.second);
        EdgeWeight new_weight = weight;
        if (changed.contains(bag::pack_pair(ordered.first, ordered.second))) {
            const auto scaled = static_cast<double>(weight) * factor;
            if (increase_only) {
                new_weight = std::max<EdgeWeight>(weight, static_cast<EdgeWeight>(std::llround(scaled)));
            } else {
                new_weight = std::max<EdgeWeight>(1, static_cast<EdgeWeight>(std::llround(scaled)));
            }
        }
        updated.set_min_undirected_edge(ordered.first, ordered.second, new_weight);
    }

    return updated;
}

std::vector<bag::Subgraph> rebuild_partition_under_weights(
    const bag::Graph& updated_global,
    const std::vector<bag::Subgraph>& base_subgraphs
) {
    std::vector<bag::Subgraph> updated_subgraphs = base_subgraphs;
    for (auto& sg : updated_subgraphs) {
        bag::Graph updated_sg_graph;
        for (const auto v : sg.graph.vertices()) {
            updated_sg_graph.insert(v);
        }
        for (const auto& [edge, _] : sg.graph.undirected_edges()) {
            const auto ordered = bag::ordered_edge(edge.first, edge.second);
            const auto weight = updated_global.get_weight(ordered.first, ordered.second);
            if (!weight.has_value()) {
                continue;
            }
            updated_sg_graph.set_min_undirected_edge(ordered.first, ordered.second, *weight);
        }
        sg.graph = std::move(updated_sg_graph);
        sg.distance = bag::all_pairs_shortest_paths(sg.graph);
    }
    return updated_subgraphs;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage();
            return 1;
        }

        const std::string command = argv[1];
        auto args = bag::parse_cli_args(argc, argv, 2);
        std::string effective_command = command;
        if (effective_command == "default") {
            apply_default_probe_profile(args);
            effective_command = "probe";
        }
        if (effective_command == "sample-queries") {
            const auto path = require_arg(args, "path");
            const auto graph = bag::load_graph_from_file(path);
            const auto query_count = require_usize(args, "count");
            const auto seed =
                static_cast<std::uint64_t>(args.contains("seed") ? require_usize(args, "seed") : 1ULL);
            const auto queries = build_query_workload(graph, query_count, seed);
            std::cout << "[";
            for (std::size_t i = 0; i < queries.size(); ++i) {
                if (i != 0) {
                    std::cout << ",";
                }
                std::cout << "{"
                          << "\"query_id\":" << i << ","
                          << "\"query_u\":" << queries[i].edge.first << ","
                          << "\"query_v\":" << queries[i].edge.second << ","
                          << "\"offset\":" << queries[i].offset
                          << "}";
            }
            std::cout << "]\n";
            return 0;
        }

        const auto path = require_arg(args, "path");
        const bool diagnostics_mode = effective_command == "diagnostics";
        const auto diagnostics_output_dir =
            diagnostics_mode ? std::filesystem::path(require_arg(args, "output-dir")) : std::filesystem::path{};
        const auto theta = require_usize(args, "theta");
        const auto partition_seed =
            static_cast<bag::VertexId>(args.contains("partition-seed") ? require_usize(args, "partition-seed") : 1U);
        const bool paper_strict = optional_bool(args, "paper-strict", false);
        const bool adaptive_z = optional_bool(args, "adaptive-z", false);
        const double alpha = optional_double(args, "alpha", 1.0);
        const bool border_min = optional_bool(args, "border-min", false);
        const bool shortcut_repartition = optional_bool(args, "shortcut-repartition", true);
        const bool use_metis_assignment = args.contains("metis-assignment");
        const auto metis_assignment_path =
            use_metis_assignment ? std::filesystem::path(require_arg(args, "metis-assignment")) : std::filesystem::path{};

        const auto graph = bag::load_graph_from_file(path);
        bag::PartitionOptions partition_options;
        partition_options.theta = theta;
        partition_options.partition_seed = partition_seed;
        partition_options.paper_strict_mode = paper_strict;
        partition_options.adaptive_z = adaptive_z;
        partition_options.adaptive_alpha = alpha;
        partition_options.border_minimization = border_min;
        partition_options.shortcut_repartition = shortcut_repartition;
        if (diagnostics_mode && optional_bool(args, "growth-trace", true)) {
            partition_options.growth_trace_output = diagnostics_output_dir / "vip_growth_trace.jsonl";
        }

        const auto partition_cache_mode =
            bag::parse_partition_cache_mode(args.contains("partition-cache-mode")
                                                ? require_arg(args, "partition-cache-mode")
                                                : "off");
        const auto partition_cache_dir =
            args.contains("partition-cache-dir")
                ? std::filesystem::path(require_arg(args, "partition-cache-dir"))
                : (std::filesystem::path("cache") / "bag_partition");

        long long partition_us = 0;
        std::vector<bag::Subgraph> subgraphs;
        bool partition_cache_hit = false;
        std::string partition_cache_key;
        std::filesystem::path partition_cache_path;
        if (!use_metis_assignment && partition_cache_mode != bag::PartitionCacheMode::Off) {
            partition_cache_key = bag::make_partition_cache_key(path, graph, partition_options);
            partition_cache_path = bag::partition_cache_file_path(
                bag::PartitionCacheConfig{partition_cache_mode, partition_cache_dir, false},
                partition_cache_key
            );
            if (bag::partition_cache_should_try_read(partition_cache_mode)) {
                bag::PartitionCacheEntry entry;
                const auto load_us = bag::calc_execution_time_in_us([&] {
                    if (bag::load_partition_cache(partition_cache_path, entry) &&
                        entry.key == partition_cache_key) {
                        subgraphs = std::move(entry.subgraphs);
                        partition_cache_hit = true;
                    }
                });
                if (partition_cache_hit) {
                    partition_us = load_us;
                }
            }
        }
        if (!partition_cache_hit) {
            partition_us = bag::calc_execution_time_in_us([&] {
                if (use_metis_assignment) {
                    subgraphs = bag::build_subgraphs_from_metis_assignment(
                        graph,
                        bag::MetisImportOptions{metis_assignment_path, true}
                    );
                } else {
                    subgraphs = VfipPartition(graph, partition_options).run();
                }
            });
            if (!use_metis_assignment &&
                bag::partition_cache_should_write(partition_cache_mode)) {
                bag::save_partition_cache(
                    partition_cache_path,
                    partition_cache_key,
                    bag::PartitionRuntimeStats{},
                    subgraphs
                );
            }
        }

        if (effective_command == "partition") {
            std::unordered_set<bag::VertexId> unique_boundary_vertices;
            std::unordered_set<std::uint64_t> unique_skeleton_edges;
            std::map<std::size_t, std::size_t> subgraph_size_histogram;
            std::size_t total_subgraph_vertices = 0;
            std::size_t total_boundary_vertices = 0;
            std::size_t total_internal_vertices = 0;
            std::size_t max_boundary_vertices = 0;
            std::size_t br_violating_subgraphs = 0;
            std::size_t br_violating_boundaries = 0;
            std::vector<double> rb_values;
            for (const auto& sg : subgraphs) {
                ++subgraph_size_histogram[sg.graph.size()];
                total_subgraph_vertices += sg.graph.size();
                total_boundary_vertices += sg.bound_vertices.size();
                total_internal_vertices += sg.internal_vertices.size();
                max_boundary_vertices = std::max(max_boundary_vertices, sg.bound_vertices.size());
                unique_boundary_vertices.insert(sg.bound_vertices.begin(), sg.bound_vertices.end());

                bool subgraph_ok = true;
                const auto sg_edges = sg.graph.undirected_edges();
                for (const auto b : sg.bound_vertices) {
                    bag::HalfWeight rb{0, false};
                    for (const auto other : sg.bound_vertices) {
                        rb = std::max(rb, bag::HalfWeight{sg.distance.get_or_inf(b, other), false});
                    }

                    bag::HalfWeight ib{0, false};
                    for (const auto v : sg.internal_vertices) {
                        ib = std::max(ib, bag::HalfWeight{sg.distance.get_or_inf(b, v), false});
                    }
                    for (const auto& [edge, w] : sg_edges) {
                        const auto lhs = sg.distance.get_or_inf(b, edge.first);
                        const auto rhs = sg.distance.get_or_inf(b, edge.second);
                        if (lhs == bag::kInfWeight || rhs == bag::kInfWeight) {
                            continue;
                        }
                        ib = std::max(ib, bag::distal_point_distance(lhs, rhs, w));
                    }

                    if (rb < ib) {
                        ++br_violating_boundaries;
                        subgraph_ok = false;
                    }
                    rb_values.push_back(rb.to_double());
                }
                if (!subgraph_ok) {
                    ++br_violating_subgraphs;
                }

                std::vector<bag::VertexId> borders(sg.bound_vertices.begin(), sg.bound_vertices.end());
                for (std::size_t i = 0; i < borders.size(); ++i) {
                    for (std::size_t j = i + 1; j < borders.size(); ++j) {
                        if (sg.distance.get_or_inf(borders[i], borders[j]) != bag::kInfWeight) {
                            const auto edge = bag::ordered_edge(borders[i], borders[j]);
                            unique_skeleton_edges.insert(bag::pack_pair(edge.first, edge.second));
                        }
                    }
                }
            }
            const auto skeleton_vertices = unique_boundary_vertices.size();
            const auto skeleton_edges = unique_skeleton_edges.size();
            const auto avg_boundary_per_subgraph =
                subgraphs.empty() ? 0.0 : static_cast<double>(total_boundary_vertices) / static_cast<double>(subgraphs.size());
            const auto vertex_duplication_factor =
                graph.size() == 0 ? 0.0 : static_cast<double>(total_subgraph_vertices) / static_cast<double>(graph.size());
            const auto unique_boundary_ratio =
                graph.size() == 0 ? 0.0 : static_cast<double>(skeleton_vertices) / static_cast<double>(graph.size());
            const auto skeleton_avg_degree =
                skeleton_vertices == 0 ? 0.0 : (2.0 * static_cast<double>(skeleton_edges)) / static_cast<double>(skeleton_vertices);
            double avg_rb = 0.0;
            double median_rb = 0.0;
            if (!rb_values.empty()) {
                avg_rb = std::accumulate(rb_values.begin(), rb_values.end(), 0.0) /
                         static_cast<double>(rb_values.size());
                std::sort(rb_values.begin(), rb_values.end());
                const auto mid = rb_values.size() / 2;
                if ((rb_values.size() % 2U) == 0U) {
                    median_rb = (rb_values[mid - 1] + rb_values[mid]) / 2.0;
                } else {
                    median_rb = rb_values[mid];
                }
            }
            std::cout
                << "{"
                << "\"subgraphs\":" << subgraphs.size() << ","
                << "\"graph_vertices\":" << graph.size() << ","
                << "\"graph_edges\":" << graph.edge_count() << ","
                << "\"skeleton_vertices\":" << skeleton_vertices << ","
                << "\"unique_boundary_vertices\":" << skeleton_vertices << ","
                << "\"skeleton_edges\":" << skeleton_edges << ","
                << "\"skeleton_avg_degree\":" << skeleton_avg_degree << ","
                << "\"total_subgraph_vertices\":" << total_subgraph_vertices << ","
                << "\"total_boundary_vertices\":" << total_boundary_vertices << ","
                << "\"total_internal_vertices\":" << total_internal_vertices << ","
                << "\"avg_boundary_per_subgraph\":" << avg_boundary_per_subgraph << ","
                << "\"avg_rb\":" << avg_rb << ","
                << "\"median_rb\":" << median_rb << ","
                << "\"max_boundary_vertices\":" << max_boundary_vertices << ","
                << "\"br_property_ok\":" << (br_violating_boundaries == 0 ? "true" : "false") << ","
                << "\"br_violating_subgraphs\":" << br_violating_subgraphs << ","
                << "\"br_violating_boundaries\":" << br_violating_boundaries << ","
                << "\"vertex_duplication_factor\":" << vertex_duplication_factor << ","
                << "\"unique_boundary_ratio\":" << unique_boundary_ratio << ","
                << "\"subgraph_size_histogram\":{";
            bool first_hist = true;
            for (const auto& [size, count] : subgraph_size_histogram) {
                if (!first_hist) {
                    std::cout << ",";
                }
                first_hist = false;
                std::cout << "\"" << size << "\":" << count;
            }
            std::cout
                << "},"
                << "\"partition_seed\":" << partition_seed << ","
                << "\"adaptive_z\":" << (adaptive_z ? "true" : "false") << ","
                << "\"alpha\":" << alpha << ","
                << "\"border_min\":" << (border_min ? "true" : "false") << ","
                << "\"shortcut_repartition\":" << (shortcut_repartition ? "true" : "false") << ","
                << "\"partition_us\":" << partition_us
                << "}\n";
            return 0;
        }

        if (effective_command == "factorized-scan") {
            const auto stats = bag::scan_factorized_transfer(subgraphs);
            const auto summary = bag::summarize_factorized_transfer(stats);
            if (args.contains("output-csv")) {
                bag::write_factorized_transfer_csv(require_arg(args, "output-csv"), stats);
            }
            std::cout
                << "{"
                << "\"subgraphs\":" << summary.subgraph_count << ","
                << "\"scanned_subgraphs\":" << summary.scanned_subgraphs << ","
                << "\"compressible_subgraphs\":" << summary.compressible_subgraphs << ","
                << "\"strong_compressible_subgraphs\":" << summary.strong_compressible_subgraphs << ","
                << "\"explicit_directed_arcs\":" << summary.explicit_directed_arcs << ","
                << "\"factorized_directed_arcs\":" << summary.factorized_directed_arcs << ","
                << "\"overall_arc_ratio\":" << summary.overall_arc_ratio << ","
                << "\"avg_hubs_per_subgraph\":" << summary.avg_hubs_per_subgraph << ","
                << "\"avg_arc_ratio\":" << summary.avg_arc_ratio
                << "}\n";
            return 0;
        }

        if (effective_command == "hierarchy-lite") {
            const auto level1_graph = build_skeleton_graph_from_subgraphs(subgraphs);
            const auto level1_summary = summarize_partition(graph, level1_graph, subgraphs);
            const auto group_sizes = parse_size_list(args, "hier-group-sizes", {4, 8, 16, 32});

            std::cout
                << "{"
                << "\"level1_theta\":" << theta << ","
                << "\"level1_partition_us\":" << partition_us << ","
                << "\"level1_subgraphs\":" << level1_summary.subgraphs << ","
                << "\"level1_skeleton_vertices\":" << level1_summary.skeleton_vertices << ","
                << "\"level1_skeleton_edges\":" << level1_summary.skeleton_edges << ","
                << "\"level1_total_boundary_vertices\":" << level1_summary.total_boundary_vertices << ","
                << "\"level1_avg_boundary_per_subgraph\":" << level1_summary.avg_boundary_per_subgraph << ","
                << "\"group_summaries\":[";

            bool first = true;
            for (const auto group_size : group_sizes) {
                const auto summary = analyze_hierarchy_lite(subgraphs, level1_summary, group_size);
                if (!first) {
                    std::cout << ",";
                }
                first = false;
                std::cout
                    << "{"
                    << "\"group_size\":" << summary.group_size << ","
                    << "\"parent_regions\":" << summary.parent_regions << ","
                    << "\"avg_children_per_parent\":" << summary.avg_children_per_parent << ","
                    << "\"max_children_per_parent\":" << summary.max_children_per_parent << ","
                    << "\"upper_unique_boundary_vertices\":" << summary.upper_unique_boundary_vertices << ","
                    << "\"upper_total_boundary_vertices\":" << summary.upper_total_boundary_vertices << ","
                    << "\"upper_parent_graph_edges\":" << summary.upper_parent_graph_edges << ","
                    << "\"upper_estimated_clique_edges\":" << summary.upper_estimated_clique_edges << ","
                    << "\"upper_unique_boundary_ratio_vs_l1\":" << summary.upper_unique_boundary_ratio_vs_l1 << ","
                    << "\"upper_total_boundary_ratio_vs_l1_total\":" << summary.upper_total_boundary_ratio_vs_l1_total << ","
                    << "\"upper_estimated_clique_ratio_vs_l1\":" << summary.upper_estimated_clique_ratio_vs_l1
                    << "}";
            }
            std::cout << "]}\n";
            return 0;
        }

        if (effective_command == "diagnostics") {
            bag::DiagnosticsOptions diagnostics_options;
            diagnostics_options.graph_name = std::filesystem::path(path).filename().string();
            diagnostics_options.impl_version = "cpp-current-diagnostics-v2";
            diagnostics_options.theta = theta;
            diagnostics_options.seed_id = partition_seed;
            diagnostics_options.paper_strict = paper_strict;
            diagnostics_options.shortcut_repartition = shortcut_repartition;
            diagnostics_options.output_dir = diagnostics_output_dir;
            const auto query_count = optional_usize(args, "query-count", 0);
            if (query_count == 0) {
                bag::write_partition_diagnostics(graph, subgraphs, partition_options, diagnostics_options);
            } else {
                const auto fc_rule = parse_fc_rule(args);
                const auto range_radius =
                    static_cast<EdgeWeight>(optional_usize(args, "range-radius", 5000));
                const auto knn_k = optional_usize(args, "knn-k", 10);
                const auto object_count = optional_usize(args, "objects", 1000);
                const auto object_seed =
                    static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
                const auto query_seed =
                    static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));

                bag::SkeletonIndex index = SkeletonIndex::build(graph, subgraphs);
                const auto objects = build_objects(graph, index, object_count, object_seed);
                const auto queries = build_query_workload(graph, query_count, query_seed);
                const auto& raw_objects = objects.objects();

                std::vector<bag::BorderExposureEvent> exposure_events;
                exposure_events.reserve(query_count * 32U);
                std::vector<bag::CliqueRowShadowRecord> row_shadow_records;
                row_shadow_records.reserve(query_count * 64U);
                std::vector<std::vector<bag::ObjId>> baseline_range_results(queries.size());
                std::vector<std::vector<bag::KnnItem>> baseline_knn_results(queries.size());
                std::vector<bag::QueryRuntimeSample> runtime_samples;
                runtime_samples.reserve(query_count * 2U);
                for (std::size_t i = 0; i < queries.size(); ++i) {
                    runtime_samples.push_back(bag::QueryRuntimeSample{
                        i,
                        bag::QueryType::Range,
                        bag::calc_execution_time_in_us([&] {
                                baseline_range_results[i] =
                                    index.range_query(
                                        queries[i],
                                        range_radius,
                                        objects,
                                        fc_rule,
                                        i,
                                        &exposure_events,
                                        &row_shadow_records,
                                        true,
                                        false
                                    ).object_ids;
                        }),
                    });
                    runtime_samples.push_back(bag::QueryRuntimeSample{
                        i,
                        bag::QueryType::Knn,
                        bag::calc_execution_time_in_us([&] {
                                baseline_knn_results[i] =
                                    index.knn_query(
                                        queries[i],
                                        knn_k,
                                        objects,
                                        i,
                                        &exposure_events,
                                        &row_shadow_records,
                                        false,
                                        false,
                                        0,
                                        false
                                    ).items;
                        }),
                    });
                }

                const auto touch_summary = bag::summarize_query_exposures(exposure_events);
                const auto exact_verdicts = bag::evaluate_exact_demotion_candidates(
                    graph,
                    subgraphs,
                    bag::DiagnosticsWorkloadData{
                        &queries,
                        &raw_objects,
                        &baseline_range_results,
                        &baseline_knn_results,
                        &exposure_events,
                        &runtime_samples,
                        fc_rule,
                        range_radius,
                        knn_k,
                        optional_usize(args, "exact-demotion-limit", 0),
                    }
                );
                bag::write_query_exposure_csv(
                    diagnostics_options.output_dir / "query_border_exposure.csv",
                    diagnostics_options,
                    exposure_events
                );
                bag::write_clique_row_shadow_csv(
                    diagnostics_options.output_dir / "first_entry_clique_rows.csv",
                    diagnostics_options,
                    row_shadow_records
                );
                bag::write_clique_row_shadow_summary_csv(
                    diagnostics_options.output_dir / "first_entry_clique_summary.csv",
                    diagnostics_options,
                    row_shadow_records
                );
                bag::write_skeleton_hotspots_csv(
                    diagnostics_options.output_dir / "skeleton_hotspots.csv",
                    graph,
                    subgraphs,
                    exposure_events,
                    runtime_samples
                );
                bag::write_query_runtime_samples_csv(
                    diagnostics_options.output_dir / "query_runtime_samples.csv",
                    runtime_samples
                );
                bag::write_query_runtime_summary_csv(
                    diagnostics_options.output_dir / "query_runtime_summary.csv",
                    runtime_samples
                );
                bag::write_partition_diagnostics(
                    graph,
                    subgraphs,
                    partition_options,
                    diagnostics_options,
                    &touch_summary,
                    &exact_verdicts
                );
            }
            std::cout
                << "{"
                << "\"output_dir\":\"" << diagnostics_options.output_dir.generic_string() << "\","
                << "\"subgraphs\":" << subgraphs.size() << ","
                << "\"partition_us\":" << partition_us
                << "}\n";
            return 0;
        }

        if (effective_command == "batch") {
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_count = require_usize(args, "objects");
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto range_radius =
                static_cast<EdgeWeight>(optional_usize(args, "range-radius", 50000));
            const auto knn_k = optional_usize(args, "knn-k", 10);
            const auto fc_rule = parse_fc_rule(args);

            long long index_build_us = 0;
            bag::SkeletonIndex index;
            index_build_us = bag::calc_execution_time_in_us([&] {
                index = SkeletonIndex::build(graph, std::move(subgraphs));
            });
            (void)index_build_us;

            const bool factorized_transfer = optional_bool(args, "factorized-transfer", false);
            const bool knn_sound_termination = optional_bool(args, "knn-sound-termination", true);
            const double factorized_arc_threshold = optional_double(args, "factorized-arc-threshold", 0.5);
            const auto factorized_border_threshold = optional_usize(args, "factorized-border-threshold", 12);
            const bool gstar_shortcuts = optional_bool(args, "gstar-shortcuts", false);
            const auto gstar_leaf_hops = optional_usize(args, "gstar-leaf-hops", 2);
            const auto gstar_shortcuts_per_subgraph =
                optional_usize(args, "gstar-shortcuts-per-subgraph", 1);
            if (factorized_transfer) {
                index.configure_factorized_transfer(factorized_arc_threshold, factorized_border_threshold);
            }
            if (gstar_shortcuts) {
                (void)index.configure_gstar_shortcuts(
                    gstar_leaf_hops,
                    gstar_shortcuts_per_subgraph
                );
            }
            long long object_build_us = 0;
            IndexedMovingObjectSet objects;
            object_build_us = bag::calc_execution_time_in_us([&] {
                objects = build_objects(graph, index, object_count, object_seed);
            });
            const auto queries = build_query_workload(graph, query_count, query_seed);

            std::vector<long long> range_times;
            std::vector<long long> knn_times;
            range_times.reserve(query_count);
            knn_times.reserve(query_count);
            std::size_t range_total_fc_subgraphs = 0;
            std::size_t range_total_pc_subgraphs = 0;
            std::size_t range_total_hits = 0;
            std::size_t range_total_auto_included_objects = 0;
            std::size_t range_total_br_fc_included_objects = 0;
            std::size_t range_total_partial_edge_auto_included_objects = 0;
            std::size_t range_total_exact_returned_objects = 0;
            std::size_t range_total_exact_checked_objects = 0;
            long long range_total_init_us = 0;
            long long range_total_skeleton_trace_us = 0;
            long long range_total_subgraph_eval_us = 0;
            std::size_t knn_total_candidates_considered = 0;
            std::size_t knn_total_final_candidates = 0;
            std::size_t knn_total_exact_evaluated = 0;
            long long knn_total_init_us = 0;
            long long knn_total_explore_us = 0;
            long long knn_total_finalize_us = 0;
            for (std::size_t i = 0; i < queries.size(); ++i) {
                bag::RangeQueryResult range_result;
                range_times.push_back(bag::calc_execution_time_in_us([&] {
                    range_result = index.range_query(
                        queries[i],
                        range_radius,
                        objects,
                        fc_rule,
                        i,
                        nullptr,
                        nullptr,
                        true,
                        factorized_transfer
                    );
                }));
                range_total_hits += range_result.object_ids.size();
                range_total_auto_included_objects += range_result.auto_included_objects;
                range_total_br_fc_included_objects += range_result.br_fc_included_objects;
                range_total_partial_edge_auto_included_objects +=
                    range_result.partial_edge_auto_included_objects;
                range_total_exact_returned_objects += range_result.exact_returned_objects;
                range_total_exact_checked_objects += range_result.exact_checked_objects;
                range_total_fc_subgraphs += range_result.fc_subgraphs;
                range_total_pc_subgraphs += range_result.pc_subgraphs;
                range_total_init_us += range_result.init_us;
                range_total_skeleton_trace_us += range_result.skeleton_trace_us;
                range_total_subgraph_eval_us += range_result.subgraph_eval_us;
                bag::KnnQueryResult knn_result;
                knn_times.push_back(bag::calc_execution_time_in_us([&] {
                    knn_result = index.knn_query(
                        queries[i],
                        knn_k,
                        objects,
                        i,
                        nullptr,
                        nullptr,
                        false,
                        false,
                        0,
                        factorized_transfer,
                        knn_sound_termination
                    );
                }));
                knn_total_candidates_considered += knn_result.candidates_considered;
                knn_total_final_candidates += knn_result.final_candidates;
                knn_total_exact_evaluated += knn_result.exact_evaluated;
                knn_total_init_us += knn_result.init_us;
                knn_total_explore_us += knn_result.explore_us;
                knn_total_finalize_us += knn_result.finalize_us;
            }

            const auto range_total = std::accumulate(range_times.begin(), range_times.end(), 0LL);
            const auto knn_total = std::accumulate(knn_times.begin(), knn_times.end(), 0LL);
            const auto range_min = range_times.empty() ? 0LL : *std::min_element(range_times.begin(), range_times.end());
            const auto range_max = range_times.empty() ? 0LL : *std::max_element(range_times.begin(), range_times.end());
            const auto knn_min = knn_times.empty() ? 0LL : *std::min_element(knn_times.begin(), knn_times.end());
            const auto knn_max = knn_times.empty() ? 0LL : *std::max_element(knn_times.begin(), knn_times.end());

            std::cout
                << "{"
                << "\"query_count\":" << query_count << ","
                << "\"query_seed\":" << query_seed << ","
                << "\"objects\":" << object_count << ","
                << "\"object_seed\":" << object_seed << ","
                << "\"partition_us\":" << partition_us << ","
                << "\"index_build_us\":" << index_build_us << ","
                << "\"object_build_us\":" << object_build_us << ","
                << "\"gstar_shortcuts\":" << (gstar_shortcuts ? "true" : "false") << ","
                << "\"gstar_shortcut_edges_added\":" << index.gstar_shortcut_edges_added() << ","
                << "\"range_radius\":" << range_radius << ","
                << "\"knn_k\":" << knn_k << ","
                << "\"range_avg_us\":" << (queries.empty() ? 0.0 : static_cast<double>(range_total) / static_cast<double>(queries.size())) << ","
                << "\"range_min_us\":" << range_min << ","
                << "\"range_max_us\":" << range_max << ","
                << "\"range_total_fc_subgraphs\":" << range_total_fc_subgraphs << ","
                << "\"range_total_pc_subgraphs\":" << range_total_pc_subgraphs << ","
                << "\"range_total_hits\":" << range_total_hits << ","
                << "\"range_total_auto_included_objects\":" << range_total_auto_included_objects << ","
                << "\"range_total_br_fc_included_objects\":" << range_total_br_fc_included_objects << ","
                << "\"range_total_partial_edge_auto_included_objects\":"
                << range_total_partial_edge_auto_included_objects << ","
                << "\"range_total_exact_returned_objects\":" << range_total_exact_returned_objects << ","
                << "\"range_total_exact_checked_objects\":" << range_total_exact_checked_objects << ","
                << "\"range_avg_init_us\":"
                << (queries.empty() ? 0.0 : static_cast<double>(range_total_init_us) / static_cast<double>(queries.size())) << ","
                << "\"range_avg_skeleton_trace_us\":"
                << (queries.empty() ? 0.0 : static_cast<double>(range_total_skeleton_trace_us) / static_cast<double>(queries.size())) << ","
                << "\"range_avg_subgraph_eval_us\":"
                << (queries.empty() ? 0.0 : static_cast<double>(range_total_subgraph_eval_us) / static_cast<double>(queries.size())) << ","
                << "\"range_br_fc_coverage_ratio\":"
                << (range_total_hits == 0
                        ? 0.0
                        : static_cast<double>(range_total_br_fc_included_objects) /
                              static_cast<double>(range_total_hits)) << ","
                << "\"range_non_br_coverage_ratio\":"
                << (range_total_hits == 0
                        ? 0.0
                        : static_cast<double>(
                              range_total_partial_edge_auto_included_objects +
                              range_total_exact_returned_objects
                          ) /
                              static_cast<double>(range_total_hits)) << ","
                << "\"knn_avg_us\":" << (queries.empty() ? 0.0 : static_cast<double>(knn_total) / static_cast<double>(queries.size())) << ","
                << "\"knn_min_us\":" << knn_min << ","
                << "\"knn_max_us\":" << knn_max << ","
                << "\"knn_total_candidates_considered\":" << knn_total_candidates_considered << ","
                << "\"knn_total_final_candidates\":" << knn_total_final_candidates << ","
                << "\"knn_total_exact_evaluated\":" << knn_total_exact_evaluated << ","
                << "\"knn_avg_init_us\":"
                << (queries.empty() ? 0.0 : static_cast<double>(knn_total_init_us) / static_cast<double>(queries.size())) << ","
                << "\"knn_avg_explore_us\":"
                << (queries.empty() ? 0.0 : static_cast<double>(knn_total_explore_us) / static_cast<double>(queries.size())) << ","
                << "\"knn_avg_finalize_us\":"
                << (queries.empty() ? 0.0 : static_cast<double>(knn_total_finalize_us) / static_cast<double>(queries.size())) << ","
                << "\"knn_refinement_avoidance_ratio\":"
                << (knn_total_candidates_considered == 0
                        ? 0.0
                        : 1.0 - static_cast<double>(knn_total_exact_evaluated) /
                                    static_cast<double>(knn_total_candidates_considered))
                << "}\n";
            return 0;
        }

        if (effective_command == "query-sweep") {
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_count = require_usize(args, "objects");
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto radii = parse_size_list(args, "radii", {});
            const auto ks = parse_size_list(args, "ks", {});
            const auto fc_rule = parse_fc_rule(args);
            const auto knn_pc_dijkstra_mode_it = args.find("knn-pc-dijkstra-mode");
            const std::string knn_pc_dijkstra_mode =
                knn_pc_dijkstra_mode_it == args.end() ? "subgraph" : knn_pc_dijkstra_mode_it->second;
            if (knn_pc_dijkstra_mode != "subgraph" && knn_pc_dijkstra_mode != "global" &&
                knn_pc_dijkstra_mode != "skeleton") {
                throw std::runtime_error("unsupported --knn-pc-dijkstra-mode, expected subgraph|global|skeleton");
            }
            const bool knn_sound_termination = optional_bool(args, "knn-sound-termination", true);
            const bool verify = optional_bool(args, "verify", false);
            const auto output_csv_it = args.find("output-csv");
            if (radii.empty() && ks.empty()) {
                throw std::runtime_error("query-sweep requires --radii and/or --ks");
            }

            long long index_build_us = 0;
            bag::SkeletonIndex index;
            index_build_us = bag::calc_execution_time_in_us([&] {
                index = SkeletonIndex::build(graph, std::move(subgraphs));
            });

            long long object_build_us = 0;
            IndexedMovingObjectSet objects;
            object_build_us = bag::calc_execution_time_in_us([&] {
                objects = build_objects(graph, index, object_count, object_seed);
            });
            const auto queries = build_query_workload(graph, query_count, query_seed);

            struct SweepRow {
                std::string mode;
                std::size_t x_value{0};
                EdgeWeight radius{0};
                double avg_us{0.0};
                long long min_us{0};
                long long max_us{0};
                long long total_us{0};
                long long avg_init_us{0};
                long long avg_skeleton_or_explore_us{0};
                long long avg_eval_or_finalize_us{0};
                std::size_t total_fc_subgraphs{0};
                std::size_t total_pc_subgraphs{0};
                std::size_t total_hits_or_returned{0};
                std::size_t total_candidates_considered{0};
                std::size_t total_final_candidates{0};
                std::size_t total_exact_evaluated{0};
                double fc_coverage_ratio{0.0};
                double refinement_avoidance_ratio{0.0};
                bool exact{true};
                long long exact_us{0};
            };

            std::vector<SweepRow> rows;
            rows.reserve(radii.size() + ks.size());

            for (const auto radius_value : radii) {
                std::vector<long long> times;
                times.reserve(queries.size());
                std::size_t total_fc_subgraphs = 0;
                std::size_t total_pc_subgraphs = 0;
                std::size_t total_hits = 0;
                std::size_t total_br_fc_included_objects = 0;
                long long total_init_us = 0;
                long long total_skeleton_us = 0;
                long long total_eval_us = 0;
                bool exact = true;
                long long exact_us = 0;
                for (std::size_t i = 0; i < queries.size(); ++i) {
                    bag::RangeQueryResult result;
                    times.push_back(bag::calc_execution_time_in_us([&] {
                        result = index.range_query(
                            queries[i],
                            static_cast<EdgeWeight>(radius_value),
                            objects,
                            fc_rule,
                            i,
                            nullptr,
                            nullptr,
                            true,
                            false
                        );
                    }));
                    total_fc_subgraphs += result.fc_subgraphs;
                    total_pc_subgraphs += result.pc_subgraphs;
                    total_hits += result.object_ids.size();
                    total_br_fc_included_objects += result.br_fc_included_objects;
                    total_init_us += result.init_us;
                    total_skeleton_us += result.skeleton_trace_us;
                    total_eval_us += result.subgraph_eval_us;
                    if (verify) {
                        std::vector<bag::ObjId> exact_ids;
                        exact_us += bag::calc_execution_time_in_us([&] {
                            exact_ids = index.exact_range_query(
                                queries[i],
                                static_cast<EdgeWeight>(radius_value),
                                objects
                            );
                        });
                        std::unordered_set<bag::ObjId> lhs(result.object_ids.begin(), result.object_ids.end());
                        std::unordered_set<bag::ObjId> rhs(exact_ids.begin(), exact_ids.end());
                        if (lhs != rhs) {
                            exact = false;
                        }
                    }
                }
                const auto total_us = std::accumulate(times.begin(), times.end(), 0LL);
                const auto min_us = times.empty() ? 0LL : *std::min_element(times.begin(), times.end());
                const auto max_us = times.empty() ? 0LL : *std::max_element(times.begin(), times.end());
                const auto denom = queries.empty() ? 1.0 : static_cast<double>(queries.size());
                rows.push_back(SweepRow{
                    "range",
                    radius_value,
                    static_cast<EdgeWeight>(radius_value),
                    times.empty() ? 0.0 : static_cast<double>(total_us) / denom,
                    min_us,
                    max_us,
                    total_us,
                    static_cast<long long>(static_cast<double>(total_init_us) / denom),
                    static_cast<long long>(static_cast<double>(total_skeleton_us) / denom),
                    static_cast<long long>(static_cast<double>(total_eval_us) / denom),
                    total_fc_subgraphs,
                    total_pc_subgraphs,
                    total_hits,
                    0,
                    0,
                    0,
                    total_hits == 0
                        ? 0.0
                        : static_cast<double>(total_br_fc_included_objects) /
                              static_cast<double>(total_hits),
                    0.0,
                    exact,
                    exact_us,
                });
            }

            for (const auto k_value : ks) {
                std::vector<long long> times;
                times.reserve(queries.size());
                std::size_t total_returned = 0;
                std::size_t total_candidates_considered = 0;
                std::size_t total_final_candidates = 0;
                std::size_t total_exact_evaluated = 0;
                long long total_init_us = 0;
                long long total_explore_us = 0;
                long long total_finalize_us = 0;
                bool exact = true;
                long long exact_us = 0;
                for (std::size_t i = 0; i < queries.size(); ++i) {
                    bag::KnnQueryResult result;
                    times.push_back(bag::calc_execution_time_in_us([&] {
                        if (knn_pc_dijkstra_mode == "skeleton") {
                            result = index.knn_query(
                                queries[i],
                                k_value,
                                objects,
                                i,
                                nullptr,
                                nullptr,
                                false,
                                false,
                                0,
                                false,
                                knn_sound_termination
                            );
                        } else if (knn_pc_dijkstra_mode == "global") {
                            result = index.knn_query_global_dijkstra(
                                queries[i],
                                k_value,
                                objects,
                                i
                            );
                        } else {
                            result = index.knn_query_local_dijkstra(
                                queries[i],
                                k_value,
                                objects,
                                fc_rule,
                                i
                            );
                        }
                    }));
                    total_returned += result.items.size();
                    total_candidates_considered += result.candidates_considered;
                    total_final_candidates += result.final_candidates;
                    total_exact_evaluated += result.exact_evaluated;
                    total_init_us += result.init_us;
                    total_explore_us += result.explore_us;
                    total_finalize_us += result.finalize_us;
                    if (verify) {
                        bag::KnnQueryResult exact_result;
                        exact_us += bag::calc_execution_time_in_us([&] {
                            exact_result = index.exact_knn_query(queries[i], k_value, objects);
                        });
                        if (result.items.size() != exact_result.items.size()) {
                            exact = false;
                        } else {
                            for (std::size_t j = 0; j < result.items.size(); ++j) {
                                if (result.items[j].id != exact_result.items[j].id ||
                                    result.items[j].distance != exact_result.items[j].distance) {
                                    exact = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                const auto total_us = std::accumulate(times.begin(), times.end(), 0LL);
                const auto min_us = times.empty() ? 0LL : *std::min_element(times.begin(), times.end());
                const auto max_us = times.empty() ? 0LL : *std::max_element(times.begin(), times.end());
                const auto denom = queries.empty() ? 1.0 : static_cast<double>(queries.size());
                rows.push_back(SweepRow{
                    "knn",
                    k_value,
                    0,
                    times.empty() ? 0.0 : static_cast<double>(total_us) / denom,
                    min_us,
                    max_us,
                    total_us,
                    static_cast<long long>(static_cast<double>(total_init_us) / denom),
                    static_cast<long long>(static_cast<double>(total_explore_us) / denom),
                    static_cast<long long>(static_cast<double>(total_finalize_us) / denom),
                    0,
                    0,
                    total_returned,
                    total_candidates_considered,
                    total_final_candidates,
                    total_exact_evaluated,
                    0.0,
                    total_candidates_considered == 0
                        ? 0.0
                        : 1.0 - static_cast<double>(total_exact_evaluated) /
                                    static_cast<double>(total_candidates_considered),
                    exact,
                    exact_us,
                });
            }

            if (output_csv_it != args.end()) {
                const auto output_path = std::filesystem::path(output_csv_it->second);
                if (!output_path.parent_path().empty()) {
                    std::filesystem::create_directories(output_path.parent_path());
                }
                std::ofstream out(output_path);
                out << "mode,x_value,radius,query_count,objects,avg_us,min_us,max_us,total_us,"
                       "avg_init_us,avg_skeleton_or_explore_us,avg_eval_or_finalize_us,"
                       "total_fc_subgraphs,total_pc_subgraphs,total_hits_or_returned,"
                       "total_candidates_considered,total_final_candidates,total_exact_evaluated,"
                       "fc_coverage_ratio,refinement_avoidance_ratio,exact,exact_us,"
                       "partition_us,index_build_us,object_build_us\n";
                for (const auto& row : rows) {
                    out << row.mode << ","
                        << row.x_value << ","
                        << row.radius << ","
                        << query_count << ","
                        << object_count << ","
                        << row.avg_us << ","
                        << row.min_us << ","
                        << row.max_us << ","
                        << row.total_us << ","
                        << row.avg_init_us << ","
                        << row.avg_skeleton_or_explore_us << ","
                        << row.avg_eval_or_finalize_us << ","
                        << row.total_fc_subgraphs << ","
                        << row.total_pc_subgraphs << ","
                        << row.total_hits_or_returned << ","
                        << row.total_candidates_considered << ","
                        << row.total_final_candidates << ","
                        << row.total_exact_evaluated << ","
                        << row.fc_coverage_ratio << ","
                        << row.refinement_avoidance_ratio << ","
                        << (row.exact ? "true" : "false") << ","
                        << row.exact_us << ","
                        << partition_us << ","
                        << index_build_us << ","
                        << object_build_us << "\n";
                }
            }

            std::cout << "{"
                      << "\"query_count\":" << query_count << ","
                      << "\"query_seed\":" << query_seed << ","
                      << "\"objects\":" << object_count << ","
                      << "\"object_seed\":" << object_seed << ","
                      << "\"knn_pc_dijkstra_mode\":\"" << knn_pc_dijkstra_mode << "\","
                      << "\"partition_us\":" << partition_us << ","
                      << "\"index_build_us\":" << index_build_us << ","
                      << "\"object_build_us\":" << object_build_us << ","
                      << "\"rows\":[";
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (i != 0) {
                    std::cout << ",";
                }
                const auto& row = rows[i];
                std::cout << "{"
                          << "\"mode\":\"" << row.mode << "\","
                          << "\"x_value\":" << row.x_value << ","
                          << "\"radius\":" << row.radius << ","
                          << "\"avg_us\":" << row.avg_us << ","
                          << "\"min_us\":" << row.min_us << ","
                          << "\"max_us\":" << row.max_us << ","
                          << "\"total_us\":" << row.total_us << ","
                          << "\"avg_init_us\":" << row.avg_init_us << ","
                          << "\"avg_skeleton_or_explore_us\":" << row.avg_skeleton_or_explore_us << ","
                          << "\"avg_eval_or_finalize_us\":" << row.avg_eval_or_finalize_us << ","
                          << "\"total_fc_subgraphs\":" << row.total_fc_subgraphs << ","
                          << "\"total_pc_subgraphs\":" << row.total_pc_subgraphs << ","
                          << "\"total_hits_or_returned\":" << row.total_hits_or_returned << ","
                          << "\"total_candidates_considered\":" << row.total_candidates_considered << ","
                          << "\"total_final_candidates\":" << row.total_final_candidates << ","
                          << "\"total_exact_evaluated\":" << row.total_exact_evaluated << ","
                          << "\"fc_coverage_ratio\":" << row.fc_coverage_ratio << ","
                          << "\"refinement_avoidance_ratio\":" << row.refinement_avoidance_ratio << ","
                          << "\"exact\":" << (row.exact ? "true" : "false") << ","
                          << "\"exact_us\":" << row.exact_us
                          << "}";
            }
            std::cout << "]}\n";
            return 0;
        }

        if (effective_command == "coverage-sweep") {
            const auto mode = require_arg(args, "mode");
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_count = require_usize(args, "objects");
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto layout_it = args.find("object-layout");
            const std::string object_layout = (layout_it == args.end()) ? "random" : layout_it->second;
            const auto output_csv_it = args.find("output-csv");

            bag::SkeletonIndex index;
            const auto index_build_us = bag::calc_execution_time_in_us([&] {
                index = SkeletonIndex::build(graph, std::move(subgraphs));
            });
            (void)index_build_us;

            IndexedMovingObjectSet objects;
            const auto object_build_us = bag::calc_execution_time_in_us([&] {
                objects = build_objects(graph, index, object_count, object_seed, object_layout);
            });
            (void)object_build_us;
            const auto queries = build_query_workload(graph, query_count, query_seed);

            struct CoverageRow {
                std::string mode;
                std::size_t x_value{0};
                std::size_t query_count{0};
                std::size_t total_result_subgraphs{0};
                std::size_t total_fully_admitted_subgraphs{0};
                double fully_admitted_subgraph_ratio{0.0};
                std::size_t total_result_objects{0};
                std::size_t total_fully_admitted_objects{0};
                double fully_admitted_object_ratio{0.0};
            };

            std::vector<CoverageRow> rows;

            if (mode == "range") {
                const auto radii = parse_size_list(args, "radii", {});
                if (radii.empty()) {
                    throw std::runtime_error("coverage-sweep range mode requires --radii");
                }
                rows.reserve(radii.size());
                for (const auto radius_value : radii) {
                    std::size_t total_result_subgraphs = 0;
                    std::size_t total_fully_admitted_subgraphs = 0;
                    std::size_t total_result_objects = 0;
                    std::size_t total_fully_admitted_objects = 0;
                    for (const auto& query : queries) {
                        const auto exact_ids =
                            index.exact_range_query(query, static_cast<EdgeWeight>(radius_value), objects);
                        const auto coverage = summarize_exact_subgraph_coverage(exact_ids, objects);
                        total_result_subgraphs += coverage.result_subgraphs;
                        total_fully_admitted_subgraphs += coverage.fully_admitted_subgraphs;
                        total_result_objects += coverage.result_objects;
                        total_fully_admitted_objects += coverage.fully_admitted_objects;
                    }
                    rows.push_back(CoverageRow{
                        "range",
                        radius_value,
                        queries.size(),
                        total_result_subgraphs,
                        total_fully_admitted_subgraphs,
                        total_result_subgraphs == 0
                            ? 0.0
                            : static_cast<double>(total_fully_admitted_subgraphs) /
                                  static_cast<double>(total_result_subgraphs),
                        total_result_objects,
                        total_fully_admitted_objects,
                        total_result_objects == 0
                            ? 0.0
                            : static_cast<double>(total_fully_admitted_objects) /
                                  static_cast<double>(total_result_objects),
                    });
                }
            } else if (mode == "knn") {
                const auto ks = parse_size_list(args, "ks", {});
                if (ks.empty()) {
                    throw std::runtime_error("coverage-sweep knn mode requires --ks");
                }
                rows.reserve(ks.size());
                for (const auto k_value : ks) {
                    std::size_t total_result_subgraphs = 0;
                    std::size_t total_fully_admitted_subgraphs = 0;
                    std::size_t total_result_objects = 0;
                    std::size_t total_fully_admitted_objects = 0;
                    for (const auto& query : queries) {
                        const auto exact_result = index.exact_knn_query(query, k_value, objects);
                        const auto exact_ids = extract_knn_ids(exact_result);
                        const auto coverage = summarize_exact_subgraph_coverage(exact_ids, objects);
                        total_result_subgraphs += coverage.result_subgraphs;
                        total_fully_admitted_subgraphs += coverage.fully_admitted_subgraphs;
                        total_result_objects += coverage.result_objects;
                        total_fully_admitted_objects += coverage.fully_admitted_objects;
                    }
                    rows.push_back(CoverageRow{
                        "knn",
                        k_value,
                        queries.size(),
                        total_result_subgraphs,
                        total_fully_admitted_subgraphs,
                        total_result_subgraphs == 0
                            ? 0.0
                            : static_cast<double>(total_fully_admitted_subgraphs) /
                                  static_cast<double>(total_result_subgraphs),
                        total_result_objects,
                        total_fully_admitted_objects,
                        total_result_objects == 0
                            ? 0.0
                            : static_cast<double>(total_fully_admitted_objects) /
                                  static_cast<double>(total_result_objects),
                    });
                }
            } else {
                throw std::runtime_error("coverage-sweep requires --mode range|knn");
            }

            if (output_csv_it != args.end()) {
                const auto output_path = std::filesystem::path(output_csv_it->second);
                if (!output_path.parent_path().empty()) {
                    std::filesystem::create_directories(output_path.parent_path());
                }
                std::ofstream out(output_path);
                out << "mode,x_value,query_count,total_result_subgraphs,total_fully_admitted_subgraphs,"
                       "fully_admitted_subgraph_ratio,total_result_objects,total_fully_admitted_objects,"
                       "fully_admitted_object_ratio\n";
                for (const auto& row : rows) {
                    out << row.mode << ","
                        << row.x_value << ","
                        << row.query_count << ","
                        << row.total_result_subgraphs << ","
                        << row.total_fully_admitted_subgraphs << ","
                        << row.fully_admitted_subgraph_ratio << ","
                        << row.total_result_objects << ","
                        << row.total_fully_admitted_objects << ","
                        << row.fully_admitted_object_ratio << "\n";
                }
            }

            std::cout << "[";
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (i != 0) {
                    std::cout << ",";
                }
                const auto& row = rows[i];
                std::cout << "{"
                          << "\"mode\":\"" << row.mode << "\","
                          << "\"x_value\":" << row.x_value << ","
                          << "\"query_count\":" << row.query_count << ","
                          << "\"total_result_subgraphs\":" << row.total_result_subgraphs << ","
                          << "\"total_fully_admitted_subgraphs\":" << row.total_fully_admitted_subgraphs << ","
                          << "\"fully_admitted_subgraph_ratio\":" << row.fully_admitted_subgraph_ratio << ","
                          << "\"total_result_objects\":" << row.total_result_objects << ","
                          << "\"total_fully_admitted_objects\":" << row.total_fully_admitted_objects << ","
                          << "\"fully_admitted_object_ratio\":" << row.fully_admitted_object_ratio
                          << "}";
            }
            std::cout << "]\n";
            return 0;
        }

        if (effective_command == "coverage-compare") {
            const auto mode = require_arg(args, "mode");
            const auto query_count = require_usize(args, "query-count");
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_count = require_usize(args, "objects");
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto output_csv_it = args.find("output-csv");
            const auto metis_assignment =
                std::filesystem::path(require_arg(args, "metis-assignment"));
            const auto kahip_assignment =
                std::filesystem::path(require_arg(args, "kahip-assignment"));

            auto br_subgraphs = VfipPartition(graph, partition_options).run();
            auto metis_subgraphs = bag::build_subgraphs_from_metis_assignment(
                graph,
                bag::MetisImportOptions{metis_assignment, true}
            );
            auto kahip_subgraphs = bag::build_subgraphs_from_metis_assignment(
                graph,
                bag::MetisImportOptions{kahip_assignment, true}
            );

            auto raw_objects = MovingObjectSet::random_uniform(graph, object_count, object_seed).objects();
            const auto queries = build_query_workload(graph, query_count, query_seed);

            auto build_indexed = [&](std::vector<bag::Subgraph> sg_list) {
                auto index = SkeletonIndex::build(graph, std::move(sg_list));
                auto indexed = IndexedMovingObjectSet::from_partition(
                    raw_objects,
                    index.edge_to_subgraph(),
                    index.subgraphs()
                );
                return std::make_pair(std::move(index), std::move(indexed));
            };

            auto [br_index, br_objects] = build_indexed(std::move(br_subgraphs));
            auto [metis_index, metis_objects] = build_indexed(std::move(metis_subgraphs));
            auto [kahip_index, kahip_objects] = build_indexed(std::move(kahip_subgraphs));

            struct CoverageCompareRow {
                std::string method;
                std::string mode;
                std::size_t x_value{0};
                std::size_t query_count{0};
                std::size_t total_result_subgraphs{0};
                std::size_t total_fully_admitted_subgraphs{0};
                double fully_admitted_subgraph_ratio{0.0};
                std::size_t total_result_objects{0};
                std::size_t total_fully_admitted_objects{0};
                double fully_admitted_object_ratio{0.0};
            };

            std::vector<CoverageCompareRow> rows;
            const auto emit_row = [&](const std::string& method,
                                      const std::string& mode_name,
                                      std::size_t x_value,
                                      const IndexedMovingObjectSet& indexed_objects,
                                      const std::vector<std::vector<bag::ObjId>>& exact_results) {
                std::size_t total_result_subgraphs = 0;
                std::size_t total_fully_admitted_subgraphs = 0;
                std::size_t total_result_objects = 0;
                std::size_t total_fully_admitted_objects = 0;
                for (const auto& exact_ids : exact_results) {
                    const auto coverage = summarize_exact_subgraph_coverage(exact_ids, indexed_objects);
                    total_result_subgraphs += coverage.result_subgraphs;
                    total_fully_admitted_subgraphs += coverage.fully_admitted_subgraphs;
                    total_result_objects += coverage.result_objects;
                    total_fully_admitted_objects += coverage.fully_admitted_objects;
                }
                rows.push_back(CoverageCompareRow{
                    method,
                    mode_name,
                    x_value,
                    queries.size(),
                    total_result_subgraphs,
                    total_fully_admitted_subgraphs,
                    total_result_subgraphs == 0
                        ? 0.0
                        : static_cast<double>(total_fully_admitted_subgraphs) /
                              static_cast<double>(total_result_subgraphs),
                    total_result_objects,
                    total_fully_admitted_objects,
                    total_result_objects == 0
                        ? 0.0
                        : static_cast<double>(total_fully_admitted_objects) /
                              static_cast<double>(total_result_objects),
                });
            };

            if (mode == "range") {
                const auto radii = parse_size_list(args, "radii", {});
                if (radii.empty()) {
                    throw std::runtime_error("coverage-compare range mode requires --radii");
                }
                rows.reserve(radii.size() * 3U);
                for (const auto radius_value : radii) {
                    std::vector<std::vector<bag::ObjId>> exact_results;
                    exact_results.reserve(queries.size());
                    for (const auto& query : queries) {
                        exact_results.push_back(
                            br_index.exact_range_query(query, static_cast<EdgeWeight>(radius_value), br_objects)
                        );
                    }
                    emit_row("BAG-VIP", "range", radius_value, br_objects, exact_results);
                    emit_row("METIS", "range", radius_value, metis_objects, exact_results);
                    emit_row("KaHIP", "range", radius_value, kahip_objects, exact_results);
                }
            } else if (mode == "knn") {
                const auto ks = parse_size_list(args, "ks", {});
                if (ks.empty()) {
                    throw std::runtime_error("coverage-compare knn mode requires --ks");
                }
                rows.reserve(ks.size() * 3U);
                for (const auto k_value : ks) {
                    std::vector<std::vector<bag::ObjId>> exact_results;
                    exact_results.reserve(queries.size());
                    for (const auto& query : queries) {
                        exact_results.push_back(
                            extract_knn_ids(br_index.exact_knn_query(query, k_value, br_objects))
                        );
                    }
                    emit_row("BAG-VIP", "knn", k_value, br_objects, exact_results);
                    emit_row("METIS", "knn", k_value, metis_objects, exact_results);
                    emit_row("KaHIP", "knn", k_value, kahip_objects, exact_results);
                }
            } else {
                throw std::runtime_error("coverage-compare requires --mode range|knn");
            }

            if (output_csv_it != args.end()) {
                const auto output_path = std::filesystem::path(output_csv_it->second);
                if (!output_path.parent_path().empty()) {
                    std::filesystem::create_directories(output_path.parent_path());
                }
                std::ofstream out(output_path);
                out << "method,mode,x_value,query_count,total_result_subgraphs,total_fully_admitted_subgraphs,"
                       "fully_admitted_subgraph_ratio,total_result_objects,total_fully_admitted_objects,"
                       "fully_admitted_object_ratio\n";
                for (const auto& row : rows) {
                    out << row.method << ","
                        << row.mode << ","
                        << row.x_value << ","
                        << row.query_count << ","
                        << row.total_result_subgraphs << ","
                        << row.total_fully_admitted_subgraphs << ","
                        << row.fully_admitted_subgraph_ratio << ","
                        << row.total_result_objects << ","
                        << row.total_fully_admitted_objects << ","
                        << row.fully_admitted_object_ratio << "\n";
                }
            }

            std::cout << "[";
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (i != 0) {
                    std::cout << ",";
                }
                const auto& row = rows[i];
                std::cout << "{"
                          << "\"method\":\"" << row.method << "\","
                          << "\"mode\":\"" << row.mode << "\","
                          << "\"x_value\":" << row.x_value << ","
                          << "\"query_count\":" << row.query_count << ","
                          << "\"total_result_subgraphs\":" << row.total_result_subgraphs << ","
                          << "\"total_fully_admitted_subgraphs\":" << row.total_fully_admitted_subgraphs << ","
                          << "\"fully_admitted_subgraph_ratio\":" << row.fully_admitted_subgraph_ratio << ","
                          << "\"total_result_objects\":" << row.total_result_objects << ","
                          << "\"total_fully_admitted_objects\":" << row.total_fully_admitted_objects << ","
                          << "\"fully_admitted_object_ratio\":" << row.fully_admitted_object_ratio
                          << "}";
            }
            std::cout << "]\n";
            return 0;
        }

        if (effective_command == "maint-bench") {
            const std::string mode =
                args.contains("mode") ? require_arg(args, "mode") : "knn";
            const double car_percent = optional_double(args, "car-percent", 0.025);
            const double change_percent = optional_double(args, "change-percent", 0.01);
            const auto query_per_update = optional_usize(args, "query-per-update", 10);
            const auto epochs = optional_usize(args, "epochs", 100);
            const auto knn_k = optional_usize(args, "knn-k", 10);
            const auto range_radius =
                static_cast<EdgeWeight>(optional_usize(args, "range-radius", 50000));
            const auto fc_rule = parse_fc_rule(args);
            const auto query_seed =
                static_cast<std::uint64_t>(optional_usize(args, "query-seed", 17));
            const auto object_seed =
                static_cast<std::uint64_t>(optional_usize(args, "object-seed", 7));
            const auto layout_it = args.find("object-layout");
            const std::string object_layout = (layout_it == args.end()) ? "random" : layout_it->second;

            long long index_build_us = 0;
            bag::SkeletonIndex index;
            index_build_us = bag::calc_execution_time_in_us([&] {
                index = SkeletonIndex::build(graph, std::move(subgraphs));
            });

            const std::size_t object_count = args.contains("objects")
                ? require_usize(args, "objects")
                : static_cast<std::size_t>(
                      std::max(1.0, std::floor(static_cast<double>(graph.size()) * car_percent))
                  );
            const std::size_t change_count = args.contains("change-count")
                ? std::min(object_count, require_usize(args, "change-count"))
                : std::min(
                      object_count,
                      static_cast<std::size_t>(std::floor(static_cast<double>(object_count) * change_percent))
                  );

            auto active_objects = build_object_vector(graph, index, object_count, object_seed, object_layout);
            const auto queries = build_query_workload(graph, std::max<std::size_t>(epochs, query_per_update), query_seed);

            long long initial_object_index_us = 0;
            IndexedMovingObjectSet indexed_objects;
            initial_object_index_us = bag::calc_execution_time_in_us([&] {
                indexed_objects = IndexedMovingObjectSet::from_partition(
                    active_objects,
                    index.edge_to_subgraph(),
                    index.subgraphs()
                );
            });

            long long one_time_knn_total_us = 0;
            for (std::size_t i = 0; i < epochs; ++i) {
                one_time_knn_total_us += bag::calc_execution_time_in_us([&] {
                    if (mode == "range") {
                        (void)index.range_query(
                            queries[i % queries.size()],
                            range_radius,
                            indexed_objects,
                            fc_rule,
                            i,
                            nullptr,
                            nullptr,
                            true,
                            false
                        );
                    } else if (mode == "knn") {
                        (void)index.knn_query(
                            queries[i % queries.size()],
                            knn_k,
                            indexed_objects,
                            i,
                            nullptr,
                            nullptr,
                            false,
                            false,
                            0,
                            false
                        );
                    } else {
                        throw std::runtime_error("maint-bench requires --mode range|knn");
                    }
                });
            }

            std::vector<std::pair<bag::Edge, EdgeWeight>> move_slots;
            move_slots.reserve(graph.edge_count());
            for (const auto& [edge, weight] : graph.undirected_edges()) {
                if (weight > 1) {
                    move_slots.push_back({bag::ordered_edge(edge.first, edge.second), weight});
                }
            }
            if (move_slots.empty()) {
                throw std::runtime_error("graph does not have any valid move slots");
            }

            std::mt19937_64 rng(object_seed + 1);
            std::uniform_int_distribution<std::size_t> object_pick(0, object_count - 1);
            std::uniform_int_distribution<std::size_t> slot_pick(0, move_slots.size() - 1);

            const bool maint_verify = optional_bool(args, "verify", false);
            std::size_t maint_verify_checked = 0;
            std::size_t maint_verify_mismatches = 0;
            long long total_update_us = 0;
            long long total_query_us = 0;
            for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
                total_update_us += bag::calc_execution_time_in_us([&] {
                    for (std::size_t i = 0; i < change_count; ++i) {
                        const auto obj_index = object_pick(rng);
                        auto& object = active_objects[obj_index];
                        const auto& [edge, weight] = move_slots[slot_pick(rng)];
                        std::uniform_int_distribution<EdgeWeight> offset_dist(1, weight - 1);
                        object.edge = edge;
                        object.edge_weight = weight;
                        object.offset = offset_dist(rng);
                        indexed_objects.move_object(
                            object,
                            index.edge_to_subgraph(),
                            index.subgraphs()
                        );
                    }
                    indexed_objects.finalize_updates();
                });

                total_query_us += bag::calc_execution_time_in_us([&] {
                    for (std::size_t i = 0; i < query_per_update; ++i) {
                        if (mode == "range") {
                            (void)index.range_query(
                                queries[i % queries.size()],
                                range_radius,
                                indexed_objects,
                                fc_rule,
                                epoch * query_per_update + i,
                                nullptr,
                                nullptr,
                                true,
                                false
                            );
                        } else if (mode == "knn") {
                            (void)index.knn_query(
                                queries[i % queries.size()],
                                knn_k,
                                indexed_objects,
                                epoch * query_per_update + i,
                                nullptr,
                                nullptr,
                                false,
                                false,
                                0,
                                false
                            );
                        } else {
                            throw std::runtime_error("maint-bench requires --mode range|knn");
                        }
                    }
                });

                // After this epoch's moves, confirm the (default) sound query path
                // still agrees with whole-graph Dijkstra, exercising the
                // incremental metadata rebuild in finalize_updates().
                if (maint_verify) {
                    for (std::size_t i = 0; i < query_per_update; ++i) {
                        const auto& q = queries[i % queries.size()];
                        if (mode == "range") {
                            auto got = index.range_query(q, range_radius, indexed_objects, fc_rule,
                                                         0, nullptr, nullptr, true, false).object_ids;
                            auto ref = index.exact_range_query(q, range_radius, indexed_objects);
                            std::sort(got.begin(), got.end());
                            std::sort(ref.begin(), ref.end());
                            ++maint_verify_checked;
                            if (got != ref) ++maint_verify_mismatches;
                        } else {
                            const auto got = index.knn_query(q, knn_k, indexed_objects, 0, nullptr, nullptr,
                                                             false, false, 0, false, true).items;
                            const auto ref = index.exact_knn_query(q, knn_k, indexed_objects).items;
                            ++maint_verify_checked;
                            bool ok = got.size() == ref.size();
                            for (std::size_t j = 0; ok && j < got.size(); ++j) {
                                if (got[j].id != ref[j].id || got[j].distance != ref[j].distance) ok = false;
                            }
                            if (!ok) ++maint_verify_mismatches;
                        }
                    }
                }
            }

            const auto total_queries = epochs * query_per_update;
            std::cout
                << "{"
                << "\"graph_vertices\":" << graph.size() << ","
                << "\"subgraphs\":" << index.subgraphs().size() << ","
                << "\"partition_us\":" << partition_us << ","
                << "\"index_build_us\":" << index_build_us << ","
                << "\"mode\":\"" << mode << "\","
                << "\"car_percent\":" << car_percent << ","
                << "\"change_percent\":" << change_percent << ","
                << "\"object_count\":" << object_count << ","
                << "\"change_count\":" << change_count << ","
                << "\"query_per_update\":" << query_per_update << ","
                << "\"epochs\":" << epochs << ","
                << "\"range_radius\":" << range_radius << ","
                << "\"knn_k\":" << knn_k << ","
                << "\"initial_object_index_us\":" << initial_object_index_us << ","
                << "\"one_time_knn_avg_us\":"
                << (epochs == 0 ? 0.0 : static_cast<double>(one_time_knn_total_us) / static_cast<double>(epochs)) << ","
                << "\"avg_update_us\":"
                << (epochs == 0 ? 0.0 : static_cast<double>(total_update_us) / static_cast<double>(epochs)) << ","
                << "\"avg_query_us\":"
                << (total_queries == 0 ? 0.0 : static_cast<double>(total_query_us) / static_cast<double>(total_queries)) << ","
                << "\"amortized_us\":"
                << (total_queries == 0 ? 0.0 : static_cast<double>(total_update_us + total_query_us) / static_cast<double>(total_queries)) << ","
                << "\"verify\":" << (maint_verify ? "true" : "false") << ","
                << "\"verify_checked\":" << maint_verify_checked << ","
                << "\"verify_mismatches\":" << maint_verify_mismatches
                << "}\n";
            return (maint_verify_mismatches == 0) ? 0 : 2;
        }

        if (effective_command == "weight-audit") {
            const double weight_change_fraction = optional_double(args, "weight-change-fraction", 0.01);
            const double weight_scale = optional_double(args, "weight-scale", 1.25);
            const std::string weight_direction =
                args.contains("weight-direction") ? require_arg(args, "weight-direction") : "increase";
            const bool increase_only =
                weight_direction == "increase" || weight_direction == "up";
            const auto weight_seed =
                static_cast<std::uint64_t>(optional_usize(args, "weight-seed", 1));

            const auto base_skeleton = build_skeleton_graph_from_subgraphs(subgraphs);
            const auto base_summary = summarize_partition(graph, base_skeleton, subgraphs);

            long long reweight_us = 0;
            bag::Graph updated_graph;
            std::vector<bag::Subgraph> updated_subgraphs;
            reweight_us = bag::calc_execution_time_in_us([&] {
                updated_graph = rebuild_weighted_graph(
                    graph,
                    weight_change_fraction,
                    weight_scale,
                    increase_only,
                    weight_seed
                );
                updated_subgraphs = rebuild_partition_under_weights(updated_graph, subgraphs);
            });

            const auto updated_skeleton = build_skeleton_graph_from_subgraphs(updated_subgraphs);
            const auto updated_summary = summarize_partition(updated_graph, updated_skeleton, updated_subgraphs);

            std::cout
                << "{"
                << "\"weight_change_fraction\":" << weight_change_fraction << ","
                << "\"weight_scale\":" << weight_scale << ","
                << "\"weight_direction\":\"" << weight_direction << "\","
                << "\"weight_seed\":" << weight_seed << ","
                << "\"base_br_property_ok\":" << (base_summary.br_property_ok ? "true" : "false") << ","
                << "\"base_br_violating_subgraphs\":" << base_summary.br_violating_subgraphs << ","
                << "\"base_br_violating_boundaries\":" << base_summary.br_violating_boundaries << ","
                << "\"base_skeleton_vertices\":" << base_summary.skeleton_vertices << ","
                << "\"base_skeleton_edges\":" << base_summary.skeleton_edges << ","
                << "\"updated_br_property_ok\":" << (updated_summary.br_property_ok ? "true" : "false") << ","
                << "\"updated_br_violating_subgraphs\":" << updated_summary.br_violating_subgraphs << ","
                << "\"updated_br_violating_boundaries\":" << updated_summary.br_violating_boundaries << ","
                << "\"updated_skeleton_vertices\":" << updated_summary.skeleton_vertices << ","
                << "\"updated_skeleton_edges\":" << updated_summary.skeleton_edges << ","
                << "\"reweight_us\":" << reweight_us
                << "}\n";
            return 0;
        }

        long long index_build_us = 0;
        bag::SkeletonIndex index;
        index_build_us = bag::calc_execution_time_in_us([&] {
            index = SkeletonIndex::build(graph, std::move(subgraphs));
        });
        const bool factorized_transfer = optional_bool(args, "factorized-transfer", false);
        const double factorized_arc_threshold = optional_double(args, "factorized-arc-threshold", 0.5);
        const auto factorized_border_threshold = optional_usize(args, "factorized-border-threshold", 12);
        const bool gstar_shortcuts = optional_bool(args, "gstar-shortcuts", false);
        const auto gstar_leaf_hops = optional_usize(args, "gstar-leaf-hops", 2);
        const auto gstar_shortcuts_per_subgraph =
            optional_usize(args, "gstar-shortcuts-per-subgraph", 1);
        if (factorized_transfer) {
            index.configure_factorized_transfer(factorized_arc_threshold, factorized_border_threshold);
        }
        if (gstar_shortcuts) {
            (void)index.configure_gstar_shortcuts(
                gstar_leaf_hops,
                gstar_shortcuts_per_subgraph
            );
        }
        const auto query = load_query_point(args);
        if (effective_command == "frontier") {
            const auto radii = parse_weight_list(args, "radii", {1000, 2000, 4000, 8000, 16000, 32000});
            std::cout << "{"
                      << "\"query_u\":" << query.edge.first << ","
                      << "\"query_v\":" << query.edge.second << ","
                      << "\"offset\":" << query.offset << ","
                      << "\"radii\":[";
            for (std::size_t i = 0; i < radii.size(); ++i) {
                const auto stats = index.frontier_boundary_stats(query, radii[i]);
                if (i != 0) {
                    std::cout << ",";
                }
                std::cout << "{"
                          << "\"radius\":" << stats.radius << ","
                          << "\"reached_boundaries\":" << stats.reached_boundaries << ","
                          << "\"frontier_boundaries\":" << stats.frontier_boundaries << ","
                          << "\"outward_edges\":" << stats.outward_edges
                          << "}";
            }
            std::cout << "]}\n";
            return 0;
        }
        long long object_build_us = 0;
        IndexedMovingObjectSet objects;
        object_build_us = bag::calc_execution_time_in_us([&] {
            objects = build_objects(graph, index, args);
        });
        const auto repeat = optional_usize(args, "repeat", 1);

        if (effective_command == "probe") {
            const auto radius = require_weight(args, "radius");
            const auto k = require_usize(args, "k");
            const auto fc_rule = parse_fc_rule(args);
            const bool verify = optional_bool(args, "verify", false);
            const bool range_row_truncation = optional_bool(args, "range-row-truncation", true);
            const bool knn_streamed_clique = optional_bool(args, "knn-streamed-clique", false);
            const bool knn_safe_coverage_shadow = optional_bool(args, "knn-safe-coverage-shadow", false);
            const auto knn_parent_shadow_size = optional_usize(args, "knn-parent-shadow-size", 0);

            bag::RangeQueryResult range_result;
            std::vector<long long> range_times;
            range_times.reserve(repeat);
            for (std::size_t i = 0; i < repeat; ++i) {
                range_times.push_back(bag::calc_execution_time_in_us([&] {
                    range_result = index.range_query(
                        query,
                        radius,
                        objects,
                        fc_rule,
                        0,
                        nullptr,
                        nullptr,
                        range_row_truncation,
                        factorized_transfer
                    );
                }));
            }
            const auto range_us = *std::min_element(range_times.begin(), range_times.end());
            const auto avg_range_us =
                static_cast<double>(std::accumulate(range_times.begin(), range_times.end(), 0LL)) /
                static_cast<double>(range_times.size());

            bag::KnnQueryResult knn_result;
            std::vector<long long> knn_times;
            knn_times.reserve(repeat);
            for (std::size_t i = 0; i < repeat; ++i) {
                knn_times.push_back(bag::calc_execution_time_in_us([&] {
                    knn_result = index.knn_query(
                        query,
                        k,
                        objects,
                        0,
                        nullptr,
                        nullptr,
                        knn_streamed_clique,
                        knn_safe_coverage_shadow,
                        knn_parent_shadow_size,
                        factorized_transfer
                    );
                }));
            }
            const auto knn_us = *std::min_element(knn_times.begin(), knn_times.end());
            const auto avg_knn_us =
                static_cast<double>(std::accumulate(knn_times.begin(), knn_times.end(), 0LL)) /
                static_cast<double>(knn_times.size());

            bool range_exact = true;
            bool knn_exact = true;
            long long range_exact_us = 0;
            long long knn_exact_us = 0;
            if (verify) {
                std::vector<bag::ObjId> exact_range_ids;
                range_exact_us = bag::calc_execution_time_in_us([&] {
                    exact_range_ids = index.exact_range_query(query, radius, objects);
                });
                std::unordered_set<bag::ObjId> lhs(range_result.object_ids.begin(), range_result.object_ids.end());
                std::unordered_set<bag::ObjId> rhs(exact_range_ids.begin(), exact_range_ids.end());
                range_exact = lhs == rhs;

                bag::KnnQueryResult exact_knn_result;
                knn_exact_us = bag::calc_execution_time_in_us([&] {
                    exact_knn_result = index.exact_knn_query(query, k, objects);
                });
                if (knn_result.items.size() != exact_knn_result.items.size()) {
                    knn_exact = false;
                } else {
                    for (std::size_t i = 0; i < knn_result.items.size(); ++i) {
                        if (knn_result.items[i].id != exact_knn_result.items[i].id ||
                            knn_result.items[i].distance != exact_knn_result.items[i].distance) {
                            knn_exact = false;
                            break;
                        }
                    }
                }
            }

            std::cout
                << "{"
                << "\"subgraphs\":" << index.subgraphs().size() << ","
                << "\"partition_us\":" << partition_us << ","
                << "\"index_build_us\":" << index_build_us << ","
                << "\"object_build_us\":" << object_build_us << ","
                << "\"gstar_shortcuts\":" << (gstar_shortcuts ? "true" : "false") << ","
                << "\"gstar_shortcut_edges_added\":" << index.gstar_shortcut_edges_added() << ","
                << "\"repeat\":" << repeat << ","
                << "\"range\":{"
                << "\"hits\":" << range_result.object_ids.size() << ","
                << "\"touched_subgraphs\":" << range_result.touched_subgraphs << ","
                << "\"boundary_vertices_reached\":" << range_result.boundary_vertices_reached << ","
                << "\"auto_included_objects\":" << range_result.auto_included_objects << ","
                << "\"br_fc_included_objects\":" << range_result.br_fc_included_objects << ","
                << "\"partial_edge_auto_included_objects\":"
                << range_result.partial_edge_auto_included_objects << ","
                << "\"exact_returned_objects\":" << range_result.exact_returned_objects << ","
                << "\"exact_checked_objects\":" << range_result.exact_checked_objects << ","
                << "\"br_fc_coverage_ratio\":"
                << (range_result.object_ids.empty()
                        ? 0.0
                        : static_cast<double>(range_result.br_fc_included_objects) /
                              static_cast<double>(range_result.object_ids.size())) << ","
                << "\"init_us\":" << range_result.init_us << ","
                << "\"skeleton_trace_us\":" << range_result.skeleton_trace_us << ","
                << "\"subgraph_eval_us\":" << range_result.subgraph_eval_us << ","
                << "\"num_clique_relax_attempts\":" << range_result.num_clique_relax_attempts << ","
                << "\"num_pq_pushes_from_clique\":" << range_result.num_pq_pushes_from_clique << ","
                << "\"range_us\":" << range_us << ","
                << "\"avg_range_us\":" << avg_range_us << ","
                << "\"exact_us\":" << range_exact_us << ","
                << "\"exact\":" << (range_exact ? "true" : "false")
                << "},"
                << "\"knn\":{"
                << "\"returned\":" << knn_result.items.size() << ","
                << "\"visited_subgraphs\":" << knn_result.visited_subgraphs << ","
                << "\"visited_boundaries\":" << knn_result.visited_boundaries << ","
                << "\"init_us\":" << knn_result.init_us << ","
                << "\"explore_us\":" << knn_result.explore_us << ","
                << "\"finalize_us\":" << knn_result.finalize_us << ","
                << "\"pq_us\":" << knn_result.pq_us << ","
                << "\"membership_us\":" << knn_result.membership_us << ","
                << "\"subgraph_bookkeeping_us\":" << knn_result.subgraph_bookkeeping_us << ","
                << "\"clique_emit_us\":" << knn_result.clique_emit_us << ","
                << "\"num_clique_relax_attempts\":" << knn_result.num_clique_relax_attempts << ","
                << "\"num_pq_pushes_from_clique\":" << knn_result.num_pq_pushes_from_clique << ","
                << "\"exact_evaluated\":" << knn_result.exact_evaluated << ","
                << "\"knn_us\":" << knn_us << ","
                << "\"avg_knn_us\":" << avg_knn_us << ","
                << "\"exact_us\":" << knn_exact_us << ","
                << "\"exact\":" << (knn_exact ? "true" : "false")
                << "}"
                << "}\n";
            return (range_exact && knn_exact) ? 0 : 2;
        }

        if (effective_command == "range") {
            const auto radius = require_weight(args, "radius");
            const auto fc_rule = parse_fc_rule(args);
            const bool range_row_truncation = optional_bool(args, "range-row-truncation", true);
            const bool compare_baseline = optional_bool(args, "compare-baseline", false);
            long long baseline_range_us = 0;
            bag::RangeQueryResult result;
            std::vector<long long> range_times;
            range_times.reserve(repeat);
            for (std::size_t i = 0; i < repeat; ++i) {
                range_times.push_back(bag::calc_execution_time_in_us([&] {
                    result = index.range_query(
                        query,
                        radius,
                        objects,
                        fc_rule,
                        0,
                        nullptr,
                        nullptr,
                        range_row_truncation,
                        factorized_transfer
                    );
                }));
            }
            const auto range_us = *std::min_element(range_times.begin(), range_times.end());
            const auto avg_range_us =
                static_cast<double>(std::accumulate(range_times.begin(), range_times.end(), 0LL)) /
                static_cast<double>(range_times.size());
            bool baseline_match = true;
            if (compare_baseline) {
                bag::RangeQueryResult baseline;
                baseline_range_us = bag::calc_execution_time_in_us([&] {
                    baseline = index.range_query(query, radius, objects, fc_rule, 0, nullptr, nullptr, false, false);
                });
                baseline_match = baseline.object_ids == result.object_ids;
            }
            const bool verify = optional_bool(args, "verify", false);
            bool exact = true;
            long long exact_us = 0;
            if (verify) {
                std::vector<bag::ObjId> exact_ids;
                exact_us = bag::calc_execution_time_in_us([&] {
                    exact_ids = index.exact_range_query(query, radius, objects);
                });
                std::unordered_set<bag::ObjId> lhs(result.object_ids.begin(), result.object_ids.end());
                std::unordered_set<bag::ObjId> rhs(exact_ids.begin(), exact_ids.end());
                exact = lhs == rhs;
            }
            std::cout
                << "{"
                << "\"hits\":" << result.object_ids.size() << ","
                << "\"fc_subgraphs\":" << result.fc_subgraphs << ","
                << "\"pc_subgraphs\":" << result.pc_subgraphs << ","
                << "\"auto_included_objects\":" << result.auto_included_objects << ","
                << "\"br_fc_included_objects\":" << result.br_fc_included_objects << ","
                << "\"partial_edge_auto_included_objects\":"
                << result.partial_edge_auto_included_objects << ","
                << "\"exact_returned_objects\":" << result.exact_returned_objects << ","
                << "\"exact_checked_objects\":" << result.exact_checked_objects << ","
                << "\"br_fc_coverage_ratio\":"
                << (result.object_ids.empty()
                        ? 0.0
                        : static_cast<double>(result.br_fc_included_objects) /
                              static_cast<double>(result.object_ids.size())) << ","
                << "\"initial_local_vertices\":" << result.initial_local_vertices << ","
                << "\"boundary_vertices_reached\":" << result.boundary_vertices_reached << ","
                << "\"touched_subgraphs\":" << result.touched_subgraphs << ","
                << "\"num_clique_relax_attempts\":" << result.num_clique_relax_attempts << ","
                << "\"num_successful_clique_relaxes\":" << result.num_successful_clique_relaxes << ","
                << "\"num_pq_pushes_from_clique\":" << result.num_pq_pushes_from_clique << ","
                << "\"num_rows_truncated\":" << result.num_rows_truncated << ","
                << "\"num_exits_skipped_by_truncation\":" << result.num_exits_skipped_by_truncation << ","
                << "\"num_redundant_subgraph_reentries\":" << result.num_redundant_subgraph_reentries << ","
                << "\"num_useful_subgraph_reentries\":" << result.num_useful_subgraph_reentries << ","
                << "\"factorized_rows_used\":" << result.factorized_rows_used << ","
                << "\"factorized_hubs_used\":" << result.factorized_hubs_used << ","
                << "\"factorized_exits_emitted\":" << result.factorized_exits_emitted << ","
                << "\"repeat\":" << repeat << ","
                << "\"range_us\":" << range_us << ","
                << "\"avg_range_us\":" << avg_range_us << ","
                << "\"baseline_range_us\":" << baseline_range_us << ","
                << "\"exact_us\":" << exact_us << ","
                << "\"exact\":" << (exact ? "true" : "false") << ","
                << "\"baseline_match\":" << (baseline_match ? "true" : "false") << ","
                << "\"range_row_truncation\":" << (range_row_truncation ? "true" : "false") << ","
                << "\"factorized_transfer\":" << (factorized_transfer ? "true" : "false")
                << "}\n";
            return (exact && baseline_match) ? 0 : 2;
        }

        if (effective_command == "knn") {
            const auto k = require_usize(args, "k");
            const bool knn_streamed_clique = optional_bool(args, "knn-streamed-clique", false);
            const bool knn_safe_coverage_shadow = optional_bool(args, "knn-safe-coverage-shadow", false);
            const auto knn_parent_shadow_size = optional_usize(args, "knn-parent-shadow-size", 0);
            const bool compare_baseline = optional_bool(args, "compare-baseline", false);
            long long baseline_knn_us = 0;
            bag::KnnQueryResult result;
            std::vector<long long> knn_times;
            knn_times.reserve(repeat);
            for (std::size_t i = 0; i < repeat; ++i) {
                knn_times.push_back(bag::calc_execution_time_in_us([&] {
                    result = index.knn_query(
                        query,
                        k,
                        objects,
                        0,
                        nullptr,
                        nullptr,
                        knn_streamed_clique,
                        knn_safe_coverage_shadow,
                        knn_parent_shadow_size,
                        factorized_transfer
                    );
                }));
            }
            const auto knn_us = *std::min_element(knn_times.begin(), knn_times.end());
            const auto avg_knn_us =
                static_cast<double>(std::accumulate(knn_times.begin(), knn_times.end(), 0LL)) /
                static_cast<double>(knn_times.size());
            bool baseline_match = true;
            if (compare_baseline) {
                bag::KnnQueryResult baseline;
                baseline_knn_us = bag::calc_execution_time_in_us([&] {
                    baseline = index.knn_query(
                        query,
                        k,
                        objects,
                        0,
                        nullptr,
                        nullptr,
                        false,
                        knn_safe_coverage_shadow,
                        knn_parent_shadow_size,
                        false
                    );
                });
                if (result.items.size() != baseline.items.size()) {
                    baseline_match = false;
                } else {
                    for (std::size_t i = 0; i < result.items.size(); ++i) {
                        if (result.items[i].id != baseline.items[i].id ||
                            result.items[i].distance != baseline.items[i].distance) {
                            baseline_match = false;
                            break;
                        }
                    }
                }
            }
            const bool verify = optional_bool(args, "verify", false);
            bool exact = true;
            long long exact_us = 0;
            if (verify) {
                bag::KnnQueryResult exact_result;
                exact_us = bag::calc_execution_time_in_us([&] {
                    exact_result = index.exact_knn_query(query, k, objects);
                });
                if (result.items.size() != exact_result.items.size()) {
                    exact = false;
                } else {
                    for (std::size_t i = 0; i < result.items.size(); ++i) {
                        if (result.items[i].id != exact_result.items[i].id ||
                            result.items[i].distance != exact_result.items[i].distance) {
                            exact = false;
                            break;
                        }
                    }
                }
            }
            std::cout
                << "{"
                << "\"k\":" << k << ","
                << "\"returned\":" << result.items.size() << ","
                << "\"visited_boundaries\":" << result.visited_boundaries << ","
                << "\"visited_subgraphs\":" << result.visited_subgraphs << ","
                << "\"candidates_considered\":" << result.candidates_considered << ","
                << "\"final_candidates\":" << result.final_candidates << ","
                << "\"exact_evaluated\":" << result.exact_evaluated << ","
                << "\"vertex_fast_path\":" << (result.vertex_fast_path ? "true" : "false") << ","
                << "\"init_us\":" << result.init_us << ","
                << "\"explore_us\":" << result.explore_us << ","
                << "\"finalize_us\":" << result.finalize_us << ","
                << "\"pq_us\":" << result.pq_us << ","
                << "\"membership_us\":" << result.membership_us << ","
                << "\"subgraph_bookkeeping_us\":" << result.subgraph_bookkeeping_us << ","
                << "\"clique_emit_us\":" << result.clique_emit_us << ","
                << "\"num_clique_relax_attempts\":" << result.num_clique_relax_attempts << ","
                << "\"num_successful_clique_relaxes\":" << result.num_successful_clique_relaxes << ","
                << "\"num_pq_pushes_from_clique\":" << result.num_pq_pushes_from_clique << ","
                << "\"num_redundant_subgraph_reentries\":" << result.num_redundant_subgraph_reentries << ","
                << "\"num_useful_subgraph_reentries\":" << result.num_useful_subgraph_reentries << ","
                << "\"boundary_pq_pushes\":" << result.boundary_pq_pushes << ","
                << "\"boundary_pq_pops\":" << result.boundary_pq_pops << ","
                << "\"row_state_pushes\":" << result.row_state_pushes << ","
                << "\"row_state_pops\":" << result.row_state_pops << ","
                << "\"streamed_rows_started\":" << result.streamed_rows_started << ","
                << "\"streamed_row_emissions\":" << result.streamed_row_emissions << ","
                << "\"streamed_rows_stopped_by_threshold\":" << result.streamed_rows_stopped_by_threshold << ","
                << "\"streamed_exits_skipped_by_threshold\":" << result.streamed_exits_skipped_by_threshold << ","
                << "\"peak_frontier_queue_size\":" << result.peak_frontier_queue_size << ","
                << "\"safe_coverage_candidate_count\":" << result.safe_coverage_candidate_count << ","
                << "\"first_k_candidates_boundary_visit_order\":" << result.first_k_candidates_boundary_visit_order << ","
                << "\"first_finite_safe_coverage_radius\":" << result.first_finite_safe_coverage_radius << ","
                << "\"first_finite_safe_coverage_boundary_visit_order\":" << result.first_finite_safe_coverage_boundary_visit_order << ","
                << "\"final_safe_coverage_radius\":" << result.final_safe_coverage_radius << ","
                << "\"safe_coverage_updates\":" << result.safe_coverage_updates << ","
                << "\"first_finite_tau\":" << result.first_finite_tau << ","
                << "\"first_finite_tau_boundary_visit_order\":" << result.first_finite_tau_boundary_visit_order << ","
                << "\"final_tau\":" << result.final_tau << ","
                << "\"tau_updates\":" << result.tau_updates << ","
                << "\"would_stop_rows_by_safe_coverage\":" << result.would_stop_rows_by_safe_coverage << ","
                << "\"would_skip_exits_by_safe_coverage\":" << result.would_skip_exits_by_safe_coverage << ","
                << "\"would_save_pq_pushes_by_safe_coverage\":" << result.would_save_pq_pushes_by_safe_coverage << ","
                << "\"parent_regions_touched\":" << result.parent_regions_touched << ","
                << "\"parent_region_size\":" << result.parent_region_size << ","
                << "\"max_children_descended_in_parent\":" << result.max_children_descended_in_parent << ","
                << "\"single_child_parent_regions\":" << result.single_child_parent_regions << ","
                << "\"avg_children_descended_per_parent\":" << result.avg_children_descended_per_parent << ","
                << "\"parent_descent_ratio\":" << result.parent_descent_ratio << ","
                << "\"factorized_rows_used\":" << result.factorized_rows_used << ","
                << "\"factorized_hubs_used\":" << result.factorized_hubs_used << ","
                << "\"factorized_exits_emitted\":" << result.factorized_exits_emitted << ","
                << "\"repeat\":" << repeat << ","
                << "\"knn_us\":" << knn_us << ","
                << "\"avg_knn_us\":" << avg_knn_us << ","
                << "\"baseline_knn_us\":" << baseline_knn_us << ","
                << "\"exact_us\":" << exact_us << ","
                << "\"exact\":" << (exact ? "true" : "false") << ","
                << "\"baseline_match\":" << (baseline_match ? "true" : "false") << ","
                << "\"streamed_clique\":" << (result.streamed_clique ? "true" : "false") << ","
                << "\"safe_coverage_shadow\":" << (knn_safe_coverage_shadow ? "true" : "false") << ","
                << "\"factorized_transfer\":" << (factorized_transfer ? "true" : "false")
                << "}\n";
            return (exact && baseline_match) ? 0 : 2;
        }

        print_usage();
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
