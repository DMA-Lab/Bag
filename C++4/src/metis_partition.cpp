#include "metis_partition.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace bag {

namespace {

std::unordered_map<VertexId, std::size_t> load_assignment_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open metis assignment csv");
    }
    std::unordered_map<VertexId, std::size_t> assignment;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (first) {
            first = false;
            if (line.find("vertex_id") != std::string::npos) {
                continue;
            }
        }
        std::stringstream ss(line);
        std::string lhs;
        std::string rhs;
        if (!std::getline(ss, lhs, ',')) {
            continue;
        }
        if (!std::getline(ss, rhs, ',')) {
            continue;
        }
        assignment[static_cast<VertexId>(std::stoul(lhs))] = static_cast<std::size_t>(std::stoull(rhs));
    }
    return assignment;
}

void fill_internal_to_nearest_border(Subgraph& sg) {
    sg.internal_to_nearest_border_dist.clear();
    for (const auto v : sg.internal_vertices) {
        EdgeWeight best = kInfWeight;
        for (const auto b : sg.bound_vertices) {
            best = std::min(best, sg.distance.get_or_inf(v, b));
        }
        sg.internal_to_nearest_border_dist[v] = best;
    }
}

void compute_rb_only(Subgraph& sg) {
    sg.rb_map.clear();
    for (const auto b : sg.bound_vertices) {
        HalfWeight rb{0, false};
        for (const auto other : sg.bound_vertices) {
            rb = std::max(rb, HalfWeight{sg.distance.get_or_inf(b, other), false});
        }
        sg.rb_map[b] = rb;
    }
}

}  // namespace

std::vector<Subgraph> build_subgraphs_from_metis_assignment(
    const Graph& global,
    const MetisImportOptions& options
) {
    const auto assignment = load_assignment_csv(options.assignment_csv);
    if (assignment.empty()) {
        throw std::runtime_error("empty metis assignment");
    }

    std::size_t part_count = 0;
    for (const auto& [vertex, part] : assignment) {
        (void)vertex;
        part_count = std::max(part_count, part + 1U);
    }

    std::vector<Subgraph> subgraphs(part_count);
    std::vector<std::size_t> owned_edge_count(part_count, 0U);
    std::unordered_map<VertexId, std::size_t> cut_degree;
    for (std::size_t part = 0; part < part_count; ++part) {
        subgraphs[part].id = part;
    }

    for (const auto& [edge, weight] : global.undirected_edges()) {
        (void)weight;
        const auto part_u_it = assignment.find(edge.first);
        const auto part_v_it = assignment.find(edge.second);
        if (part_u_it == assignment.end() || part_v_it == assignment.end()) {
            continue;
        }
        if (part_u_it->second != part_v_it->second) {
            ++cut_degree[edge.first];
            ++cut_degree[edge.second];
        }
    }

    for (const auto& [edge, weight] : global.undirected_edges()) {
        const auto part_u_it = assignment.find(edge.first);
        const auto part_v_it = assignment.find(edge.second);
        if (part_u_it == assignment.end() || part_v_it == assignment.end()) {
            continue;
        }
        const auto part_u = part_u_it->second;
        const auto part_v = part_v_it->second;
        std::size_t owner = part_u;
        if (part_u != part_v) {
            const auto cut_u = cut_degree.contains(edge.first) ? cut_degree[edge.first] : 0U;
            const auto cut_v = cut_degree.contains(edge.second) ? cut_degree[edge.second] : 0U;
            if (cut_u < cut_v) {
                owner = part_u;
            } else if (cut_v < cut_u) {
                owner = part_v;
            } else {
                owner = (owned_edge_count[part_u] <= owned_edge_count[part_v]) ? part_u : part_v;
            }
        }
        auto& sg = subgraphs[owner];
        sg.graph.set_min_undirected_edge(edge.first, edge.second, weight);
        ++owned_edge_count[owner];
    }

    for (const auto& [vertex, part] : assignment) {
        subgraphs[part].graph.insert(vertex);
        if (subgraphs[part].seed_vertex == kInvalidVertex || vertex < subgraphs[part].seed_vertex) {
            subgraphs[part].seed_vertex = vertex;
        }
    }

    for (auto& sg : subgraphs) {
        const auto vertices = sg.graph.vertices();
        for (const auto v : vertices) {
            if (sg.check_internal_vertex(global, v)) {
                sg.internal_vertices.insert(v);
            } else {
                sg.bound_vertices.insert(v);
            }
        }
        sg.distance = all_pairs_shortest_paths(sg.graph);
        if (options.rb_only_mode) {
            compute_rb_only(sg);
        }
        fill_internal_to_nearest_border(sg);
        sg.insertion_order = vertices;
    }

    return subgraphs;
}

}  // namespace bag
