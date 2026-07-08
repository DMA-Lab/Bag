#include "loader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bag {

Graph load_graph_from_file(const std::string& file) {
    std::ifstream in(file);
    if (!in) {
        throw std::runtime_error("failed to open graph file: " + file);
    }

    Graph graph;
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
            std::string kind;
            std::size_t vertex_count = 0;
            std::size_t edge_count = 0;
            ss >> kind >> vertex_count >> edge_count;
            (void)kind;
            (void)vertex_count;
            (void)edge_count;
            continue;
        }
        if (op != 'a') {
            throw std::runtime_error("unknown line in graph file: " + line);
        }

        VertexId source = 0;
        VertexId destination = 0;
        EdgeWeight weight = 0;
        ss >> source >> destination >> weight;
        graph.add_directed_edge(source, destination, weight);
    }

    return graph;
}

}  // namespace bag
