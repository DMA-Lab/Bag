use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use bap::graph::{EdgeWeight, MayHalfPathWeight, VertexId, INF_WEIGHT};
use bap::loader::load_graph;
use bap::pair::pair_in_order;
use bap::partition::VfipPartition;
use clap::Parser;
use serde::Serialize;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    path: String,

    #[arg(long)]
    theta: usize,

    #[arg(long, default_value = "1")]
    seed: u32,
}

#[derive(Debug, Serialize)]
struct PartitionMetrics {
    subgraphs: usize,
    graph_vertices: usize,
    graph_edges: usize,
    skeleton_vertices: usize,
    unique_boundary_vertices: usize,
    skeleton_edges: usize,
    skeleton_avg_degree: f64,
    total_subgraph_vertices: usize,
    total_boundary_vertices: usize,
    total_internal_vertices: usize,
    avg_boundary_per_subgraph: f64,
    max_boundary_vertices: usize,
    br_property_ok: bool,
    br_violating_subgraphs: usize,
    br_violating_boundaries: usize,
    vertex_duplication_factor: f64,
    unique_boundary_ratio: f64,
    subgraph_size_histogram: BTreeMap<usize, usize>,
    partition_seed: u32,
    theta: usize,
    partition_us: u128,
}

fn distal_point_distance(lhs: EdgeWeight, rhs: EdgeWeight, edge_weight: EdgeWeight) -> MayHalfPathWeight {
    let sum = lhs + rhs + edge_weight;
    (sum >> 1, sum & 1 != 0)
}

fn main() {
    let args = Args::parse();

    let graph = load_graph(&args.path).expect("failed to load graph");
    let started = Instant::now();
    let subgraphs = VfipPartition::new(&graph, args.theta).run(args.seed);
    let partition_us = started.elapsed().as_micros();

    let mut unique_boundary_vertices = HashSet::new();
    let mut unique_skeleton_edges = HashSet::new();
    let mut subgraph_size_histogram = BTreeMap::new();
    let mut total_subgraph_vertices = 0usize;
    let mut total_boundary_vertices = 0usize;
    let mut total_internal_vertices = 0usize;
    let mut max_boundary_vertices = 0usize;
    let mut br_violating_subgraphs = 0usize;
    let mut br_violating_boundaries = 0usize;

    for sg in &subgraphs {
        *subgraph_size_histogram.entry(sg.graph.size()).or_insert(0) += 1;
        total_subgraph_vertices += sg.graph.size();
        total_boundary_vertices += sg.bound_vertices.len();
        total_internal_vertices += sg.internal_vertices.len();
        max_boundary_vertices = max_boundary_vertices.max(sg.bound_vertices.len());

        let mut subgraph_ok = true;
        for b in sg.bound_vertices.iter() {
            let rb = sg
                .bound_vertices
                .iter()
                .map(|other| (sg.distance[(b, other)], false))
                .max()
                .unwrap_or((0, false));

            let mut ib: MayHalfPathWeight = sg
                .internal_vertices
                .iter()
                .map(|v| (sg.distance[(b, v)], false))
                .max()
                .unwrap_or((0, false));

            for ((u, v), w) in sg.graph.edges().filter(|((u, v), _)| u < v) {
                let lhs = sg.distance.get_or_inf(b, u);
                let rhs = sg.distance.get_or_inf(b, v);
                if lhs == INF_WEIGHT || rhs == INF_WEIGHT {
                    continue;
                }
                ib = ib.max(distal_point_distance(lhs, rhs, w));
            }

            if rb < ib {
                br_violating_boundaries += 1;
                subgraph_ok = false;
            }
        }
        if !subgraph_ok {
            br_violating_subgraphs += 1;
        }

        for b in sg.bound_vertices.iter() {
            unique_boundary_vertices.insert(b);
        }

        for b in sg.bound_vertices.iter() {
            for (other_b, _dist) in sg.distance.distance_from(b) {
                if b < other_b && sg.bound_vertices.contains(&other_b) {
                    unique_skeleton_edges.insert(pair_in_order((b, other_b)));
                }
            }
        }
    }

    let skeleton_vertices = unique_boundary_vertices.len();
    let skeleton_edges = unique_skeleton_edges.len();
    let avg_boundary_per_subgraph = if subgraphs.is_empty() {
        0.0
    } else {
        total_boundary_vertices as f64 / subgraphs.len() as f64
    };
    let vertex_duplication_factor = if graph.size() == 0 {
        0.0
    } else {
        total_subgraph_vertices as f64 / graph.size() as f64
    };
    let unique_boundary_ratio = if graph.size() == 0 {
        0.0
    } else {
        skeleton_vertices as f64 / graph.size() as f64
    };
    let skeleton_avg_degree = if skeleton_vertices == 0 {
        0.0
    } else {
        2.0 * skeleton_edges as f64 / skeleton_vertices as f64
    };

    let metrics = PartitionMetrics {
        subgraphs: subgraphs.len(),
        graph_vertices: graph.size(),
        graph_edges: graph.edge_count(),
        skeleton_vertices,
        unique_boundary_vertices: skeleton_vertices,
        skeleton_edges,
        skeleton_avg_degree,
        total_subgraph_vertices,
        total_boundary_vertices,
        total_internal_vertices,
        avg_boundary_per_subgraph,
        max_boundary_vertices,
        br_property_ok: br_violating_boundaries == 0,
        br_violating_subgraphs,
        br_violating_boundaries,
        vertex_duplication_factor,
        unique_boundary_ratio,
        subgraph_size_histogram,
        partition_seed: args.seed,
        theta: args.theta,
        partition_us,
    };

    println!("{}", serde_json::to_string(&metrics).unwrap());
}
