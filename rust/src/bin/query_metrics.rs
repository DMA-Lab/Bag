use std::time::Instant;

use bap::graph::VertexId;
use bap::index::SkeletonGraphBuilder;
use bap::loader::{load_graph, load_or_generate_moving_objects, load_or_partition};
use bap::object::{IndexedMovingObjectSet, MovingObject};
use bap::partition::VfipPartition;
use clap::Parser;
use serde::Serialize;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    mode: String,

    #[arg(long)]
    path: String,

    #[arg(long)]
    theta: usize,

    #[arg(long, default_value = "1")]
    seed: u32,

    #[arg(long = "query-u")]
    query_u: u32,

    #[arg(long = "query-v", default_value = "0")]
    query_v: u32,

    #[arg(long, default_value = "0")]
    offset: u64,

    #[arg(long, default_value = "0")]
    radius: u64,

    #[arg(long, default_value = "0")]
    k: usize,

    #[arg(long)]
    objects: usize,

    #[arg(long, default_value = "5")]
    repeat: usize,
}

#[derive(Debug, Serialize)]
struct QueryMetrics {
    mode: String,
    dataset: String,
    theta: usize,
    seed: u32,
    query_u: u32,
    query_v: u32,
    offset: u64,
    effective_query_u: u32,
    effective_query_v: u32,
    effective_offset: u64,
    radius: u64,
    k: usize,
    objects: usize,
    repeat: usize,
    result_count: usize,
    min_query_us: u128,
    avg_query_us: f64,
}

fn resolve_query_vertex(
    graph: &bap::graph::Graph,
    subgraphs: &[bap::partition::Subgraph],
    requested_query_u: VertexId,
    requested_query_v: VertexId,
    requested_offset: u64,
) -> (VertexId, VertexId, u64) {
    if requested_query_u != 0 {
        let query_v = if requested_query_v != 0 {
            requested_query_v
        } else {
            graph
                .get_out_adjacent_edges(&requested_query_u)
                .map(|(v, _)| v)
                .min()
                .expect("query-u has no adjacent edge")
        };
        return (requested_query_u, query_v, requested_offset);
    }

    for sg in subgraphs {
        let mut internal_vertices = sg.internal_vertices.iter().collect::<Vec<_>>();
        internal_vertices.sort_unstable();
        for v in internal_vertices {
            if let Some(query_v) = graph
                .get_out_adjacent_edges(&v)
                .map(|(u, _)| u)
                .min()
            {
                return (v, query_v, 0);
            }
        }
    }

    panic!("failed to find an internal vertex with an adjacent edge");
}

fn main() {
    let args = Args::parse();

    let graph = load_graph(&args.path).expect("failed to load graph");
    let subgraphs = load_or_partition(
        &args.path,
        &graph,
        args.seed as VertexId,
        args.theta,
        |graph, theta, start| VfipPartition::new(graph, theta).run(start),
    );
    let skeleton = SkeletonGraphBuilder::default()
        .subgraphs(&subgraphs)
        .global_graph(&graph)
        .build();
    let (effective_query_u, effective_query_v, effective_offset) = resolve_query_vertex(
        &graph,
        &subgraphs,
        args.query_u,
        args.query_v,
        args.offset,
    );

    let raw_objects = load_or_generate_moving_objects(&args.path, &graph, args.objects);
    let objects = IndexedMovingObjectSet::new(&subgraphs, raw_objects);

    let mut times = Vec::with_capacity(args.repeat);
    let mut result_count = 0usize;

    match args.mode.as_str() {
        "range" => {
            let query = MovingObject::query_point((effective_query_u, effective_query_v), effective_offset);
            for _ in 0..args.repeat {
                let started = Instant::now();
                let result = {
                    let mut q = skeleton.scan((query, args.radius), &objects);
                    q.results()
                };
                let elapsed = started.elapsed().as_micros();
                result_count = result.len();
                times.push(elapsed);
            }
        }
        "knn" => {
            if effective_offset != 0 {
                panic!("Rust kNN benchmark currently supports only vertex queries; use --offset 0");
            }
            for _ in 0..args.repeat {
                let started = Instant::now();
                let result = skeleton.knn_query(&objects, effective_query_u, args.k).run();
                let elapsed = started.elapsed().as_micros();
                result_count = result.len();
                times.push(elapsed);
            }
        }
        _ => panic!("unsupported --mode, expected range|knn"),
    }

    let min_query_us = *times.iter().min().unwrap_or(&0);
    let avg_query_us = if times.is_empty() {
        0.0
    } else {
        times.iter().sum::<u128>() as f64 / times.len() as f64
    };

    let metrics = QueryMetrics {
        mode: args.mode,
        dataset: args.path,
        theta: args.theta,
        seed: args.seed,
        query_u: args.query_u,
        query_v: args.query_v,
        offset: args.offset,
        effective_query_u,
        effective_query_v,
        effective_offset,
        radius: args.radius,
        k: args.k,
        objects: args.objects,
        repeat: args.repeat,
        result_count,
        min_query_us,
        avg_query_us,
    };

    println!("{}", serde_json::to_string(&metrics).unwrap());
}
