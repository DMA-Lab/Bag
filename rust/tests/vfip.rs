use std::collections::{BTreeMap, HashSet};

use bap::calc_time;
use bap::graph::VertexId;
use bap::loader::load_graph;
use bap::partition::VfipPartition;

fn count_by_group(
    iter: impl Iterator<Item=usize>,
    group_size: usize,
) -> Vec<((usize, usize), usize)> {
    use std::collections::btree_map::Entry;

    let mut result = BTreeMap::<usize, usize>::new();
    iter.for_each(|value| {
        let key: usize = value / group_size;

        match result.entry(key) {
            Entry::Occupied(e) => {
                *e.into_mut() += 1;
            }
            Entry::Vacant(e) => {
                e.insert(1);
            }
        };
    });

    result
        .into_iter()
        .map(|(start, count)| ((start * group_size, start * group_size + group_size), count))
        .collect()
}

fn test_and_count(path: &str, (seed, theta): (VertexId, usize)) {
    let graph =
        calc_time("load_graph", || load_graph(path)).expect("unable to load graph file.");
    let result = calc_time("VFIP", || {
        let vfip = VfipPartition::new(&graph, theta);
        vfip.run(seed)
    });

    let graph_size_list: Vec<_> = result.iter().map(|sg| sg.graph.size()).collect();
    let avg_size = graph_size_list.iter().sum::<usize>() / graph_size_list.len();

    println!("count: {}", graph_size_list.len());
    println!("avg size: {avg_size}");
    let distribution = count_by_group(graph_size_list.into_iter(), 10);
    println!("{:?}", distribution);

    let mut deduped_bound_vertex_set = HashSet::new();
    result
        .iter()
        .for_each(|sg| deduped_bound_vertex_set.extend(sg.bound_vertices.iter()));

    println!(
        "count of bound vertices: {}",
        deduped_bound_vertex_set.len()
    );
}

#[test]
fn test_graph_on_paper() {
    let path = "../dataset/on-paper.gr";

    test_and_count(path, (1, 8));
}

#[test]
fn test_graph_231028() {
    let path = "../dataset/test-231028.gr";

    test_and_count(path, (1, 8));
}

#[test]
fn test_graph_new_york_size_30() {
    let path = "../dataset/USA-road-d.NY.gr";

    test_and_count(path, (1, 30));
}

#[test]
fn test_graph_new_york_size_40() {
    let path = "../dataset/USA-road-d.NY.gr";

    test_and_count(path, (1, 40));
}

#[test]
fn test_graph_new_york_size_50() {
    let path = "../dataset/USA-road-d.NY.gr";

    test_and_count(path, (1, 50));
}

#[test]
fn test_graph_new_york_size_100() {
    let path = "../dataset/USA-road-d.NY.gr";

    test_and_count(path, (1, 100));
}
