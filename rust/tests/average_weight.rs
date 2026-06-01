use bap::graph::EdgeWeight;
use bap::loader::load_graph;

#[test]
fn test_average_weight() {
    let graph = load_graph("../dataset/USA-road-d.NY.gr").unwrap();

    let total_weight: EdgeWeight = graph.edges().map(|(_, w)| w).sum();
    println!("total weight is {total_weight}");
    println!("edge count is {}", graph.edge_count());
    println!(
        "average weight is {}",
        total_weight as usize / graph.edge_count()
    );
}
