use bap::calc_time;
use bap::distance::{DijkstraQuery, QueryDistance};
use bap::graph::Graph;
use bap::loader::load_graph;

fn load_new_york_graph() -> Graph {
    load_graph("../dataset/USA-road-t.NY.gr").expect("unable to load USA-road-d.NY.gr for test.")
}

#[test]
fn test_dijkstra_implementation() {
    let graph = load_new_york_graph();

    let dijkstra = calc_time("dijkstra", || DijkstraQuery::new(&graph, 100086).build());

    println!("{}", dijkstra.query(10086));
    println!("{}", dijkstra.query(85));
    println!("{}", dijkstra.query(3254));
}
