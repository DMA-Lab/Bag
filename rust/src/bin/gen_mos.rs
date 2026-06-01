use bap::loader::{export_moving_objects_text, load_graph};
use bap::object::{MovingObject, MovingObjectSet};

fn main() {
    let graph = load_graph("../dataset/USA-road-d.NY.gr").unwrap();
    let moving_objects = MovingObjectSet::random(&graph, 1000000).unwrap();

    assert_eq!(moving_objects.len(), 1000000);
    for MovingObject { edge, offset, .. } in moving_objects.iter() {
        let &(v1, v2) = edge;
        assert!(graph.has_edge(v1, v2));

        let w = graph.get_weight(v1, v2).unwrap();
        assert!((1..w).contains(offset));
    }

    match export_moving_objects_text("../dataset/USA-road-d.NY.1000000.mos", &moving_objects) {
        Ok(_) => println!("Exported moving objects to text file successfully."),
        Err(e) => eprintln!("Failed to export moving objects to text file: {}", e),
    }
}