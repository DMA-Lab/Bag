use mimalloc::MiMalloc;

use bap::graph::VertexId;
use bap::index::SkeletonGraph;
use bap::loader::{load_graph, load_or_generate_moving_objects, load_or_partition};
use bap::object::IndexedMovingObjectSet;
use bap::partition::VfipPartition;
use bap::{get_slots, reset_slots};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn find_good_v_q(seed: VertexId, skeleton_graph: &SkeletonGraph) -> VertexId {
    if skeleton_graph
        .subgraphs
        .iter()
        .find(|sg| sg.internal_vertices.contains(&seed))
        .is_some() {
        return seed;
    }

    skeleton_graph.inverted_map[&seed]
        .iter()
        .copied()
        .filter_map(|sg_id| {
            let sg = &skeleton_graph.subgraphs[sg_id];
            sg.internal_vertices.iter().next()
        })
        .next()
        .unwrap_or(seed)
}

fn do_test(name: &str, graph_path: &str) {
    let full_graph = load_graph(graph_path).unwrap();

    let subgraphs = load_or_partition(
        graph_path,
        &full_graph,
        1,
        30,
        |graph, theta, start| VfipPartition::new(graph, theta).run(start),
    );
    println!("subgraphs: {:?}", subgraphs.len());
    let skeleton_graph = bap::index::SkeletonGraphBuilder::default()
        .subgraphs(&subgraphs)
        .global_graph(&full_graph)
        .build();

    let objects = load_or_generate_moving_objects(graph_path, &full_graph, 10000);
    let objects = IndexedMovingObjectSet::new(&subgraphs, objects);


    // let mut rng = thread_rng();
    // let quires: Vec<_> = full_graph.vertices().choose_multiple(&mut rng, 100);
    let quires = [100000]
        .iter()
        .copied()
        .map(|v| find_good_v_q(v, &skeleton_graph))
        .collect::<Vec<_>>();

    // test parameters
    let k_range = [50];
    // let k_range = vec![1];
    let times = 3u32;

    for &k in k_range.iter() {
        let mut min_time_vec = [0u64; 5];
        for _ in 0..times {
            reset_slots!();
            for &v_q in quires.iter() {
                let query = skeleton_graph.knn_query(&objects, v_q, k);
                let results = query.run();
                // assert_eq!(results.len(), k);

                let slot_iter = get_slots!(5);
                for (slot, &time) in slot_iter.enumerate() {
                    if time < min_time_vec[slot] || min_time_vec[slot] == 0 {
                        min_time_vec[slot] = time;
                    }
                }
            }
        }

        print!("{name}\t{k}\t");
        for &time in min_time_vec.iter() {
            print!("{}\t", time);
        }

        print!("{}", min_time_vec.iter().sum::<u64>());

        println!();
    }
}

fn test_knn_on_NY() {
    let graph_path = "../dataset/USA-road-d.NY.gr";
    do_test("NY", graph_path);
}

fn test_knn_on_CAL() {
    let graph_path = "../dataset/USA-road-d.CAL.gr";
    do_test("CAL", graph_path);
}

fn test_knn_on_FLA() {
    let graph_path = "../dataset/USA-road-d.FLA.gr";
    do_test("FLA", graph_path);
}

fn main() {
    test_knn_on_NY();
    // test_knn_on_CAL();
    // test_knn_on_FLA();
    // test_knn_on_CUSA();

    println!("task finished.");
}