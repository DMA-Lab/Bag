use ahash::{HashSet, HashSetExt};

use bap::calc_time;
use bap::distance::{DijkstraQuery, QueryDistance};
use bap::graph::{EdgeWeight, Graph, VertexId};
use bap::index::{SkeletonGraph, SkeletonGraphBuilder};
use bap::loader::load_graph;
use bap::loader::moving_objects::load_or_generate;
use bap::object::{IndexedMovingObjectSet, MovingObject, ObjId};
use bap::partition::VfipPartition;

use crate::common::load_graph_suite;

mod common;


fn search_by_knn(objects: &IndexedMovingObjectSet, graph: &SkeletonGraph, v_q: VertexId, k: usize) -> (Vec<ObjId>, EdgeWeight) {
    let query = graph.knn_query(&objects, v_q, k);
    query.run()
}


fn search_by_dijkstra(objects: &IndexedMovingObjectSet, graph: &mut Graph, v_q: VertexId, k: usize) -> Vec<ObjId> {
    /* kNN 操作中，假设 v_q 是一个顶点而不是移动点，已经在图中. */
    let v_query = v_q;

    // 这里不需要将移动对象插入图中，只需要计算源点到所有顶点的距离，再算所有移动对象的距离即可。
    calc_time("Build Dijkstra distance array", || {
        let dijkstra = DijkstraQuery::new(graph, v_query).build();

        let mut obj_dist = pheap::PairingHeap::new();
        for obj in objects.iter() {
            let (v_left, v_right) = obj.edge;
            let weight = graph.get_weight(v_left, v_right).unwrap();
            let d_left = dijkstra.query(v_left) + obj.offset;
            let d_right = dijkstra.query(v_right) + weight - obj.offset;
            obj_dist.insert(obj.unique_id, d_left.min(d_right));
        }

        let mut result = Vec::with_capacity(k);
        for _ in 0..k {
            if let Some((obj_id, _)) = obj_dist.delete_min() {
                result.push(obj_id);
            } else {
                unimplemented!("Not enough objects in the graph")
            }
        }
        result
    })
}

fn test(graph_name: &str, v_q: VertexId, count: usize) {
    let (mut g, subgraphs, objects) = load_graph_suite(graph_name, v_q, count, 100);

    let objs = IndexedMovingObjectSet::new(&subgraphs, objects);
    let graph = SkeletonGraphBuilder::default()
        .subgraphs(&subgraphs)
        .global_graph(&g)
        .build();

    let result_by_knn = {
        let (vec, _) = calc_time("kNN", || search_by_knn(&objs, &graph, v_q, count));
        HashSet::from_iter(vec)
    };

    let result_by_dijkstra = {
        let vec = calc_time("Dijkstra", || search_by_dijkstra(&objs, &mut g, v_q, count));
        HashSet::from_iter(vec)
    };

    assert_eq!(result_by_knn, result_by_dijkstra);
}

#[test]
fn test_knn_240217() {
    let graph_name = "../dataset/test-240217";
    let v_q = 1 as VertexId;

    test(graph_name, v_q, 9);
}


#[test]
fn test_on_new_york_road() {
    let graph_name = "../dataset/USA-road-d.NY";
    let v_q = 1 as VertexId;

    test(graph_name, v_q, 10000);
}

#[test]
fn test_knn_on_new_york() {
    let graph_path = "../dataset/USA-road-d.NY.gr";
    let full_graph = load_graph(graph_path).unwrap();

    let subgraphs = VfipPartition::new(&full_graph, 30).run(1);
    let skeleton_graph = SkeletonGraphBuilder::default()
        .subgraphs(&subgraphs)
        .global_graph(&full_graph)
        .build();

    let mos_path = "../dataset/USA-road-d.NY.mos.gr";
    let objects = load_or_generate(mos_path, &skeleton_graph, 10000);

    for MovingObject { edge, offset, .. } in objects.iter() {
        // println!("edge: {:?}, offset: {:?}", edge, offset);

        let &(v1, v2) = edge;
        assert!(full_graph.has_edge(v1, v2));

        let w = full_graph.get_weight(v1, v2).unwrap();
        assert!((1..w).contains(offset));
    }

    let objects = IndexedMovingObjectSet::new(&subgraphs, objects);
    for MovingObject { edge, offset, .. } in objects.iter() {
        assert!(objects.find_edge(edge).is_some());
    }

    println!("objects: {:?}", objects.len());

    let k_range = 1..1000;
    println!("k_range: {:?}", k_range);
    let mut previous_result = HashSet::new();
    let mut last_d_max = 0 as EdgeWeight;
    for k in k_range {
        println!("k = {k}");
        // TODO: let knn query returns the distance of the k-nearest neighbor.
        let query = skeleton_graph.knn_query(&objects, 1000, k);
        let (result, d_max) = query.run();

        let result_set: HashSet<_> = result.iter().copied().collect();
        assert_eq!(result.len(), std::cmp::min(k, objects.len()));
        // assert!(result_set.is_superset(&previous_result));
        if d_max < last_d_max && last_d_max == EdgeWeight::MAX {
            eprintln!("error: d_max = {d_max}, last_d_max = {last_d_max}");
            last_d_max = 0;
        }
        last_d_max = d_max;
        previous_result = result_set;
    }
}