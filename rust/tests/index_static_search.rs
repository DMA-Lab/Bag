use std::collections::HashSet;

use bap::calc_time;
use bap::distance::{DijkstraQuery, QueryDistance};
use bap::graph::{EdgeWeight, Graph};
use bap::index::SkeletonGraphBuilder;
use bap::loader::{load_graph, moving_objects};
use bap::object::{IndexedMovingObjectSet, MovingObject, MovingObjectSet, ObjId};
use bap::partition::VfipPartition;
use utils::calculate_cost_time;

use crate::common::{add_object_to_graph, port_moving_objects_to_graph};

mod common;

/// 查询参数
#[derive(Clone)]
struct SearchInstance<'a> {
    /// 起始点
    v_q: MovingObject,
    /// 查询半径
    radius: EdgeWeight,

    /// 移动点（这会儿是静态的）集合
    object_set: &'a MovingObjectSet,
}

fn search_by_bap(
    graph: &Graph,
    theta: usize,
    SearchInstance {
        v_q,
        radius,
        object_set,
    }: SearchInstance,
) -> Vec<ObjId> {
    // 子图划分
    let vfip = VfipPartition::new(graph, theta);
    let subgraphs = vfip.run(1);

    // 克隆 object set
    let object_set = {
        let mut s = MovingObjectSet::default();
        object_set
            .iter()
            .for_each(|o| s.emplace(o.unique_id, o.edge, o.offset));
        s
    };

    // 关联移动对象及子图
    let tagged_object_set = IndexedMovingObjectSet::new(&subgraphs, object_set);
    // 生成骨架图
    let skeleton = calc_time("Build skeleton graph", || {
        SkeletonGraphBuilder::default()
            .subgraphs(&subgraphs)
            .global_graph(graph)
            .build()
    });

    // 查询.
    let mut binding = skeleton.scan((v_q, radius), &tagged_object_set);
    calc_time("detect type of sg and collect results", || {
        binding.results()
    })
}

fn search_by_dijkstra(
    graph: &mut Graph,
    SearchInstance {
        v_q,
        radius,
        object_set,
    }: SearchInstance,
) -> Vec<ObjId> {
    /* 将查询点加入图 */
    let v_query = 0;
    add_object_to_graph(graph, v_q, v_query);

    let (_, vertex_object_mapping) = port_moving_objects_to_graph(graph, object_set);

    let distance = calc_time("Build Dijkstra distance array", || {
        DijkstraQuery::new(graph, v_query).build()
    });

    vertex_object_mapping
        .iter()
        .filter(|(v, id)| **v != v_query && distance.query(**v) <= radius)
        .map(|(_, id)| *id)
        .collect()
}

fn test_correctness_of_index(
    graph_name: &str,
    count: usize,
    theta: usize,
    v_q: MovingObject,
    radius: EdgeWeight,
) {
    // load graph
    let graph_path = format!("{graph_name}.gr");
    let mut graph = load_graph(graph_path).unwrap();

    /* 加载移动对象集合, 并删除和查询点在同一条边上的移动对象. (TODO: 将来应该支持这样的查询.) */
    // MOS = Moving Object Set
    let mos_path = format!("{graph_name}.mos.gr");
    let mut moving_objects = moving_objects::load_or_generate(mos_path, &graph, count);
    common::remove_moving_objects_on_same_edge(&mut moving_objects, v_q);

    let parameter = SearchInstance {
        v_q,
        radius,
        object_set: &moving_objects,
    };

    let result_bap = calc_time("Search by BAP", || {
        search_by_bap(&graph, theta, parameter.clone())
    })
        .iter()
        .fold(HashSet::new(), |mut set, e| {
            set.insert(*e);
            set
        });

    let result_dijkstra = calc_time("Search by Dijkstra", || {
        search_by_dijkstra(&mut graph, parameter.clone())
    })
        .iter()
        .fold(HashSet::new(), |mut set, e| {
            set.insert(*e);
            set
        });

    let diff = result_dijkstra.difference(&result_bap);
    for &obj_id in diff {
        // 对于每一个在 Dijkstra 搜索结果中, 但是不在 BAP 结果中的点.
        let object = &moving_objects[obj_id];
        println!("BAP has missed {object:?}");
    }

    let diff = result_bap.difference(&result_dijkstra);
    for &obj_id in diff {
        // 对于每一个在 BAP 搜索结果中, 但是不在 Dijkstra 结果中的点.
        let object = &moving_objects[obj_id];
        println!("Dijkstra has missed {object:?}");
    }
    assert_eq!(result_bap.len(), result_dijkstra.len());
}

#[test]
fn test_graph_on_paper() {
    let path = "../dataset/on-paper.gr";

    let v_q = MovingObject::new(0, (4, 6), 1);
    test_correctness_of_index(path, 8, 8, v_q, 24);

    let v_q = MovingObject::new(0, (7, 8), 1);
    test_correctness_of_index(path, 8, 8, v_q, 20);
}

#[test]
fn test_graph_231028() {
    let path = "../dataset/test-231028.gr";

    let v_q = MovingObject::new(0, (4, 5), 2);
    test_correctness_of_index(path, 8, 8, v_q, 5);
}

#[test]
fn test_graph_new_york() {
    let path = "../dataset/USA-road-d.NY.gr";

    let v_q = MovingObject::new(0, (1, 2), 6);
    test_correctness_of_index(path, 800000, 30, v_q, 70000);
}

#[test]
fn test_graph_new_york_400() {
    let path = "../dataset/USA-road-d.NY-stripped-400.gr";

    let v_q = MovingObject::new(0, (1, 2), 1);
    test_correctness_of_index(path, 800, 30, v_q, 700);
}
