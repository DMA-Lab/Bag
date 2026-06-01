use bap::calc_time;
use bap::graph::{EdgeWeight, Graph};
use bap::index::SkeletonGraphBuilder;
use bap::loader::{load_graph, moving_objects};
use bap::object::{IndexedMovingObjectSet, MovingObject, MovingObjectSet, ObjId};
use bap::partition::VfipPartition;

use crate::common::remove_moving_objects_on_same_edge;

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
    let result1 = binding.results();
    println!("The results, before v_q moves: {:?}", result1.len());

    let mut continuous_query = binding.into_continuous_query();
    continuous_query.move_query_point(v_q.offset + 599);

    let result2 = continuous_query.results();
    println!("The results, after v_q moves: {:?}", result2.len());
    result2
}

fn test(
    graph_name: &str,
    count: usize,
    theta: usize,
    v_q: MovingObject,
    radius: EdgeWeight,
) {
    // load graph
    let graph_path = format!("{graph_name}.gr");
    let graph = load_graph(graph_path).unwrap();

    // load moving objects
    let mos_path = format!("{graph_name}.mos.gr");
    let mut objects = moving_objects::load_or_generate(mos_path, &graph, count);
    remove_moving_objects_on_same_edge(&mut objects, v_q);

    let parameter = SearchInstance {
        v_q,
        radius,
        object_set: &objects,
    };

    let _ = calc_time("Search by BAP", || {
        search_by_bap(&graph, theta, parameter)
    });
}

#[test]
fn test_graph_on_paper() {
    let path = "../dataset/on-paper.gr";

    let v_q = MovingObject::query_point((7, 8), 1);
    test(path, 8, 8, v_q, 8)
}

#[test]
fn test_graph_new_york() {
    let path = "../dataset/USA-road-d.NY.gr";

    let v_q = MovingObject::query_point((1, 2), 6);
    test(path, 800000, 30, v_q, 70000);
}
