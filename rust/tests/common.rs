use std::collections::{hash_map, HashMap};

use bap::graph::{Graph, VertexId};
use bap::loader::{load_graph, moving_objects};
use bap::object::{MovingObject, MovingObjectSet, ObjId};
use bap::partition::{Subgraph, VfipPartition};

/// 加载并使用 VFIP 划分图.
pub fn load_partition_graph(graph_name: &str, seed: VertexId, theta: usize) -> (Graph, Vec<Subgraph>) {
    let graph_path = format!("{graph_name}.gr");
    let graph = load_graph(graph_path).unwrap();

    let vfip = VfipPartition::new(&graph, theta);
    let subgraphs = vfip.run(seed);

    (graph, subgraphs)
}


/// 加载图和移动点集合.
pub fn load_graph_suite(graph_name: &str, seed: VertexId, count: usize, theta: usize) -> (Graph, Vec<Subgraph>, MovingObjectSet) {
    /* Load graph */
    let (graph, subgraphs) = load_partition_graph(graph_name, seed, theta);

    /* Load moving object set */
    let mos_path = format!("{graph_name}.mos.gr");
    let objects = moving_objects::load_or_generate(mos_path, &graph, count);
    assert_eq!(objects.len(), count);

    (graph, subgraphs, objects)
}

pub fn remove_moving_objects_on_same_edge(objects: &mut MovingObjectSet, v_q: MovingObject) {
    let size = objects.len();
    let object_to_delete = objects
        .iter()
        .filter(|obj| obj.edge == v_q.edge)
        .map(|obj| obj.unique_id)
        .collect::<Vec<_>>();
    for object in object_to_delete {
        objects.remove(object);
    }
    let deleted = size - objects.len();
    if deleted != 0 {
        println!(
            "{deleted} objects removed, because they are on the same edge as the query point."
        );
    }
}


/// 将某个移动对象作为顶点插入图中.
pub fn add_object_to_graph(graph: &mut Graph, MovingObject { edge, offset, .. }: MovingObject, new_vertex_id: VertexId) {
    let (v_l, v_r) = edge;
    let (d_l, d_r) = (offset, graph.get_weight(v_l, v_r).unwrap() - offset);
    graph.insert(new_vertex_id);
    graph.disconnect(v_l, v_r);
    graph.connect(v_l, new_vertex_id, d_l);
    graph.connect(new_vertex_id, v_r, d_r);
}

/// 将移动对象集合插入图中.
///
/// 返回值包括下一个顶点的编号, 以及一个映射关系, 将顶点编号映射到移动对象编号.
pub fn port_moving_objects_to_graph(graph: &mut Graph, moving_objects: &MovingObjectSet) -> (VertexId, HashMap<VertexId, ObjId>) {
    /* 将移动点插入图:
      基本想法是: 删除原来的边，将其拆成两条边, 中间插入移动对象（作为一个顶点）.
      然而，由于一条边上可能有多个移动对象，需要先 collect 一下, 再做处理.
    */
    let mut edge_to_objects = HashMap::new();
    for object in moving_objects.iter() {
        match edge_to_objects.entry(object.edge) {
            hash_map::Entry::Vacant(e) => {
                e.insert(vec![object]);
            }
            hash_map::Entry::Occupied(mut e) => {
                (*e.get_mut()).push(object);
            }
        }
    }

    // 插入移动点.
    // 由于最终结果需要 Object 的编号, 而 Dijkstra 中我们将移动对象作为顶点, 图上顶点编号.
    // 所以需要保存一个映射关系.
    let mut vertex_object_mapping = HashMap::new();
    let mut next_vertex = (graph.size() as VertexId) + 1;
    for (edge, mut objects) in edge_to_objects.into_iter() {
        let weight = if let Some(weight) = graph.get_weight(edge.0, edge.1) {
            weight
        } else {
            continue;
        };
        objects.sort_by(|a, b| a.offset.cmp(&b.offset));

        // 打断原始的边.
        graph.disconnect(edge.0, edge.1);
        let mut last_joint = edge.0;
        let mut last_offset = 0;

        for MovingObject {
            offset, unique_id, ..
        } in objects
        {
            graph.insert(next_vertex);
            graph.connect(last_joint, next_vertex, *offset - last_offset);
            vertex_object_mapping.insert(next_vertex, *unique_id);

            last_offset = *offset;
            last_joint = next_vertex;
            next_vertex += 1;
        }
        graph.connect(last_joint, edge.1, weight - last_offset);
    }

    (next_vertex, vertex_object_mapping)
}