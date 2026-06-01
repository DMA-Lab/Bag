use bap::index::{SkeletonGraph, SkeletonGraphBuilder};
use bap::loader::{load_graph, load_or_partition};
use bap::partition::VfipPartition;

// 这个函数是你需要实现的，用来验证你的核心数据结构是否一致
fn validate_ghost_nodes_and_edges(
    skeleton: &SkeletonGraph,
) -> Result<(), String> {
    println!("Starting advanced partition validation (checking ghost nodes and edges)...");

    let global_graph = skeleton.global;
    let subgraphs = skeleton.subgraphs;
    let inverted_map = &skeleton.inverted_map;

    // 1. 遍历全局图中的每一条边
    for ((u, v), _) in global_graph.edges() { // 假设 global_graph 有一个返回所有边的方法
        // 2. 检查从 u 的角度出发是否一致
        if let Some(u_subgraph_ids) = inverted_map.get(&u) {
            // 3. 对于每一个包含 u 的子图...
            for &sg_id in u_subgraph_ids {
                let subgraph = &subgraphs[sg_id]; // 获取该子图的引用

                // 4. 核心验证：该子图也必须包含 v 和边 (u, v)

                // 验证点 A: 子图是否包含了边的另一个端点 v
                if !subgraph.graph.contains(v) { // 假设 Subgraph 内部的 graph 有 contains_node 方法
                    return Err(format!(
                        "VALIDATION FAILED: Inconsistency found for edge ({}, {}).\n\
                         Node {} is in Subgraph {}, but this subgraph is MISSING the ghost node {}!",
                        u, v, u, sg_id, v
                    ));
                }

                // 验证点 B: 子图是否包含了这条边本身
                if !subgraph.graph.has_edge(u, v) { // 假设有 contains_edge 方法
                    return Err(format!(
                        "VALIDATION FAILED: Inconsistency found for edge ({}, {}).\n\
                         Subgraph {} contains both nodes {} and {}, but is MISSING the edge between them!",
                        u, v, sg_id, u, v
                    ));
                }
            }
        }

        // (可选但推荐) 对称地从 v 的角度再检查一遍，确保万无一失
        if let Some(v_subgraph_ids) = inverted_map.get(&v) {
            for &sg_id in v_subgraph_ids {
                let subgraph = &subgraphs[sg_id];
                if !subgraph.graph.contains(u) {
                    return Err(format!(
                        "VALIDATION FAILED (Symmetric Check): Inconsistency for edge ({}, {}).\n\
                         Node {} is in Subgraph {}, but this subgraph is MISSING the ghost node {}!",
                        u, v, v, sg_id, u
                    ));
                }
                if !subgraph.graph.has_edge(u, v) {
                    return Err(format!(
                        "VALIDATION FAILED (Symmetric Check): Inconsistency for edge ({}, {}).\n\
                         Subgraph {} contains both nodes {} and {}, but is MISSING the edge between them!",
                        u, v, sg_id, u, v
                    ));
                }
            }
        }
    }

    println!("Advanced partition validation successful. All ghost nodes and border edges are consistent.");
    Ok(())
}

// 在你的主流程中调用它：
// let skeleton_graph = build_skeleton_graph(...);
// validate_ghost_nodes_and_edges(&skeleton_graph)
//     .expect("FATAL: The constructed subgraphs are inconsistent with the partition maps!");

// ... 只有验证通过后，才继续执行后续的查询 ...
fn main() {
    let path = "../dataset/USA-road-d.NY.gr";
    let full_graph = load_graph(path).unwrap();

    let subgraphs = load_or_partition(
        path,
        &full_graph,
        1,
        30,
        |graph, theta, start| VfipPartition::new(graph, theta).run(start),
    );

    let skeleton_graph = SkeletonGraphBuilder::default()
        .subgraphs(&subgraphs)
        .global_graph(&full_graph)
        .build();
    validate_ghost_nodes_and_edges(&skeleton_graph).expect("Partitioning logic is broken!"); // 如果验证失败，程序应该立即崩溃
}