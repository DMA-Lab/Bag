// src/index/knn.rs

use crate::graph::MayHalfPathWeight;
use crate::graph::{EdgeWeight, VertexId, INF_WEIGHT};
use crate::index::SkeletonGraph;
use crate::object::{IndexedMovingObjectSet, ObjId};
use ahash::{HashMapExt, HashSetExt};
use std::cmp::{min, Reverse};
use std::collections::BinaryHeap;

/// 辅助函数：将 MayHalfPathWeight 转换为 f64 以便计算。
/// 在实际工程中，为了避免浮点数误差，可能会选择将所有权重乘以2来处理。
fn may_half_to_f64(weight: MayHalfPathWeight) -> f64 {
    weight.0 as f64 + if weight.1 { 0.5 } else { 0.0 }
}

/// 存储在候选集 S_c 中的子图信息，对应论文中的 OS_i。
struct SubgraphCandidateInfo {
    /// 子图 id
    sg_id: SgId,
    /// 距离下界 (LD)
    lower_bound: f64,
    /// 距离上界 (UD)
    upper_bound: f64,
}

/// kNN 查询结构体。
pub struct KnnQuery<'a> {
    /// 查询点 v_q 的顶点编号
    v_q: VertexId,
    /// 期望的结果数
    k: usize,
    /// 骨架图
    skeleton: &'a SkeletonGraph<'a>,
    /// 索引的移动对象集合
    objects: &'a IndexedMovingObjectSet,
}

impl<'a> SkeletonGraph<'a> {
    /// 创建一个 kNN 查询。
    pub fn knn_query(
        &'a self,
        objects: &'a IndexedMovingObjectSet,
        v_q: VertexId,
        k: usize,
    ) -> KnnQuery<'a> {
        KnnQuery {
            v_q,
            k,
            skeleton: self,
            objects,
        }
    }
}

use crate::partition::SgId;
use crate::{calc_elapsed_time, reset_time_point};

impl<'a> KnnQuery<'a> {
    fn nearest_border_dist(sg: &crate::partition::Subgraph, v: VertexId) -> EdgeWeight {
        if sg.bound_vertices.contains(&v) {
            return 0;
        }
        if let Some(&dist) = sg.internal_to_nearest_border_dist.get(&v) {
            return dist;
        }

        sg.bound_vertices
            .iter()
            .map(|border_v| sg.distance.get_or_inf(v, border_v))
            .min()
            .unwrap_or(INF_WEIGHT)
    }

    /// 计算对象在初始子图（根子图）内的精确距离。（无变化）
    fn distance_in_init_subgraph(&self, sg_id: usize, obj_id: ObjId) -> EdgeWeight {
        let sg = &self.skeleton.subgraphs[sg_id];
        let obj = &self.objects[obj_id];
        let (v_x, v_y) = obj.edge;
        let weight = sg.get_weight(v_x, v_y).unwrap();

        let d_x = sg.distance[(v_x, self.v_q)] + obj.offset;
        let d_y = sg.distance[(v_y, self.v_q)] + weight - obj.offset;
        min(d_x, d_y)
    }

    /// 计算一个对象的精确最短距离。（无变化）
    fn calculate_exact_distance(
        &self,
        obj_id: ObjId,
        sg_id: usize,
        init_sg_id: usize,
        root_objects: &[(ObjId, f64)],
        skeleton_distances: &ahash::HashMap<VertexId, f64>,
    ) -> f64 {
        if sg_id == init_sg_id {
            // 对于根子图中的对象，其精确距离已在初始阶段计算好
            root_objects.iter().find(|(id, _)| *id == obj_id).unwrap().1
        } else {
            // 对于其他子图的对象，通过其所有边界顶点计算最短路径
            let sg = &self.skeleton.subgraphs[sg_id];
            let obj = &self.objects[obj_id];
            let (v_x, v_y) = obj.edge;
            let edge_weight = sg.get_weight(v_x, v_y).unwrap();
            let mut min_exact_dist = f64::MAX;

            for border_vertex in sg.bound_vertices.iter() {
                if let Some(&dist_q_to_border) = skeleton_distances.get(&border_vertex) {
                    let dist_border_to_vx = sg.distance[(border_vertex, v_x)];
                    let dist_border_to_vy = sg.distance[(border_vertex, v_y)];

                    let path_dist_via_x = if dist_border_to_vx == INF_WEIGHT { f64::MAX } else { (dist_border_to_vx + obj.offset) as f64 };
                    let path_dist_via_y = if dist_border_to_vy == INF_WEIGHT { f64::MAX } else { (dist_border_to_vy + edge_weight - obj.offset) as f64 };

                    let dist_border_to_obj = path_dist_via_x.min(path_dist_via_y);

                    if dist_border_to_obj < f64::MAX {
                        min_exact_dist = min_exact_dist.min(dist_q_to_border + dist_border_to_obj);
                    }
                }
            }
            min_exact_dist
        }
    }

    /// [重构核心] 运行 BAG-kNN 算法的优化版本
    pub fn run(&self) -> Vec<ObjId> {
        // --- 步骤 0: 初始化与边界情况处理 ---
        if self.objects.len() < self.k {
            return self.objects.iter().map(|o| o.unique_id).collect();
        }
        if self.k == 0 {
            return Vec::new();
        }

        // --- 初始化计时和核心数据结构 ---
        reset_time_point!();

        // exploration_heap: 用于在骨架图上进行Dijkstra式探索，按【距离v_q的距离】排序
        let mut exploration_heap: BinaryHeap<(Reverse<u64>, VertexId)> = BinaryHeap::with_capacity(512);
        // skeleton_distances: 记录v_q到已访问边界点的最短距离
        let mut skeleton_distances: ahash::HashMap<VertexId, f64> = ahash::HashMap::with_capacity(1024);

        // object_candidate_heap: 维护当前【RLB最佳】的候选对象，按【RLB】排序
        // 这是本次重构的关键，它将取代 s_c 和后续的排序/建堆过程。
        // 我们用一个最大堆来维护RLB最小的k个候选，方便剪枝。
        let mut object_candidate_heap: BinaryHeap<(u64, ObjId, SgId)> = BinaryHeap::with_capacity(self.k * 2);

        // visited_subgraphs: 确保每个子图只被处理一次以计算其内部对象的RLB
        let mut visited_subgraphs = ahash::HashSet::<SgId>::with_capacity(512);

        // --- 步骤 1: 处理根子图，填充初始状态 ---
        let init_sg_id = self.skeleton.vertex_to_subgraph[&self.v_q];
        visited_subgraphs.insert(init_sg_id);
        let init_sg = &self.skeleton.subgraphs[init_sg_id];

        // 1.1 计算根子图内对象的精确距离，并直接作为RLB加入候选堆
        let mut root_objects_vec: Vec<(ObjId, f64)> = Vec::with_capacity(init_sg.graph.size());
        for obj_id in self.objects.objects_in(init_sg.id) {
            let dist = self.distance_in_init_subgraph(init_sg.id, obj_id) as f64;
            root_objects_vec.push((obj_id, dist));
            object_candidate_heap.push((dist.to_bits(), obj_id, init_sg_id));
        }

        // 1.2 将初始子图的边界点加入探索堆
        for v_b in init_sg.bound_vertices.iter() {
            let dist_vb = init_sg.distance[(self.v_q, v_b)] as f64;
            exploration_heap.push((Reverse(dist_vb.to_bits()), v_b));
            skeleton_distances.insert(v_b, dist_vb);
        }
        calc_elapsed_time!(slot = 0); // slot 0: 初始化阶段

        // --- 步骤 2: 统一的、双堆驱动的探索与剪枝循环 ---
        calc_elapsed_time!(slot = 1); // slot 1 & 3 合并为探索阶段
        while let Some((Reverse(dist_vb_bits), v_b)) = exploration_heap.pop() {
            let dist_vb = f64::from_bits(dist_vb_bits);

            // **核心剪枝**: 如果当前要探索的边界点 v_b 本身，
            // 就比我们已经找到的第 k 个候选对象的 RLB 还要远，
            // 那么从 v_b 出发不可能找到更好的 kNN 了。
            // 由于 exploration_heap 是最小堆，后续所有点的距离都会更大，可以直接终止探索。
            if object_candidate_heap.len() >= self.k {
                let kth_best_rlb_bits = object_candidate_heap.peek().unwrap().0;
                if dist_vb_bits >= kth_best_rlb_bits {
                    break;
                }
            }

            // Dijkstra 的经典剪枝
            if dist_vb > *skeleton_distances.get(&v_b).unwrap_or(&f64::MAX) {
                continue;
            }

            // 扩展 v_b 所属的子图
            if let Some(sg_ids) = self.skeleton.inverted_map.get(&v_b) {
                for &sg_id in sg_ids {
                    // 确保每个子图只被处理一次，避免重复计算和添加对象
                    if visited_subgraphs.insert(sg_id) {
                        let sg = &self.skeleton.subgraphs[sg_id];
                        // 立即计算该新子图中所有对象的 RLB，并加入对象候选堆
                        for obj_id in self.objects.objects_in(sg_id) {
                            let obj = &self.objects[obj_id];
                            let (v_a, v_b_obj) = obj.edge; // 变量名区分，避免与外层 v_b 混淆

                            let md_a = Self::nearest_border_dist(sg, v_a);
                            let md_b = Self::nearest_border_dist(sg, v_b_obj);
                            if md_a == INF_WEIGHT && md_b == INF_WEIGHT {
                                continue;
                            }

                            let edge_weight = sg.get_weight(v_a, v_b_obj).unwrap();
                            let rlb_suffix = min(md_a + obj.offset, md_b + edge_weight - obj.offset) as f64;
                            let rlb = dist_vb + rlb_suffix; // 注意：这里的RLB是基于当前入口v_b的距离估算的，更准确的应该是用sg的LB
                            // 修正：RLB应该用子图的LB，而不是当前dist_vb
                            // 找到这个子图所有已访问边界点中最小的距离作为LB
                            let sg_lb = sg.bound_vertices.iter()
                                .filter_map(|bv| skeleton_distances.get(&bv))
                                .fold(f64::MAX, |min_dist, &dist| min_dist.min(dist));

                            let rlb = sg_lb + rlb_suffix;

                            object_candidate_heap.push((rlb.to_bits(), obj_id, sg_id));
                            // 维持候选堆的大小，超过k个时移除RLB最大的那个
                            if object_candidate_heap.len() > self.k {
                                object_candidate_heap.pop();
                            }
                        }

                        // 将该子图的其他边界点加入探索堆（子图扩展策略）
                        for other_vb in sg.bound_vertices.iter() {
                            if other_vb != v_b {
                                let dist_b_to_other = sg.distance.get_or_inf(v_b, other_vb);
                                if dist_b_to_other == INF_WEIGHT {
                                    continue;
                                }
                                let new_dist_to_other = dist_vb + dist_b_to_other as f64;
                                if new_dist_to_other < *skeleton_distances.get(&other_vb).unwrap_or(&f64::MAX) {
                                    skeleton_distances.insert(other_vb, new_dist_to_other);
                                    exploration_heap.push((Reverse(new_dist_to_other.to_bits()), other_vb));
                                }
                            }
                        }
                    }
                }
            }
        }
        calc_elapsed_time!(slot = 1); // slot 1: 统一探索阶段结束

        // --- 步骤 3: 从最终的候选堆中计算精确距离并确定kNN ---
        calc_elapsed_time!(slot = 4); // slot 4: 最终确定阶段

        // 此时 object_candidate_heap 中是RLB最有希望的k个候选者
        // 我们需要对它们进行精确计算和排序
        let final_candidates = object_candidate_heap.into_sorted_vec(); // O(k log k)

        let mut final_knn_heap: BinaryHeap<(u64, ObjId)> = BinaryHeap::with_capacity(self.k);
        let mut d_k_bits = u64::MAX;

        for (rlb_bits, obj_id, sg_id) in final_candidates.into_iter() {
            // 这里的 rlb 是 f64::from_bits(rlb_bits)
            if rlb_bits >= d_k_bits {
                // 剪枝，但因为已经是排序好的，所以这个 break 可能用处不大，
                // 主要是为了理论上的完备性
                break;
            }

            let exact_dist = self.calculate_exact_distance(obj_id, sg_id, init_sg_id, &root_objects_vec, &skeleton_distances);
            let exact_dist_bits = exact_dist.to_bits();

            if exact_dist_bits < d_k_bits {
                if final_knn_heap.len() == self.k {
                    final_knn_heap.pop();
                }
                final_knn_heap.push((exact_dist_bits, obj_id));
                if final_knn_heap.len() == self.k {
                    d_k_bits = final_knn_heap.peek().unwrap().0;
                }
            }
        }

        // --- 步骤 4: 合并最终结果 ---
        let final_result: Vec<ObjId> = final_knn_heap.into_iter().map(|(_, id)| id).collect();

        calc_elapsed_time!(slot = 4);

        final_result
    }
}
