use crate::graph::{EdgeWeight, Graph, VertexId, INF_WEIGHT, INVALID_VERTEX};
use crate::matrix::{Array, ArrayBuilder};
use crate::object::{Edge, IndexedMovingObjectSet, MovingObject, Offset};
use crate::pair::pair_in_order;
use crate::partition::{SgId, Subgraph};
use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use std::cmp::min;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Deref;

/// 查询参数: 查询点、查询半径.
type Query = (MovingObject, EdgeWeight);

#[derive(Default)]
pub struct SkeletonGraphBuilder<'a> {
    /// 存储已划分好的子图列表
    subgraphs: Option<&'a Vec<Subgraph>>,
    /// 全局图的引用
    global: Option<&'a Graph>,
}

impl<'a> SkeletonGraphBuilder<'a> {
    pub fn global_graph(mut self, ref_global: &'a Graph) -> Self {
        self.global = Some(ref_global);
        self
    }

    pub fn subgraphs(mut self, ref_subgraphs: &'a Vec<Subgraph>) -> Self {
        self.subgraphs = Some(ref_subgraphs);
        self
    }

    pub fn build(self) -> SkeletonGraph<'a> {
        // 首先获取构建所需的组件，避免多次调用 .expect()
        let subgraphs = self.subgraphs.expect("No subgraphs collection was set.");
        let global = self.global.expect("No global graph set.");

        // 初始化骨架图、边界点反向映射、以及新的内部点映射
        let mut graph = Graph::new();
        let mut inverted_map = HashMap::<_, Vec<SgId>>::new();
        let mut vertex_to_subgraph = HashMap::<VertexId, SgId>::new();

        // 遍历所有子图，构建索引
        for subgraph in subgraphs { // 使用 &subgraphs 进行迭代，这样 subgraphs 本身不会被消耗
            /*
             对每一个子图:
             1. 将其边界顶点和它们之间的边插入骨架图;
             2. 维护边界顶点所属子图的反向映射 (inverted_map);
             3. (新增) 维护内部顶点所属子图的映射 (vertex_to_subgraph).
            */

            // --- 新增逻辑：构建内部顶点到子图的映射 ---
            // 这个映射用于在kNN查询时快速定位查询点所在的初始子图。
            // 由于每个内部顶点只属于一个子图，直接插入即可。
            for v_id in subgraph.internal_vertices.iter() {
                vertex_to_subgraph.insert(v_id, subgraph.id);
            }
            // ------------------------------------------

            // 对每个边界点, 遍历其到其它边界点的距离
            for v_b in subgraph.bound_vertices.iter() {
                graph.insert(v_b);

                for (other_v_b, weight) in subgraph.distance.distance_from(v_b) {
                    // 只处理同在当前子图边界上的顶点对
                    if !subgraph.bound_vertices.contains(&other_v_b) {
                        continue;
                    }

                    graph.insert(other_v_b);

                    /* 某个点对可能存在于多个子图中, 且它们在不同子图上的最短距离不一样。
                    如果某条边已经插入过, 需要尝试其能否变得更小。*/
                    if let Some(old_weight) = graph.get_weight(v_b, other_v_b) {
                        if weight < old_weight {
                            graph.replace(v_b, other_v_b, weight);
                        }
                    } else {
                        graph.connect(v_b, other_v_b, weight);
                    }
                }

                // 更新 inverted_map
                inverted_map.entry(v_b).or_default().push(subgraph.id);
            }
        }

        // 构建并返回最终的 SkeletonGraph 实例
        SkeletonGraph {
            graph,
            subgraphs, // 将原始的 subgraphs 集合移入
            global,
            inverted_map,
            vertex_to_subgraph, // 包含新创建的映射
        }
    }
}

pub struct SkeletonGraph<'a> {
    /// 骨架图实际存储的信息
    pub graph: Graph,
    /// 边界点到子图关系的映射, 该集合会在判断子图类型时用到.
    pub inverted_map: HashMap<VertexId, Vec<SgId>>,
    /// 顶点到子图的映射, 该集合用于快速查询顶点所在的子图.
    pub vertex_to_subgraph: HashMap<VertexId, usize>,

    /// 所有子图信息
    pub subgraphs: &'a Vec<Subgraph>,
    /// 全局图的信息
    pub global: &'a Graph,
}

impl Deref for SkeletonGraph<'_> {
    type Target = Graph;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl<'a> SkeletonGraph<'a> {
    /// The algorithm GRAPH-COVERAGE-COMPUTATION, described on "6.1.1 Identify the candidate subgraphs."
    ///
    /// 算法的实现假设查询点 v_query 已经插入到图 graph 中, 并且替换了原来的边 v_l, v_r.
    /// Returns: V_l, V_r, V_p
    fn gcc(
        graph: &Graph,
        (v_q, d_q): Query,
    ) -> (
        Array,             // v_q 到其他顶点的距离向量
        HashSet<VertexId>, // 最短路径经过 v_l 的点的点集
        HashSet<VertexId>, // 最短路径经过 v_r 的点的点集
        HashSet<VertexId>, // V_p, 等于 v_l union v_r
    ) {
        let v_query = INVALID_VERTEX;

        let (v_l, v_r) = v_q.edge;
        let weight_lr = graph.get_weight(v_l, v_r).unwrap();
        let (d_l, d_r) = (v_q.offset, weight_lr - v_q.offset);

        // Shortest distance from v_q.
        let mut sd_q = ArrayBuilder::new()
            .from(v_query)
            .capacity(graph.size())
            .build();
        sd_q[v_l] = d_l;
        sd_q[v_r] = d_r;

        // First, we set the shortest distances from 𝑣𝑞 to nonadjacent vertices infinity and use a priority queue 𝑄𝑝
        // to keep the unprocessed vertices whose current distances to 𝑣𝑞 are not infinite in ascending order of their
        // current distances to 𝑣𝑞.
        let mut heap: BinaryHeap<Reverse<(EdgeWeight, VertexId, VertexId)>> = BinaryHeap::new();
        // v_l 和 v_r 的上一跳（介绍人）都是自身（用于标记起点方向）
        heap.push(Reverse((d_l, v_l, v_l)));
        heap.push(Reverse((d_r, v_r, v_r)));

        // 为了得到 vertex_by_l 以及 vertex_by_r （即，区分 inevitable vertex）
        // 需要在查询最短距离的同时记录这条最短路径是从 v_l 还是 v_r 开始的.
        let mut ancestor = HashMap::new();
        ancestor.insert(v_l, v_l);
        ancestor.insert(v_r, v_r);

        let set = HashSet::new;
        let mut v_p = set();
        // 类似 Dijkstra 的迭代过程, 将半径内（含边界）的顶点都加入到 V_p 集合中.
        while let Some(Reverse((cost_q_f, v_f, iv))) = heap.pop() {
            if cost_q_f > sd_q[v_f] {
                continue;
            }

            if cost_q_f <= d_q {
                v_p.insert(v_f);
            } else {
                // 此处, 由于优先队列是按 cost_q_f 升序排序的, 一旦当前顶点到查询点 v_q 的代价大于半径
                // 队列中所有后续顶点均不满足条件. 故此处可以使用 break.
                break;
            }

            for (v_a, weight_f_a) in graph.get_out_adjacent_edges(&v_f) {
                let old_distance = sd_q.get_or_inf(v_a);
                let new_distance = cost_q_f + weight_f_a;
                // 这里, 考虑到移动点和骨架图上的顶点重合的情况，因此不能将 SD(v_q, v_f) == r 的顶点 v_f 排除在 v_p 之外.
                if new_distance < old_distance && new_distance <= d_q {
                    sd_q[v_a] = new_distance;
                    ancestor.insert(v_a, iv);
                    heap.push(Reverse((new_distance, v_a, iv)));
                }
            }
        }

        let mut vertices_by_l = set();
        let mut vertices_by_r = set();
        for &v in v_p.iter() {
            // Inevitable Vertex of v.
            let iv = ancestor[&v];
            if iv == v_l {
                vertices_by_l.insert(v);
            } else {
                vertices_by_r.insert(v);
            }
        }

        (sd_q, vertices_by_l, vertices_by_r, v_p)
    }

    /// 在骨架图中扫描范围内的顶点.
    ///
    /// 这个函数主要实现了论文中提到的 GCC 算法, 步骤为：
    /// 1. 向骨架图添加查询点及其所在的边
    /// 2. 类似 Dijkstra 的思想, 迭代更新最短距离，并生成 V_q 集合
    ///
    /// 函数将返回一个 InitQuery 对象, 用于生成查询结果, 或进行移动对象查询.
    pub fn scan(
        &self,
        (query_point, range): Query,
        moving_object_set: &'a IndexedMovingObjectSet,
    ) -> InitQuery {

        // 使用“种子 + 堆”的方式，避免克隆和临时连边。
        let (v_l, v_r) = pair_in_order(query_point.edge);

        // v_q 到 v_l 的距离为 offset, 到 v_r 的距离为 w_{l, r} - offset.
        let weight_lr = self.global.get_weight(v_l, v_r).unwrap_or_else(|| {
            let msg = format!("edge ({v_l}, {v_r}) could not found.");
            panic!("{}", msg);
        });
        assert!(query_point.offset < weight_lr);
        let (d_l, d_r) = (query_point.offset, weight_lr - query_point.offset);

        // 找到 e_{l,r} 所在的子图（优先从 vertex_to_subgraph 命中）
        let initial_subgraph = if let Some(&sg_id) = self.vertex_to_subgraph.get(&v_l) {
            &self.subgraphs[sg_id]
        } else {
            self.subgraphs
                .iter()
                .find(|sg| sg.graph.has_edge(v_l, v_r))
                .expect("fatal: No subgraph contains e_{l, r}")
        };

        // 距离数组
        let v_query = INVALID_VERTEX;
        let mut sd_q = ArrayBuilder::new()
            .from(v_query)
            .capacity(self.graph.size())
            .build();
        sd_q[v_l] = d_l;
        sd_q[v_r] = d_r;

        // 最小堆（重复入堆）: (cost, vertex, origin: 1=left, 2=right)
        let mut heap: BinaryHeap<Reverse<(EdgeWeight, VertexId, u8)>> = BinaryHeap::new();
        heap.push(Reverse((d_l, v_l, 1)));
        heap.push(Reverse((d_r, v_r, 2)));

        // 把“临时连边”的效果改成“边界点种子”
        for b in initial_subgraph.bound_vertices.iter() {
            if b == v_l || b == v_r { continue; }
            let dlb = d_l + initial_subgraph.distance[(v_l, b)];
            let drb = d_r + initial_subgraph.distance[(v_r, b)];
            let (cost, ori) = if dlb <= drb { (dlb, 1u8) } else { (drb, 2u8) };
            if cost < sd_q.get_or_inf(b) && cost <= range {
                sd_q[b] = cost;
                heap.push(Reverse((cost, b, ori)));
            }
        }

        // 方向记录 + V_p
        let mut origin = vec![0u8; self.global.size() + 1];
        origin[v_l as usize] = 1;
        origin[v_r as usize] = 2;
        let mut v_p = HashSet::new();

        // Dijkstra 带半径剪枝
        while let Some(Reverse((cost, u, ori))) = heap.pop() {
            if cost > sd_q[u] { continue; }
            if cost <= range {
                v_p.insert(u);
            } else {
                break;
            }
            for (v, w) in self.graph.get_out_adjacent_edges(&u) {
                let nd = cost + w;
                if nd < sd_q.get_or_inf(v) && nd <= range {
                    sd_q[v] = nd;
                    origin[v as usize] = ori;
                    heap.push(Reverse((nd, v, ori)));
                }
            }
        }

        let mut vertices_by_l = HashSet::new();
        let mut vertices_by_r = HashSet::new();
        for &u in v_p.iter() {
            if origin[u as usize] == 1 { vertices_by_l.insert(u); } else if origin[u as usize] == 2 { vertices_by_r.insert(u); }
        }

        InitQuery {
            skeleton: &self.graph,
            inverted_map: &self.inverted_map,
            v_q: query_point,
            radius: range,
            subgraphs: self.subgraphs,
            global: self.global,
            moving_objects: moving_object_set,
            initial_subgraph,
            distance_from_v_q: sd_q,
            v_p,
            vertices_by_l,
            vertices_by_r,
        }
    }
}

pub struct InitQuery<'a> {
    /// 骨架图存储的相关信息
    skeleton: &'a Graph,
    /// 边界点到子图关系的映射, 该集合会在判断子图类型时用到.
    inverted_map: &'a HashMap<VertexId, Vec<SgId>>,

    /// 查询起点
    v_q: MovingObject,
    /// 查询半径
    radius: EdgeWeight,

    /// 全部子图的列表
    subgraphs: &'a Vec<Subgraph>,
    /// 全局图
    global: &'a Graph,
    /// 移动对象集合
    moving_objects: &'a IndexedMovingObjectSet,

    /// 在查询范围中的骨架图顶点集合
    v_p: HashSet<VertexId>,
    /// v_p 中, 满足条件的、经过 v_l 距离最短的顶点集合
    vertices_by_l: HashSet<VertexId>,
    /// v_p 中, 满足条件的、经过 v_r 距离最短的顶点集合
    vertices_by_r: HashSet<VertexId>,
    /// 初始子图
    initial_subgraph: &'a Subgraph,
    /// v_q 到骨架图上各点的距离
    distance_from_v_q: Array,
}

impl<'a> InitQuery<'a> {
    pub fn results(&mut self) -> Vec<SgId> {
        use std::cmp::Reverse;
        use std::collections::{BinaryHeap, HashMap, HashSet};

        let mut result = Vec::new();
        let init_sg = self.initial_subgraph;

        // 1) 初始子图是否 fully-covered（边界点是否都在 V_p 中）
        let covered_count_init = init_sg
            .bound_vertices
            .iter()
            .filter(|&b| self.v_p.contains(&b))
            .count();
        let is_initial_sg_fully_covered = covered_count_init == init_sg.bound_vertices.len();

        // 2) 每个子图被命中的“边界点”数量（只统计边界点）
        let mut hit = vec![0usize; self.subgraphs.len()];
        for &v in self.v_p.iter() {
            if let Some(sg_list) = self.inverted_map.get(&v) {
                for &sg_id in sg_list {
                    hit[sg_id] += 1;
                }
            }
        }

        // 3) 分类：fully-covered / partially-covered
        let (mut fc_subgraphs, mut pc_subgraphs) = (Vec::<SgId>::new(), Vec::<SgId>::new());
        for (sg_id, cnt) in hit.into_iter().enumerate() {
            if cnt == 0 { continue; }
            let sg = &self.subgraphs[sg_id];
            if cnt == sg.bound_vertices.len() || (sg_id == init_sg.id && is_initial_sg_fully_covered) {
                fc_subgraphs.push(sg_id);
            } else {
                pc_subgraphs.push(sg_id);
            }
        }

        // 4) fully-covered：整批加入对象
        for sg_id in fc_subgraphs.into_iter() {
            result.extend(self.moving_objects.objects_in(sg_id));
        }

        // 5) partially-covered：对子图跑一次 Dijkstra（用 HashMap 避免“域”崩溃）
        for sg_id in pc_subgraphs.into_iter() {
            let sg = &self.subgraphs[sg_id];
            debug_assert_eq!(sg.id, sg_id);

            type DistMap = HashMap<VertexId, EdgeWeight>;
            let mut dist: DistMap = DistMap::with_capacity(sg.bound_vertices.len() * 4 + 16);
            let mut heap: BinaryHeap<Reverse<(EdgeWeight, VertexId)>> = BinaryHeap::new();

            // 用 V_p ∩ boundary(sg) 作为堆种子
            for b in sg.bound_vertices.iter() {
                if !self.v_p.contains(&b) { continue; }
                let d0 = self.distance_from_v_q[b];
                if d0 < *dist.get(&b).unwrap_or(&INF_WEIGHT) {
                    dist.insert(b, d0);
                    if d0 <= self.radius {
                        heap.push(Reverse((d0, b))); // (距离, 顶点)
                    }
                }
            }

            let mut fully_visited_edges: HashSet<Edge> = HashSet::new();

            while let Some(Reverse((du, u))) = heap.pop() {
                if du > self.radius { break; }
                if let Some(&best) = dist.get(&u) {
                    if du != best { continue; } // 过期状态
                } else { continue; }

                for (v, w) in sg.get_out_adjacent_edges(&u) {
                    let e = pair_in_order((u, v));
                    if fully_visited_edges.contains(&e) { continue; }

                    // 松弛
                    let nd = du + w;
                    if nd < *dist.get(&v).unwrap_or(&INF_WEIGHT) {
                        dist.insert(v, nd);
                        if nd <= self.radius {
                            heap.push(Reverse((nd, v)));
                        }
                    }

                    let du_cur = *dist.get(&u).unwrap_or(&INF_WEIGHT);
                    let dv_cur = *dist.get(&v).unwrap_or(&INF_WEIGHT);

                    // 两端点均在半径内：整边收集
                    if du_cur <= self.radius && dv_cur <= self.radius {
                        result.extend(self.moving_objects.objects_on(e));
                        fully_visited_edges.insert(e);
                        continue;
                    }

                    // 中点在半径内：整边收集
                    if dv_cur != INF_WEIGHT && (dv_cur + du_cur + w) / 2 <= self.radius {
                        result.extend(self.moving_objects.objects_on(e));
                        fully_visited_edges.insert(e);
                        continue;
                    }

                    // 否则逐对象检查
                    for obj in self
                        .moving_objects
                        .objects_on(e)
                        .map(|obj_id| &self.moving_objects[obj_id])
                    {
                        let left = du_cur.saturating_add(obj.offset);
                        let right = dv_cur.saturating_add(w - obj.offset);
                        if core::cmp::min(left, right) <= self.radius {
                            result.push(obj.unique_id);
                        }
                    }
                }
            }
        }

        result
    }

    pub fn into_continuous_query(self) -> ContinuousQuery<'a> {
        ContinuousQuery {
            skeleton: self.skeleton,
            inverted_map: self.inverted_map,
            v_q: self.v_q,
            radius: self.radius,
            subgraphs: self.subgraphs,
            global: self.global,
            moving_objects: self.moving_objects,
            initial_subgraph: self.initial_subgraph,
            distance_from_v_q: self.distance_from_v_q,
            v_p: self.v_p,
            vertices_by_l: self.vertices_by_l,
            vertices_by_r: self.vertices_by_r,
        }
    }
}

pub struct ContinuousQuery<'a> {
    /// 骨架图存储的相关信息
    skeleton: &'a Graph,
    /// 边界点到子图关系的映射, 该集合会在判断子图类型时用到.
    inverted_map: &'a HashMap<VertexId, Vec<SgId>>,

    /// 查询起点
    v_q: MovingObject,
    /// 查询半径
    radius: EdgeWeight,
    /// 全部子图的列表
    subgraphs: &'a Vec<Subgraph>,
    /// 全局图
    global: &'a Graph,
    /// 移动对象集合
    moving_objects: &'a IndexedMovingObjectSet,

    /// 初始子图
    initial_subgraph: &'a Subgraph,
    /// v_q 到骨架图上各点的距离
    distance_from_v_q: Array,
    /// 在查询范围中的骨架图顶点集合
    v_p: HashSet<VertexId>,
    /// v_p 中, 满足条件的、经过 v_l 距离最短的顶点集合
    vertices_by_l: HashSet<VertexId>,
    /// v_p 中, 满足条件的、经过 v_r 距离最短的顶点集合
    vertices_by_r: HashSet<VertexId>,
}

impl<'a> ContinuousQuery<'a> {
    pub fn move_query_point(&mut self, new_offset: Offset) {
        assert_ne!(new_offset, self.v_q.offset, "The query point need to move!");

        let (v_l, v_r) = self.v_q.edge;
        let weight_lr = self.global.get_weight(v_l, v_r).unwrap();
        assert!(new_offset <= weight_lr);

        let delta_w = new_offset.abs_diff(self.v_q.offset) as EdgeWeight;
        let (vertex_set_h, vertex_set_t) = if new_offset > self.v_q.offset {
            (&mut self.vertices_by_r, &mut self.vertices_by_l)
        } else {
            (&mut self.vertices_by_l, &mut self.vertices_by_r)
        };

        let set = || HashSet::with_capacity(self.skeleton.size());
        let array = || {
            ArrayBuilder::new()
                .from(0)
                .capacity(self.skeleton.size())
                .build()
        };
        // HSD = hypothetical shortest distance
        let (mut hsd_from_v_q_prime, mut sd_from_v_q_prime) = (array(), array());

        println!("old V_h: {}", vertex_set_h.len());
        println!("old V_t: {:?}", vertex_set_t.len());
        let mut undetermined = set();
        undetermined.extend(
            self.skeleton
                .vertices()
                .filter(|v| !vertex_set_h.contains(v)),
        );

        // 把 V_h 中所有的点视为 HS-vertices(v_q')
        // 对 v_i \in V_h, SD(v_i, v_q') = SD(v_i, v_q) - \delta w
        for &v_i in vertex_set_h.iter() {
            sd_from_v_q_prime[v_i] = self.distance_from_v_q[v_i] - delta_w;
        }
        // println!("sd_from_v_q_prime: {sd_from_v_q_prime:?}");

        // 对 v_j \in V_t, HSD(v_j, v_i') = SD(v_j, v_q) + \delta w
        // 并标记 v_j 未确定，即，不确定它是 HS-v or TS-v(v_q')
        for &v_j in vertex_set_t.iter() {
            hsd_from_v_q_prime[v_j] = self.distance_from_v_q[v_j] + delta_w;
        }
        // println!("hsd_from_v_q_prime: {hsd_from_v_q_prime:?}");

        // Put the vertices from V_h with undetermined adjacent vertices into Q_m.
        let mut q_m: BinaryHeap<Reverse<(EdgeWeight, VertexId)>> = BinaryHeap::new();
        vertex_set_h
            .iter()
            .flat_map(|v| {
                // v 有一个相邻顶点的距离未确定
                self.skeleton
                    .get_out_adjacent_edges(v)
                    .filter_map(|(v_a, w)| {
                        if undetermined.contains(&v_a) {
                            Some((sd_from_v_q_prime[*v] + w, v_a))
                        } else {
                            None
                        }
                    })
            })
            .for_each(|(d, v)| {
                q_m.push(Reverse((d, v)));
            });

        while let Some(Reverse((cost, v_f))) = q_m.pop() {
            if sd_from_v_q_prime[v_f] < cost {
                break;
            }

            if cost <= self.radius {
                // 首先，处理 v_f, 并决定它在 TS-v(v_q') 中还是 HS-v 中.
                if undetermined.contains(&v_f) {
                    if !vertex_set_t.contains(&v_f) {
                        // 如果 v_f 未被更新，并且它不在 V_t 中，那么它便是 HS-v(v'_q) 中的点, 移动到 V_h
                        vertex_set_h.insert(v_f);
                        for (v_a, w_f_a) in self
                            .skeleton
                            .get_out_adjacent_edges(&v_f)
                            .filter(|&(v_f, _)| undetermined.contains(&v_f))
                        {
                            let old_distance = sd_from_v_q_prime.get_or_inf(v_a);
                            let new_distance = cost + w_f_a;
                            if old_distance > new_distance {
                                sd_from_v_q_prime[v_a] = new_distance;
                            }

                            q_m.push(Reverse((new_distance, v_a)));
                        }
                    }
                    /* v_f is in the set V_t */
                    else if cost >= hsd_from_v_q_prime[v_f] {
                        // Do noting
                        // v_f is a TS-Vertex(v'_q), and retained in V_t.
                    } else {
                        vertex_set_t.take(&v_f);
                        vertex_set_h.insert(v_f);
                    }
                } // v_f is undetermined.
            }
            /* SD(v_f, v'_q) > d_q */
            else if vertex_set_t.contains(&v_f) && hsd_from_v_q_prime[v_f] <= self.radius {
                sd_from_v_q_prime[v_f] = hsd_from_v_q_prime[v_f];
            }
        }

        for &v_j in vertex_set_t.intersection(&undetermined) {
            if hsd_from_v_q_prime[v_j] <= self.radius {
                sd_from_v_q_prime[v_j] = hsd_from_v_q_prime[v_j];
            }
        }

        self.v_q.offset = new_offset;
        /* Now V_t and V_h is updated. */

        println!("new V_h: {}", vertex_set_h.len());
        println!("new V_t: {:?}", vertex_set_t.len());

        // println!("previous v_p size = {}", self.v_p.len());
        self.v_p = vertex_set_h
            .union(vertex_set_t)
            .copied()
            .collect::<HashSet<_>>();
        println!("new v_p size = {}", self.v_p.len());
    }

    /// 查询移动对象, 返回移动对象 Id 集合.
    ///
    /// 1. 将子图分为 fully-covered, partially-covered 和不用管的, 分别处理.
    /// 2. 处理初始子图.
    pub fn results(&mut self) -> Vec<SgId> {
        let mut result = Vec::new();

        let (v_l, v_r) = pair_in_order(self.v_q.edge);
        let init_sg = self.initial_subgraph;

        /* Processing the initial-subgraph.
        这里调整一下处理顺序: 先看看 initial-subgraph 是否满足 v_l, v_r 都是边界点的情况.
        如果满足的话, 它是个 fully-covered subgraph, 后面就不用管它了. */
        let covered_count__init = init_sg
            .bound_vertices
            .iter()
            .filter(|v| self.v_p.contains(v))
            .count();
        let is_initial_sg_fully_covered = covered_count__init == init_sg.bound_vertices.len();

        /* 根据顶点被遍历的情况, 得到完全覆盖和部分覆盖的子图列表. */

        // 子图 -> V_p 中包含该子图顶点的数量

        // 子图 -> V_p 中包含该子图边界顶点的数量（向量化计数，避免哈希）
        let mut hit = vec![0usize; self.subgraphs.len()];
        for &v_b in &self.v_p {
            if let Some(subs) = self.inverted_map.get(&v_b) {
                for &sg in subs.iter() {
                    hit[sg] += 1;
                }
            } else {
                // inverted_map 可能不含初始子图内的非边界点，视为命中初始子图
                hit[init_sg.id] += 1;
            }
        }

        // 统计每个子图被覆盖到的“边界点命中数”
        let mut hit: Vec<usize> = vec![0; self.subgraphs.len()];

        // self.v_p：半径内访问到的顶点集合（HashSet<VertexId> 或 Vec<VertexId> 都可以）
        // self.inverted_map：边界点 -> 所属子图ID列表（HashMap<VertexId, Vec<usize>>）
        for &v_b in self.v_p.iter() {
            if let Some(sg_list) = self.inverted_map.get(&v_b) {
                for &sg_id in sg_list {
                    hit[sg_id] += 1;
                }
            }
        }

        // 根据命中数把子图分成 fully-covered / partially-covered
        let (mut fc_subgraphs, mut pc_subgraphs) = (Vec::new(), Vec::new());
        for (sg_id, &cnt) in hit.iter().enumerate() {
            if cnt == 0 { continue; } // 完全没被涉及到的子图跳过
            let sg = &self.subgraphs[sg_id];
            if cnt == sg.bound_vertices.len() {
                // 边界点全部命中 => fully covered
                fc_subgraphs.push(sg_id);
            } else {
                // 命中了但不全 => partially covered
                pc_subgraphs.push(sg_id);
            }
        }

        // 添加 fc-subgraph 中的结果（实际测试中该部分可以不计时）
        for fc_subgraph in fc_subgraphs.into_iter() {
            result.extend(self.moving_objects.objects_in(fc_subgraph));
        }

        // 添加 pc-subgraph 中的结果
        // v_q 到 object 的路径是这样的:
        // v_q - v_b of subgraph - v_x/v_y - object, 因此可以将几段路径相加, 并和查询半径作比较.
        for pc_subgraph in pc_subgraphs.into_iter() {
            let sg = &self.subgraphs[pc_subgraph];
            assert_eq!(sg.id, pc_subgraph);

            let covered_borders = sg
                .bound_vertices
                .iter()
                .filter(|v_b| self.v_p.contains(v_b))
                .collect::<Vec<_>>();

            for object in self
                .moving_objects
                .objects_in(pc_subgraph)
                .map(|obj_id| &self.moving_objects[obj_id])
            {
                let (left_vertex, right_vertex) = object.edge;
                let (left_weight, right_weight) = (
                    object.offset,
                    sg.get_weight(left_vertex, right_vertex).unwrap() - object.offset,
                );

                for &v_b in covered_borders.iter() {
                    let distance = min(
                        self.distance_from_v_q[v_b] + sg.distance[(v_b, left_vertex)] + left_weight,
                        self.distance_from_v_q[v_b]
                            + sg.distance[(v_b, right_vertex)]
                            + right_weight,
                    );

                    if distance <= self.radius {
                        result.push(object.unique_id);
                        break;
                    }
                }
                // Continue to next object.
            }
            // Continue to next subgraph.
        }

        result
    }
}

#[cfg(test)]
mod test {
    use crate::index::SkeletonGraph;
    use crate::loader::load_graph;
    use crate::object::MovingObject;

    #[test]
    fn test_skeleton_gcc_on_paper() {
        let path = "../dataset/skeleton.gr";

        let v_q = MovingObject::query_point((1, 2), 1);
        let graph = load_graph(path).unwrap();

        let (_sd_q, vertices_by_l, vertices_by_r, v_p) = SkeletonGraph::gcc(&graph, (v_q, 8));

        assert_eq!(v_p.len(), 10);
        assert_eq!(vertices_by_r.len(), 4);
        assert_eq!(vertices_by_l.len(), 6);
        for v in [2, 3, 4, 5] {
            assert!(vertices_by_r.contains(&v));
        }
        for v in [1, 6, 7, 8, 9, 10] {
            assert!(vertices_by_l.contains(&v));
        }
    }
}
