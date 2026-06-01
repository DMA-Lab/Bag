use std::cmp::{max, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use bincode::{Decode, Encode};

use crate::graph::{EdgeWeight, Graph, MayHalfPathWeight, VertexId, INF_WEIGHT};
use crate::matrix::{Matrix, TMatrix};
use crate::pair::pair_in_order;
use crate::transaction::{TQueue, TSet, Transaction};

/// 子图编号
pub type SgId = usize;

/// 顶层 run 的“后处理合并”重入保护（防止递归触发）
static POST_REPARTITION_GUARD: AtomicBool = AtomicBool::new(false);

#[derive(Default, Encode, Decode)]
pub struct Subgraph {
    /// 子图编号.
    pub id: SgId,

    /// 当前子图的基本数据（边和点）
    pub graph: Graph,
    /// 当前子图中顶点的最短距离矩阵
    pub distance: TMatrix,

    /// 边界点 点集
    pub bound_vertices: TSet<VertexId>,
    /// 内部点 点集
    pub internal_vertices: TSet<VertexId>,
    /// 边界顶点的 Reachable Bound
    pub rb_map: HashMap<VertexId, MayHalfPathWeight>,
    /// 存储每个内部点到其最近边界点的距离 (MD(v))
    pub internal_to_nearest_border_dist: HashMap<VertexId, EdgeWeight>,
}

impl Subgraph {
    /// 判断顶点 v 是否是子图中的内部点
    fn check_internal_vertex(&self, global: &Graph, v: VertexId) -> bool {
        for (adjacent_vertex, _) in global.get_out_adjacent_edges(&v) {
            if !self.graph.contains(adjacent_vertex) || !self.graph.has_edge(adjacent_vertex, v) {
                return false;
            }
        }
        true
    }

    /// 转成内部点
    fn turn_to_internal_vertex(&mut self, v: VertexId) {
        self.internal_vertices.insert(v);
        self.bound_vertices.remove(&v);
    }

    /// 获取当前子图的所有顶点
    pub fn all_vertices(&self) -> HashSet<VertexId> {
        self.bound_vertices
            .iter()
            .chain(self.internal_vertices.iter())
            .collect()
    }
}

impl Deref for Subgraph {
    type Target = Graph;
    fn deref(&self) -> &Self::Target { &self.graph }
}
impl DerefMut for Subgraph {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.graph }
}

impl Debug for Subgraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SG{}, {:?}", self.id, self.graph)?;
        writeln!(
            f,
            " {:?} are bound vertices, and {:?} are internal vertices.",
            self.bound_vertices, self.internal_vertices
        )
    }
}

impl Transaction for Subgraph {
    fn begin(&mut self) {
        self.graph.begin();
        self.bound_vertices.begin();
        self.internal_vertices.begin();
    }
    fn rollback(&mut self) {
        self.graph.rollback();
        self.bound_vertices.rollback();
        self.internal_vertices.rollback();
    }
    fn commit(&mut self) {
        self.graph.commit();
        self.bound_vertices.commit();
        self.internal_vertices.commit();
    }
}

// ============================== AIP ==============================

pub struct AipPartition<'a> {
    global: &'a Graph,
    visited_vertices: TSet<VertexId>,
    added_edges: TSet<(VertexId, VertexId)>,
    seed_vertices: HashSet<VertexId>,
}

impl<'a> AipPartition<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Self {
            global: graph,
            visited_vertices: Default::default(),
            added_edges: Default::default(),
            seed_vertices: Default::default(),
        }
    }

    fn unallocated_adjacent_edge(
        &'a self,
        source: &'a VertexId,
    ) -> impl Iterator<Item = (VertexId, EdgeWeight)> + 'a {
        self.global
            .get_out_adjacent_edges(source)
            .filter(move |(v, _w)| !self.added_edges.contains(&pair_in_order((*v, *source))))
    }

    fn relax(
        distance: &mut Matrix,
        src: VertexId,
        dst: VertexId,
        relay: VertexId,
    ) -> Option<(EdgeWeight, EdgeWeight)> {
        if src == dst { return None; }
        let (new_distance, overflow) =
            distance[(src, relay)].overflowing_add(distance[(relay, dst)]);
        let old_distance = &mut distance[(src, dst)];
        if !overflow && *old_distance > new_distance {
            let result = *old_distance;
            *old_distance = new_distance;
            Some((result, new_distance))
        } else {
            None
        }
    }

    fn expand_from_seed(&mut self, seed: VertexId) -> (Subgraph, HashSet<VertexId>) {
        let mut subgraph = Subgraph::default();
        let mut distance = TMatrix::with_capacity(20);

        let mut queue = TQueue::default();
        queue.push(seed);

        'add_vertex: while let Some(v_i) = queue.pop() {
            subgraph.begin();
            self.added_edges.begin();
            self.visited_vertices.begin();

            subgraph.insert(v_i);
            subgraph.bound_vertices.insert(v_i);

            queue.begin();
            for (adj_v, weight) in self.unallocated_adjacent_edge(&v_i).collect::<Vec<_>>() {
                if self.visited_vertices.contains(&adj_v) && subgraph.contains(adj_v) {
                    self.added_edges.insert(pair_in_order((adj_v, v_i)));
                    subgraph.connect(v_i, adj_v, weight);
                    distance[(adj_v, v_i)] = weight;
                    distance[(v_i, adj_v)] = weight; // 双向
                    if subgraph.check_internal_vertex(self.global, adj_v) {
                        subgraph.turn_to_internal_vertex(adj_v);
                    }
                } else {
                    queue.push(adj_v);
                }
            }

            if v_i != seed {
                let bound_vertices = subgraph.bound_vertices.iter().collect::<Vec<_>>();
                for v_b in bound_vertices.into_iter() {
                    if v_b == v_i { continue; }
                    let mut pi = v_b;
                    for (v_j, weight) in subgraph.get_out_adjacent_edges(&v_i) {
                        if v_b == v_j { continue; }
                        let (new_distance, overflow) = distance[(v_b, v_j)].overflowing_add(weight);
                        if !overflow && new_distance < distance[(v_b, v_i)] {
                            distance[(v_b, v_i)] = new_distance;
                            distance[(v_i, v_b)] = new_distance; // 双向
                            pi = v_j;
                        }
                    }
                    let rb: MayHalfPathWeight = subgraph
                        .bound_vertices
                        .iter()
                        .map(|v_j| (distance[(v_b, v_j)], false))
                        .max()
                        .unwrap_or((0, false));

                    let mut invalid_flag = false;
                    let mut q_v = VecDeque::new();
                    let mut relaxed = HashSet::new();
                    for (v_j, _) in subgraph.get_out_adjacent_edges(&v_i) {
                        if v_j != pi { q_v.push_back(v_j); }
                    }
                    while let Some(v_f) = q_v.pop_front() {
                        relaxed.insert(v_f);
                        let distal_distance = if let Some((old_distance, new_distance)) =
                            Self::relax(&mut distance, v_b, v_f, v_i)
                        {
                            for (adj_v, _) in subgraph.get_out_adjacent_edges(&v_f) {
                                if !relaxed.contains(&adj_v) { q_v.push_back(adj_v); }
                            }
                            ((old_distance + new_distance) >> 1, (old_distance + new_distance) & 1 != 0)
                        } else {
                            let last_distance = distance[(v_b, v_f)];
                            let by_v_i_distance = distance[(v_b, v_i)] + distance[(v_i, v_f)];
                            ((last_distance + by_v_i_distance) >> 1,
                             (last_distance + by_v_i_distance) & 1 != 0)
                        };
                        if distal_distance > rb { invalid_flag = true; }
                    }

                    if !invalid_flag {
                        let ib: MayHalfPathWeight = distance
                            .distance_from(v_b)
                            .filter(|(v_j, _)| subgraph.internal_vertices.contains(v_j))
                            .map(|(_, w)| (w, false))
                            .max()
                            .unwrap_or((0, false));
                        if ib > rb { invalid_flag = true; }
                    }

                    if invalid_flag {
                        queue.rollback();
                        subgraph.rollback();
                        self.added_edges.rollback();
                        self.visited_vertices.rollback();
                        continue 'add_vertex;
                    } else {
                        subgraph.rb_map.insert(v_b, rb);
                    }
                }
            }

            if subgraph.bound_vertices.len() > 2 && subgraph.check_internal_vertex(self.global, v_i) {
                subgraph.turn_to_internal_vertex(v_i);
            }

            self.visited_vertices.insert(v_i);
            queue.commit();
            subgraph.commit();
            self.added_edges.commit();
            self.visited_vertices.commit();
        }

        let possible_seed_vertex = subgraph
            .bound_vertices
            .iter()
            .filter(|bound_vertex| self.unallocated_adjacent_edge(bound_vertex).next().is_some())
            .fold(HashSet::new(), |mut set, v| { set.insert(v); set });

        (subgraph, possible_seed_vertex)
    }

    fn remove_unavailable_seeds(&self, seed_set: &mut HashSet<VertexId>) {
        let seeds: Vec<VertexId> = seed_set.iter().copied().collect();
        for seed in seeds {
            if self.unallocated_adjacent_edge(&seed).next().is_none() {
                seed_set.remove(&seed);
            }
        }
    }

    pub fn run(mut self, seed: VertexId) -> Vec<Subgraph> {
        let mut seed_vertices = HashSet::new();
        seed_vertices.insert(seed);

        let mut result = Vec::new();
        while let Some(&first_vertex) = seed_vertices.iter().next() {
            let (last_subgraph, new_seeds) = self.expand_from_seed(first_vertex);
            result.push(last_subgraph);
            seed_vertices.extend(new_seeds.iter());
            self.remove_unavailable_seeds(&mut seed_vertices);
        }
        result
    }
}

// ============================== VFIP ==============================

pub struct VfipPartition<'a> {
    global: &'a Graph,
    theta: usize,
    added_edges: TSet<(VertexId, VertexId)>,
    seed_vertices: HashSet<VertexId>,
}

impl<'a> VfipPartition<'a> {
    pub fn new(graph: &'a Graph, theta: usize) -> Self {
        Self { global: graph, theta, added_edges: Default::default(), seed_vertices: Default::default() }
    }

    fn unallocated_adjacent_edge(
        &'a self,
        source: &'a VertexId,
    ) -> impl Iterator<Item = (VertexId, EdgeWeight)> + 'a {
        self.global
            .get_out_adjacent_edges(source)
            .filter(move |(v, _w)| {
                let (v1, v2) = pair_in_order((*v, *source));
                !self.added_edges.contains(&(v1, v2))
            })
    }

    fn relax(
        distance: &mut Matrix,
        src: VertexId,
        dst: VertexId,
        relay: VertexId,
    ) -> Option<(EdgeWeight, EdgeWeight)> {
        if src == dst { return None; }
        let (new_distance, overflow) = distance.get_or_inf(src, relay).overflowing_add(distance[(relay, dst)]);
        let old_distance = &mut distance[(src, dst)];
        if !overflow && new_distance < *old_distance {
            let backup = *old_distance;
            *old_distance = new_distance;
            Some((backup, new_distance))
        } else {
            None
        }
    }

    fn expand_next(&mut self) -> Option<Subgraph> {
        let seed = *self.seed_vertices.iter().next()?;
        self.seed_vertices.remove(&seed);
        debug_assert!(self.unallocated_adjacent_edge(&seed).next().is_some());

        let mut subgraph = Subgraph::default();

        // 阶段1：插点（Dijkstra风格）
        let mut alternative_edges = pheap::PairingHeap::new();
        subgraph.insert(seed);
        subgraph.bound_vertices.insert(seed);
        for (adj_v, weight) in self.unallocated_adjacent_edge(&seed).take(self.theta - 1) {
            alternative_edges.insert((seed, adj_v), weight);
        }

        while let Some(((v_j, v_i), cost)) = alternative_edges.delete_min() {
            if subgraph.size() == self.theta { break; }
            if subgraph.bound_vertices.contains(&v_i)
                || cost > subgraph.distance.get_or_inf(seed, v_i) {
                continue;
            }

            subgraph.begin();
            self.added_edges.begin();

            let weight = self.global.get_weight(v_j, v_i).unwrap_or(INF_WEIGHT);
            subgraph.insert(v_i);
            subgraph.distance[(seed, v_i)] = cost;
            subgraph.distance[(v_i, seed)] = cost;  // 双向
            subgraph.distance[(v_i, v_j)] = weight;
            subgraph.distance[(v_j, v_i)] = weight;  // 双向
            subgraph.graph.connect(v_i, v_j, subgraph.distance[(v_i, v_j)]);
            self.added_edges.insert(pair_in_order((v_i, v_j)));
            subgraph.bound_vertices.insert(v_i);
            for v in [v_i, v_j] {
                if subgraph.check_internal_vertex(self.global, v) {
                    self.seed_vertices.take(&v);
                    if v != seed && self.global.get_out_adjacent_edges(&v).count() != 1 {
                        subgraph.turn_to_internal_vertex(v);
                    }
                } else if self.unallocated_adjacent_edge(&v).next().is_none() {
                    self.seed_vertices.take(&v);
                }
            }

            // 从 v_i 单源更新
            let mut q = VecDeque::new();
            let mut processed = HashSet::new();
            q.push_back((v_i, 0));
            while let Some((current, cost_from_v_i)) = q.pop_front() {
                processed.insert(current);
                for (v, w) in subgraph.graph.get_out_adjacent_edges(&current) {
                    if v_i != v && subgraph.contains(v) && !processed.contains(&v) {
                        let d = cost_from_v_i + w;
                        subgraph.distance[(v, v_i)] = d;
                        subgraph.distance[(v_i, v)] = d; // 双向
                        q.push_back((v, d));
                    }
                }
            }

            let (rb, ib) = (
                subgraph.bound_vertices.iter().map(|v_j| (subgraph.distance[(v_i, v_j)], false)).max().unwrap_or((0, false)),
                subgraph.internal_vertices.iter().map(|v_j| (subgraph.distance[(v_i, v_j)], false)).max().unwrap_or((0, false)),
            );
            if rb < ib {
                subgraph.rollback();
                self.added_edges.rollback();
                continue;
            }

            for (adj_v, weight) in self.unallocated_adjacent_edge(&v_i) {
                let new_distance = subgraph.distance[(seed, v_i)] + weight;
                if new_distance < subgraph.distance.get_or_inf(seed, adj_v) {
                    alternative_edges.insert((v_i, adj_v), new_distance);
                }
            }

            subgraph.commit();
            subgraph.rb_map.insert(v_i, rb);
            self.added_edges.commit();
        }

        // 阶段2：潜在边（保留你当前写法，不展开大改）
        let mut potential_edges = HashMap::new();
        for v_b in subgraph.bound_vertices.iter() {
            for (dst, weight) in self.unallocated_adjacent_edge(&v_b) {
                if subgraph.contains(dst) && !subgraph.has_edge(v_b, dst) {
                    potential_edges.insert(pair_in_order((v_b, dst)), weight);
                }
            }
        }

        let mut potential_edges = potential_edges.iter().map(|((u, v), w)| (*u, *v, *w)).collect::<Vec<_>>();
        let queue_length = potential_edges.len();
        potential_edges.sort_by(|(_, _, a), (_, _, b)| b.cmp(a));

        let mut failed_insertion_count = 0;
        'insert_edge: while let Some((v_x, v_y, weight)) = potential_edges.pop() {
            if (failed_insertion_count > 0 && potential_edges.is_empty())
                || failed_insertion_count >= queue_length { break; }

            subgraph.begin();
            self.added_edges.begin();
            subgraph.distance.begin();

            subgraph.connect(v_x, v_y, weight);
            self.added_edges.insert(pair_in_order((v_x, v_y)));
            subgraph.distance[(v_x, v_y)] = weight;
            subgraph.distance[(v_y, v_x)] = weight; // 双向

            for v_i in [v_x, v_y] {
                if subgraph.check_internal_vertex(self.global, v_i) {
                    subgraph.turn_to_internal_vertex(v_i);
                    self.seed_vertices.take(&v_i);
                } else if self.unallocated_adjacent_edge(&v_i).next().is_none() {
                    self.seed_vertices.take(&v_i);
                }
            }

            let bound_vertices: Vec<_> = subgraph.bound_vertices.iter().collect();
            for v_b in bound_vertices {
                let mut q_e = VecDeque::new();
                q_e.push_back((v_x, v_y, weight));

                let mut ib: (EdgeWeight, bool) = (0, false);
                while let Some((v_i, v_j, w)) = q_e.pop_front() {
                    if !subgraph.contains(v_i) || !subgraph.contains(v_j) {
                        continue;
                    }
                    let relax_v_i = Self::relax(&mut subgraph.distance, v_b, v_i, v_j);
                    let relax_v_j = Self::relax(&mut subgraph.distance, v_b, v_j, v_i);

                    match (relax_v_i, relax_v_j) {
                        (Some(_), Some(_)) => unreachable!(),
                        (Some(_), None) => {
                            for (adj_v, w2) in subgraph.get_out_adjacent_edges(&v_i) {
                                q_e.push_back((v_i, adj_v, w2));
                            }
                        }
                        (None, Some(_)) => {
                            for (adj_v, w2) in subgraph.get_out_adjacent_edges(&v_j) {
                                q_e.push_back((v_j, adj_v, w2));
                            }
                        }
                        (None, None) => {
                            let x = subgraph.distance[(v_b, v_j)] + subgraph.distance[(v_b, v_i)] + w;
                            let distal = (x >> 1, x & 1 != 0);
                            ib = max(ib, distal);
                        }
                    }
                }

                let ib = max(
                    ib,
                    subgraph.internal_vertices.iter().map(|v| (subgraph.distance[(v_b, v)], false)).max().unwrap_or((0, false)),
                );
                let rb = subgraph.bound_vertices.iter().map(|v| (subgraph.distance[(v_b, v)], false)).max().unwrap_or((0, false));
                if rb < ib {
                    failed_insertion_count += 1;
                    potential_edges.insert(0, (v_x, v_y, weight));
                    subgraph.rollback();
                    self.added_edges.rollback();
                    subgraph.distance.rollback();
                    continue 'insert_edge;
                } else {
                    subgraph.rb_map.insert(v_b, rb);
                }
            }

            subgraph.commit();
            self.added_edges.commit();
            subgraph.distance.commit();
            failed_insertion_count = 0;
        }

        for v_b in subgraph.bound_vertices.iter() {
            if self.unallocated_adjacent_edge(&v_b).next().is_some() {
                self.seed_vertices.insert(v_b);
            }
        }

        for internal_v in subgraph.internal_vertices.iter() {
            let min_dist_to_border = subgraph
                .bound_vertices
                .iter()
                .map(|border_v| subgraph.distance[(internal_v, border_v)])
                .min()
                .unwrap_or(INF_WEIGHT);
            subgraph.internal_to_nearest_border_dist.insert(internal_v, min_dist_to_border);
        }

        Some(subgraph)
    }

    pub fn run(mut self, seed: VertexId) -> Vec<Subgraph> {
        let mut result = Vec::new();
        self.seed_vertices.insert(seed);
        while let Some(new_subgraph) = self.expand_next() {
            result.push(new_subgraph);
        }

        // 统一编号
        for (i, subgraph) in result.iter_mut().enumerate() {
            subgraph.id = i;
        }

        // 顶层 run：做“只合不拆”的 tiny 碎片合并；内部 run 跳过
        let already_inside = POST_REPARTITION_GUARD.swap(true, Ordering::SeqCst);
        let mut out = result;
        if !already_inside {
            let opt = ShortcutRepartitionOptions {
                small_upper_bound: 3, // 只合 2–3 点碎片
                k_neighbors: 200,     // 每个碎片连 200 个最近碎片
                r_limit: 6000,        // 多源 Dijkstra 半径（0=两跳BFS）
            };
            out = shortcut_local_repartition_tiny_only(self.global, self.theta, out, opt);
            POST_REPARTITION_GUARD.store(false, Ordering::SeqCst);
        }

        // ====== 覆盖检测 ======
        let edges_global = {
            let mut s = HashSet::new();
            for v in self.global.vertices() {
                for (u, _) in self.global.get_out_adjacent_edges(&v) {
                    s.insert(pair_in_order((v, u)));
                }
            }
            s
        };

        let mut edges_partition = HashSet::new();
        let mut counter = HashMap::<(VertexId, VertexId), usize>::new();
        for sg in &out {
            for v in sg.graph.vertices() {
                for (u, _) in sg.graph.get_out_adjacent_edges(&v) {
                    let e = pair_in_order((v, u));
                    edges_partition.insert(e);
                    *counter.entry(e).or_default() += 1;
                }
            }
        }

        if edges_global != edges_partition {
            let missing: Vec<_> = edges_global.difference(&edges_partition).collect();
            let extra: Vec<_> = edges_partition.difference(&edges_global).collect();
            eprintln!(
                "❌ Edge coverage mismatch! missing {} edges, extra {} edges",
                missing.len(),
                extra.len()
            );
            if !missing.is_empty() {
                eprintln!("Example missing edge: {:?}", missing[0]);
            }
            if !extra.is_empty() {
                eprintln!("Example extra edge: {:?}", extra[0]);
            }
        } else {
            println!("✅ Edge coverage OK: {} edges", edges_global.len());
        }


        // ======================

        out
    }
}


// ==========================  只合并 2–3 点碎片：Kruskal-Shortcut  ==========================

#[derive(Clone, Copy)]
pub struct ShortcutRepartitionOptions {
    pub small_upper_bound: usize, // 传 3：只选 2–3 点碎片
    pub k_neighbors: usize,       // 每个碎片找 K 个最近碎片
    pub r_limit: EdgeWeight,      // 0=两跳BFS；>0=半径Dijkstra
}

/// 收集子图（无向）边集合 (min,max)
fn subgraph_edge_set(sg: &Subgraph) -> HashSet<(VertexId, VertexId)> {
    let mut s = HashSet::new();
    for v in sg.graph.vertices() {
        for (u, _w) in sg.graph.get_out_adjacent_edges(&v) {
            s.insert(pair_in_order((v, u)));
        }
    }
    s
}

/// 由边集合在全图上构造“边诱导子图”
fn build_graph_from_edge_set(global: &Graph, es: &HashSet<(VertexId, VertexId)>) -> Graph {
    let mut g = Graph::default();
    for &(u, v) in es {
        g.insert(u);
        g.insert(v);
        if let Some(w) = global.get_weight(u, v).or_else(|| global.get_weight(v, u)) {
            g.connect(u, v, w);
        }
    }
    g
}

/// 仅用于 KNN：边界点 -> 碎片下标（注意用 &Subgraph 以避免 clone）
fn build_small_boundary_index<'a>(small_subs: &[&'a Subgraph]) -> HashMap<VertexId, Vec<usize>> {
    let mut ix: HashMap<VertexId, Vec<usize>> = HashMap::new();
    for (i, sg) in small_subs.iter().enumerate() {
        for b in sg.bound_vertices.iter() {
            ix.entry(b).or_default().push(i);
        }
    }
    ix
}

/// 从碎片 sid 的所有边界点作为“多源”，在原图上找 K 个最近碎片
fn knn_small_neighbors<'a>(
    sid: usize,
    small_subs: &[&'a Subgraph],
    global: &Graph,
    b2small: &HashMap<VertexId, Vec<usize>>,
    k: usize,
    r_limit: EdgeWeight,
) -> Vec<(usize, EdgeWeight)> {
    let src_sg = small_subs[sid];
    if src_sg.bound_vertices.is_empty() { return vec![]; }

    let mut dist: HashMap<VertexId, EdgeWeight> = HashMap::new();
    let mut pq = BinaryHeap::new();
    for s in src_sg.bound_vertices.iter() {
        dist.insert(s, 0);
        pq.push(Reverse((0 as EdgeWeight, s)));
    }

    let mut best: HashMap<usize, EdgeWeight> = HashMap::new();
    while let Some(Reverse((d, v))) = pq.pop() {
        if d > *dist.get(&v).unwrap_or(&INF_WEIGHT) { continue; }
        if r_limit > 0 && d > r_limit { break; }

        if let Some(list) = b2small.get(&v) {
            for &oid in list {
                if oid != sid {
                    let e = best.entry(oid).or_insert(INF_WEIGHT);
                    if d < *e { *e = d; }
                }
            }
        }

        if !best.is_empty() && best.len() >= k {
            let worst = *best.values().max().unwrap();
            if d > worst { break; }
        }

        for (nv, w) in global.get_out_adjacent_edges(&v) {
            let (nd, overflow) = d.overflowing_add(w);
            if overflow { continue; }
            if nd < *dist.get(&nv).unwrap_or(&INF_WEIGHT) {
                dist.insert(nv, nd);
                pq.push(Reverse((nd, nv)));
            }
        }
    }

    let mut items: Vec<(usize, EdgeWeight)> = best.into_iter().collect();
    items.sort_by_key(|&(_, d)| d);
    items.truncate(k);
    items
}

/// DSU（并查集），按“顶点总数”做容量控制
struct DSU {
    parent: Vec<usize>,
    size_verts: Vec<usize>,
}
impl DSU {
    fn new(n: usize, sizes: &[usize]) -> Self {
        DSU { parent: (0..n).collect(), size_verts: sizes.to_vec() }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x { self.parent[x] = self.find(self.parent[x]); }
        self.parent[x]
    }
    fn unite_if_fit(&mut self, a: usize, b: usize, theta: usize) -> bool {
        let mut x = self.find(a);
        let mut y = self.find(b);
        if x == y { return false; }
        let new_sz = self.size_verts[x] + self.size_verts[y];
        if new_sz > theta { return false; }
        if self.size_verts[x] < self.size_verts[y] { std::mem::swap(&mut x, &mut y); }
        self.parent[y] = x;
        self.size_verts[x] = new_sz;
        true
    }
}

/// 只合并 2–3 点碎片：Kruskal-Shortcut 合并（只合不拆，不跑 VFIP，不引入残余）
pub fn shortcut_local_repartition_tiny_only(
    global: &Graph,
    theta: usize,
    mut subgraphs: Vec<Subgraph>,
    opt: ShortcutRepartitionOptions,
) -> Vec<Subgraph> {
    // 1) 拆分：只挑选 |SG|<=3 的碎片，其它原样保留
    let mut tiny: Vec<(usize, Subgraph)> = Vec::new(); // (原序, 子图)
    let mut keep: Vec<Subgraph> = Vec::new();
    for (idx, sg) in subgraphs.drain(..).enumerate() {
        if sg.graph.vertices().count() <= 3 { tiny.push((idx, sg)); }
        else { keep.push(sg); }
    }
    if tiny.is_empty() { return keep; }

    // 2) 近邻图：对 tiny 之间做 KNN（原图上）
    let tiny_refs: Vec<&Subgraph> = tiny.iter().map(|(_, sg)| sg).collect();
    let sizes: Vec<usize> = tiny_refs.iter().map(|sg| sg.graph.vertices().count()).collect();
    let b2small = build_small_boundary_index(&tiny_refs);

    let mut edges: Vec<(usize, usize, EdgeWeight)> = Vec::new();
    for sid in 0..tiny_refs.len() {
        let nbrs = knn_small_neighbors(sid, &tiny_refs, global, &b2small, opt.k_neighbors, opt.r_limit);
        for (oid, d) in nbrs { edges.push((sid, oid, d)); }
    }
    // 去重为无向边（sid < oid）
    let mut seen = HashSet::new();
    let mut undirected = Vec::new();
    edges.sort_by_key(|e| e.2);
    for (a, b, d) in edges {
        let (x, y) = if a < b { (a, b) } else { (b, a) };
        if x == y { continue; }
        if seen.insert((x, y)) { undirected.push((x, y, d)); }
    }

    // 3) DSU 按距离合并：仅当“顶点总数 ≤ θ”
    let n = tiny_refs.len();
    let mut dsu = DSU::new(n, &sizes);
    for (u, v, _d) in &undirected { let _ = dsu.unite_if_fit(*u, *v, theta); }

    // 4) 收集每个簇 → 真实边并集 → 直接建一个新子图（允许多连通分量，不跑 VFIP）
    let mut comp: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n { let r = dsu.find(i); comp.entry(r).or_default().push(i); }

    let mut merged: Vec<Subgraph> = Vec::new();
    let mut taken_tiny = vec![false; n];
    for (_r, members) in comp {
        if members.len() <= 1 { continue; } // 单碎片不合并
        // 真实边并集
        let mut e_union: HashSet<(VertexId, VertexId)> = HashSet::new();
        for sid in &members { e_union.extend(subgraph_edge_set(tiny_refs[*sid])); }
        if e_union.is_empty() { continue; }

        let mut sg = Subgraph::default();
        sg.graph = build_graph_from_edge_set(global, &e_union);

        // 边界/内部：内部=其 incident 全在 sg.graph
        for v in sg.graph.vertices() { sg.bound_vertices.insert(v); }
        let verts: Vec<VertexId> = sg.graph.vertices().collect();
        for v in verts {
            let mut ok = true;
            for (adj, _) in global.get_out_adjacent_edges(&v) {
                if !(sg.graph.contains(adj) && sg.graph.has_edge(adj, v)) { ok = false; break; }
            }
            if ok { sg.turn_to_internal_vertex(v); }
        }

        // 距离矩阵（小规模执行 Dijkstra）
        for s in sg.graph.vertices() {
            let mut dist: HashMap<VertexId, EdgeWeight> = HashMap::new();
            let mut pq = BinaryHeap::new();
            dist.insert(s, 0);
            pq.push(Reverse((0 as EdgeWeight, s)));
            while let Some(Reverse((d, v))) = pq.pop() {
                if d > *dist.get(&v).unwrap_or(&INF_WEIGHT) { continue; }
                for (u, w) in sg.graph.get_out_adjacent_edges(&v) {
                    let (nd, overflow) = d.overflowing_add(w);
                    if overflow { continue; }
                    if nd < *dist.get(&u).unwrap_or(&INF_WEIGHT) {
                        dist.insert(u, nd);
                        pq.push(Reverse((nd, u)));
                    }
                }
            }
            for (t, d) in dist {
                if s == t {
                    continue;
                }
                sg.distance[(s, t)] = d;
                sg.distance[(t, s)] = d;
            }
        }

        merged.push(sg);
        for sid in members { taken_tiny[sid] = true; }
    }

    // 5) 输出：保留未被合并的碎片 + 合并后的子图 + 其它大子图；统一编号 & 覆盖守恒
    for (i, (_idx, sg)) in tiny.into_iter().enumerate() {
        if !taken_tiny[i] { keep.push(sg); }
    }
    keep.extend(merged.into_iter());
    for (i, sg) in keep.iter_mut().enumerate() { sg.id = i; }

    debug_assert_eq!(
        keep.iter().map(|sg| sg.graph.edges().count()).sum::<usize>(),
        global.edges().count(),
        "edge coverage mismatch after tiny-only shortcut merge"
    );
    keep
}

#[cfg(test)]
mod test {
    use crate::distance::{DijkstraQuery, QueryDistance};
    use crate::graph::Graph;
    use crate::loader::load_graph;
    use crate::partition::{shortcut_local_repartition_tiny_only, ShortcutRepartitionOptions, VfipPartition};

    fn load_graph_on_paper() -> Graph {
        load_graph("../dataset/on-paper.gr").expect("failed to load on-paper.gr")
    }
    fn load_graph_231028() -> Graph {
        load_graph("../dataset/test-231028.gr").expect("failed to load test-231028.gr")
    }
    fn load_graph_new_york() -> Graph {
        load_graph("../dataset/USA-road-d.NY.gr").expect("failed to load USA-road-d.NY.gr")
    }

    #[test]
    fn test_vfip() {
        let graph = load_graph_on_paper();
        let subgraphs = VfipPartition::new(&graph, 8).run(1);
        assert_eq!(subgraphs.len(), 4);

        let graph = load_graph_231028();
        let subgraphs = VfipPartition::new(&graph, 8).run(1);
        assert_eq!(subgraphs.len(), 6);
    }

    #[test]
    fn test_subgraph_distance() {
        let graph = load_graph_new_york();
        let subgraphs = VfipPartition::new(&graph, 30).run(1);
        for sg in subgraphs {
            for v1 in sg.vertices() {
                let dijkstra = DijkstraQuery::new(&sg.graph, v1).build();
                for (v2, d) in sg.distance.distance_from(v1) {
                    assert_eq!(dijkstra.query(v2), d);
                }
            }
        }
    }

    #[test]
    fn test_tiny_merge_coverage() {
        let graph = load_graph_new_york();
        let subgraphs = VfipPartition::new(&graph, 30).run(1);
        let opt = ShortcutRepartitionOptions { small_upper_bound: 3, k_neighbors: 200, r_limit: 6000 };
        let merged = shortcut_local_repartition_tiny_only(&graph, 30, subgraphs, opt);
        let total_edges: usize = merged.iter().map(|sg| sg.graph.edges().count()).sum();
        assert_eq!(total_edges, graph.edges().count());
    }
}
