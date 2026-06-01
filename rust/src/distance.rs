use crate::graph::{EdgeWeight, Graph, VertexId, INF_WEIGHT};
use crate::matrix::{Array, ArrayBuilder};

#[derive(Debug)]
pub struct DijkstraQuery<'a> {
    graph: &'a Graph,
    source: VertexId,
}

#[derive(Debug)]
pub struct DijkstraResult<'a> {
    graph: &'a Graph,
    distance: Array,
}

#[derive(Debug)]
pub struct DijkstraProcess<'a> {
    graph: &'a Graph,
    source: VertexId,
    distance: Array,
    queue: pheap::PairingHeap<VertexId, EdgeWeight>,
}

impl<'a> DijkstraQuery<'a> {
    pub fn new(graph: &'a Graph, source: VertexId) -> Self {
        Self {
            graph,
            source,
        }
    }

    pub fn build(self) -> DijkstraResult<'a> {
        // 使用 Reverse 构造小根堆.
        let mut queue = pheap::PairingHeap::<VertexId, EdgeWeight>::new();
        let mut distance = ArrayBuilder::new()
            .from(self.source)
            .capacity(self.graph.size())
            .build();

        queue.insert(self.source, 0);
        while let Some((u, d)) = queue.delete_min() {
            if d > distance.get_or_inf(u) {
                continue;
            }

            for (adjacent_vertex, adjacent_weight) in self.graph.get_out_adjacent_edges(&u) {
                // 如果通过点 u 绕路的路径（distance[u] + adjacent_weight）小于原来直接从 source 到 adjacent_vertex 的长度
                if d == INF_WEIGHT {
                    continue;
                }
                let new_distance = d + adjacent_weight;
                if new_distance < distance.get_or_inf(adjacent_vertex) {
                    // 松弛操作
                    distance[adjacent_vertex] = new_distance;
                    // 该点相关的距离受到影响，需要后续进一步更新
                    queue.insert(adjacent_vertex, new_distance);
                }
            }
        }

        DijkstraResult {
            graph: self.graph,
            distance,
        }
    }

    pub fn continuous_query(self) -> DijkstraProcess<'a> {
        let mut queue = pheap::PairingHeap::<VertexId, EdgeWeight>::new();
        queue.insert(self.source, 0);

        let distance = ArrayBuilder::new()
            .from(self.source)
            .capacity(self.graph.size())
            .build();
        DijkstraProcess {
            graph: self.graph,
            source: self.source,
            distance,
            queue,
        }
    }

    pub fn custom_query(self) -> DijkstraProcess<'a> {
        let queue = pheap::PairingHeap::<VertexId, EdgeWeight>::new();

        let distance = ArrayBuilder::new()
            .from(self.source)
            .capacity(self.graph.size())
            .build();
        DijkstraProcess {
            graph: self.graph,
            source: self.source,
            distance,
            queue,
        }
    }
}

impl DijkstraProcess<'_> {
    pub fn give_vertex(&mut self, v: VertexId, distance: EdgeWeight) {
        self.queue.insert(v, distance);
    }

    pub fn query(&mut self, destination: VertexId) -> EdgeWeight {
        if let Some(weight) = self.distance.get(destination) {
            return weight;
        }

        while let Some((u, d)) = self.queue.delete_min() {
            if d > self.distance.get_or_inf(u) {
                continue;
            }
            let mut return_flag = false;
            if u == destination {
                return_flag = true;
            }

            for (adjacent_vertex, adjacent_weight) in self.graph.get_out_adjacent_edges(&u) {
                // 如果通过点 u 绕路的路径（distance[u] + adjacent_weight）小于原来直接从 source 到 adjacent_vertex 的长度
                if d == INF_WEIGHT {
                    continue;
                }
                let new_distance = d + adjacent_weight;
                if new_distance < self.distance.get_or_inf(adjacent_vertex) {
                    // 松弛操作
                    self.distance[adjacent_vertex] = new_distance;
                    // 该点相关的距离受到影响，需要后续进一步更新
                    self.queue.insert(adjacent_vertex, new_distance);
                }
            }

            if return_flag {
                return d;
            }
        }
        INF_WEIGHT
    }
}

impl DijkstraResult<'_> {
    pub fn query(&self, destination: VertexId) -> EdgeWeight {
        self.distance.get_or_inf(destination)
    }
}

pub struct BellmanFordQuery<'a> {
    graph: &'a Graph,
    source: VertexId,

    distance: Array,
}

impl<'a> BellmanFordQuery<'a> {
    pub fn new(graph: &'a Graph, source: VertexId) -> Self {
        debug_assert!(graph.contains(source));

        // 这里需要保证: 当前图已经被锁定, 不会再增加顶点
        let graph_size = graph.size();
        let distance = ArrayBuilder::new()
            .from(source)
            .capacity(graph_size)
            .build();

        Self {
            graph,
            source,
            distance,
        }
    }

    /// 进行松弛操作.
    ///
    /// 尝试从源点通过 u 而到达 v. 谨记, w 表示 $w_{u, v}$
    fn relax(&mut self, u: VertexId, v: VertexId, w: EdgeWeight) -> bool {
        // new_distance 表示截至目前 source 到 u 的最短路径, 加上 u 和 v 之间的权重.
        let (new_distance, is_overflowing) = self.distance.get_or_inf(u).overflowing_add(w);

        // 若 source 到 v 的距离, 经过 u 会变小, 则更新 source 到 v 的距离.
        if !is_overflowing && self.distance.get_or_inf(v) > new_distance {
            self.distance[v] = new_distance;
            true
        } else {
            false
        }
    }

    pub fn build(mut self) -> Self {
        // TODO: 使用队列优化性能.
        let mut modify_flag;
        for _ in 1..self.graph.edge_count() {
            modify_flag = false;

            for ((v1, v2), w) in self.graph.edges() {
                modify_flag |= self.relax(v1, v2, w);
            }

            if !modify_flag {
                break;
            }
        }
        self
    }
}

impl BellmanFordQuery<'_> {
    pub fn query(&self, destination: VertexId) -> EdgeWeight {
        self.distance.get_or_inf(destination)
    }
}

#[cfg(test)]
mod test {
    use crate::distance::{BellmanFordQuery, DijkstraQuery};
    use crate::graph::Graph;
    use crate::loader::load_graph;

    fn load_test_graph() -> Graph {
        load_graph("../dataset/on-paper.gr").expect("unable to load on-paper.gr for test.")
    }

    fn load_new_york_graph() -> Graph {
        load_graph("../dataset/USA-road-d.NY.gr")
            .expect("unable to load USA-road-d.NY.gr for test.")
    }

    #[test]
    fn test_dijkstra_implementation() {
        let graph = load_test_graph();
        let mut dijkstra = DijkstraQuery::new(&graph, 1).build();

        let distance = [
            0, 0, 2, 2, 4, 5, 6, 7, 11, 13, 14, 16, 15, 15, 16, 29, 17, 21, 25, 24, 23, 27, 28, 30,
            31, 35,
        ];
        for v in 1..=25 {
            assert_eq!(dijkstra.query(v), distance[v as usize]);
        }
    }

    #[test]
    fn test_dijkstra_implementation_2() {
        let mut graph = Graph::new();

        graph.insert(1);
        graph.insert(2);
        graph.insert(3);

        graph.connect(1, 3, 2);
        graph.connect(3, 2, 2);

        let mut distance = DijkstraQuery::new(&graph, 1).build();
        assert_eq!(distance.query(2), 4);
    }

    #[test]
    fn test_bellman_ford() {
        let graph = load_test_graph();
        let mut bellman_ford = BellmanFordQuery::new(&graph, 1).build();

        let distance = [
            0, 0, 2, 2, 4, 5, 6, 7, 11, 13, 14, 16, 15, 15, 16, 29, 17, 21, 25, 24, 23, 27, 28, 30,
            31, 35,
        ];
        for v in 1..=25 {
            assert_eq!(bellman_ford.query(v), distance[v as usize]);
        }
    }

    #[test]
    fn compare_two_algo_on_new_york() {
        let graph = load_new_york_graph();

        let mut bellman_ford = BellmanFordQuery::new(&graph, 1).build();
        let mut dijkstra = DijkstraQuery::new(&graph, 1).build();

        for v in graph.vertices() {
            assert_eq!(bellman_ford.query(v), dijkstra.query(v));
        }
    }

    #[test]
    fn test_continuous_dijkstra_on_new_york() {
        let graph = load_new_york_graph();

        let mut dijkstra = DijkstraQuery::new(&graph, 1).continuous_query();
        for v in graph.vertices() {
            assert_eq!(dijkstra.query(v), dijkstra.query(v));
        }
    }
}