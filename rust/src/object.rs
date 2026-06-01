use std::fmt::{Debug, Formatter};
use std::ops::{Deref, Index};

use ahash::{HashMap, HashMapExt};
use anyhow::{anyhow, Context, Result};
use bincode::{Decode, Encode};

use crate::graph::{EdgeWeight, Graph, VertexId};
use crate::pair::pair_in_order;
use crate::partition::{SgId, Subgraph};

pub type Edge = (VertexId, VertexId);

pub type Offset = EdgeWeight;

/// 移动对象编号
pub type ObjId = usize;

#[derive(PartialOrd, Ord, PartialEq, Eq, Copy, Clone, Encode, Decode)]
/// 边上的移动对象
pub struct MovingObject {
    /// 移动对象唯一标识
    pub unique_id: ObjId,
    /// 移动对象所在的边
    pub edge: Edge,
    /// 移动对象距离较小的顶点的距离
    pub offset: Offset,
}

impl MovingObject {
    pub fn new(id: ObjId, edge: Edge, offset: Offset) -> Self {
        Self {
            unique_id: id,
            edge,
            offset,
        }
    }

    pub fn query_point(edge: Edge, offset: Offset) -> Self {
        Self::new(0, edge, offset)
    }
}

impl Debug for MovingObject {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Object {} ({}, {}, {})",
            self.unique_id, self.edge.0, self.edge.1, self.offset
        )
    }
}

#[derive(Default, bincode::Encode, bincode::Decode)]
pub struct MovingObjectSet {
    inner: HashMap<ObjId, MovingObject>,
}

impl MovingObjectSet {
    pub fn emplace(&mut self, id: ObjId, edge: Edge, edge_weight: EdgeWeight) {
        self.push(MovingObject::new(id, edge, edge_weight))
    }

    pub fn push(&mut self, object: MovingObject) {
        self.inner.insert(object.unique_id, object);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item=&MovingObject> + '_ {
        self.inner.values()
    }

    pub fn remove(&mut self, id: ObjId) {
        self.inner.remove(&id);
    }

    pub fn contains(&self, id: ObjId) -> bool {
        self.inner.contains_key(&id)
    }

    pub fn contains_obj(&self, obj: &MovingObject) -> bool {
        self.iter().any(|o| o.edge == obj.edge && o.offset == obj.offset)
    }

    /// 在给定的图 graph 的边上随机生成 count 个移动对象.
    ///
    /// 在当前实现中, 移动对象的分布是均匀的.
    pub fn random(graph: &Graph, count: usize) -> Result<Self> {
        type EdgeWeight64 = usize;

        // 将全部边的长度加起来，得到一条路径长度，然后在路径上的整数位置（不含端点）添加随机移动点.
        // 则移动点的位置有 sum_i^e {w_i - 1} 种可能.
        // 有点类似高中数学-组合数学中的隔板法.
        let points = graph
            .edges()
            .filter(|((v1, v2), _)| v1 < v2)
            .map(|(_, w)| (w - 1) as EdgeWeight64)
            .sum::<EdgeWeight64>();

        if (points as usize) < count {
            let e =
                anyhow!("The candidates of edge is {points}, not enough to {count} moving objects.");
            return Err(e);
        }
        if (points as usize) == count {
            let mut last_obj_id = 0 as ObjId;
            // 所有位置都有移动点
            let inner = graph
                .edges()
                .filter(|((v1, v2), _)| v1 < v2)
                .flat_map(|((v1, v2), w)| {
                    let edge = pair_in_order((v1, v2));
                    last_obj_id += 1;

                    (1..w).map(move |offset| MovingObject::new(last_obj_id, edge, offset))
                })
                .fold(HashMap::new(), |mut map, obj| {
                    map.insert(obj.unique_id, obj);
                    map
                });
            return Ok(MovingObjectSet { inner });
        }
        /* Now `points` is larger than `count`. */

        /* 在可行的位置中, 随机取移动点 */
        use rand::seq::IteratorRandom;

        // 我们假设图中所有的边首尾相连, 一条长度为 3 的边有两个可以放置移动点的地方（1 和 2），两条这样的边有 4 个可以放移动点的地方.
        // v1 - [1] - [2] - v2 - [3] - [4] - v3, 其中 [x] 表示第 x 个移动点可以放的位置, v 表示顶点.
        // candidates 为升序的、所有选中的移动点位置.
        let mut rng = rand::thread_rng();
        let mut candidates = (1..=points).choose_multiple(&mut rng, count);
        candidates.sort();

        // 根据选好的移动点位置 candidates, 在图的边上放置移动点.
        let mut walked = 1 as EdgeWeight64;
        let mut i = 0usize; // candidates 下标
        let mut result = Self::default();

        // 1. Graph 是以有向图为基础的, 对于一个无向边, 其边会存储两次, 影响遍历效果. 因此, 通过 filter 去掉一半的边遍历;
        // 2. 对每条边遍历一次. 每一条边有 weight - 1 个候选位置, 这些位置在 1..range 上的偏移量是 walked, 对应的范围为
        //     walked..(walked + weight). 我们可以从 candidates 中取该范围中的位置, 并生成移动点.
        let mut index = 1 as ObjId;
        let unique_edges = graph.edges().filter(|((v1, v2), _)| v1 < v2);
        for ((v1, v2), w) in unique_edges {
            let w = w as EdgeWeight64;
            while i < candidates.len() && (walked..(walked + w - 1)).contains(&candidates[i]) {
                let edge = pair_in_order((v1, v2));
                let offset = candidates[i] + 1 - walked;
                result.emplace(index, edge, offset as EdgeWeight);

                i += 1;
                index += 1;
            }

            if i == candidates.len() {
                break;
            }

            walked += w - 1;
        }
        debug_assert!(result.len() == count);
        Ok(result)
    }
}

impl Index<ObjId> for MovingObjectSet {
    type Output = MovingObject;

    fn index(&self, index: ObjId) -> &Self::Output {
        &self.inner[&index]
    }
}

impl Debug for MovingObjectSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (index, object) in self.inner.values().enumerate() {
            if index == 0 {
                write!(f, "{object:?}")?;
            } else {
                write!(f, ", {object:?}")?;
            }
        }
        write!(f, "}}")
    }
}

/// 维护了 "子图", "边" 和 "移动对象" 关系的**移动对象集合**
pub struct IndexedMovingObjectSet {
    /// 维护一个从 "边" 到子图编号（即，子图在数组中的下标）的映射
    /// 以便快速地找到移动点对应的子图
    edge_subgraph: HashMap<Edge, ObjId>,
    /// 子图上的移动点映射, 维护了 "子图下标" 到 “移动点编号数组” 的关系.
    /// 用于在 fc_subgraphs 结果中快速找到符合条件的移动点.
    subgraph_objects: HashMap<SgId, Vec<ObjId>>,
    /// "边" 对 "移动点列表" 的映射
    edge_objects: HashMap<Edge, Vec<ObjId>>,

    /// 移动对象集合
    inner: MovingObjectSet,
}

impl IndexedMovingObjectSet {
    pub fn new(subgraphs: &Vec<Subgraph>, raw: MovingObjectSet) -> Self {
        type EdgeWeight64 = usize;
        // 建立边 -> 子图的映射
        let mut edge_subgraph = HashMap::new();
        let mut range = 0 as EdgeWeight64;
        for sg in subgraphs {
            for (edge, w) in sg.edges().filter(|((v1, v2), _)| *v1 < *v2) {
                edge_subgraph.insert(edge, sg.id);
                range += w as EdgeWeight64 - 1;
            }
        }

        if range < raw.len() {
            panic!("moving objects can not all be put onto the graph.");
        }

        // 建立 "子图" -> "移动点数组" 的映射
        use std::collections::hash_map::Entry;

        let mut subgraph_objects: HashMap<SgId, Vec<ObjId>> = HashMap::new();
        let mut edge_objects: HashMap<Edge, Vec<ObjId>> = HashMap::new();
        for object in raw.iter() {
            let sg_index = edge_subgraph
                .get(&object.edge)
                .copied()
                .with_context(|| format!("The edge {:?} is not existed.", object.edge))
                .unwrap();

            match subgraph_objects.entry(sg_index) {
                Entry::Occupied(mut e) => (*e.get_mut()).push(object.unique_id),
                Entry::Vacant(e) => {
                    e.insert(vec![object.unique_id]);
                }
            }

            match edge_objects.entry(object.edge) {
                Entry::Occupied(mut e) => (*e.get_mut()).push(object.unique_id),
                Entry::Vacant(e) => {
                    e.insert(vec![object.unique_id]);
                }
            }
        }

        Self {
            edge_subgraph,
            subgraph_objects,
            edge_objects,
            inner: raw,
        }
    }

    pub fn find_edge(&self, edge: &Edge) -> Option<SgId> {
        self.edge_subgraph.get(edge).copied()
    }

    pub fn emplace(&mut self, id: ObjId, edge: Edge, edge_weight: EdgeWeight) {
        self.push(MovingObject::new(id, edge, edge_weight))
    }

    pub fn push(&mut self, object: MovingObject) {
        use std::collections::hash_map::Entry;

        let sg_index = *self
            .edge_subgraph
            .get(&object.edge)
            .expect("The edge is not existed.");

        match self.subgraph_objects.entry(sg_index) {
            Entry::Occupied(mut e) => (*e.get_mut()).push(object.unique_id),
            Entry::Vacant(e) => {
                e.insert(vec![object.unique_id]);
            }
        }
        self.inner.push(object);
    }

    pub fn iter(&self) -> impl Iterator<Item=&MovingObject> + '_ {
        self.inner.iter()
    }

    pub fn objects_in(&self, subgraph: SgId) -> Box<dyn Iterator<Item=ObjId> + '_> {
        if let Some(vec) = self.subgraph_objects.get(&subgraph) {
            Box::new(vec.iter().copied())
        } else {
            Box::new(std::iter::empty())
        }
    }

    pub fn objects_on(&self, edge: Edge) -> Box<dyn Iterator<Item=ObjId> + '_> {
        if let Some(vec) = self.edge_objects.get(&edge) {
            Box::new(vec.iter().copied())
        } else {
            Box::new(std::iter::empty())
        }
    }

    pub fn count_objects_in(&self, subgraph: SgId) -> usize {
        if let Some(vec) = self.subgraph_objects.get(&subgraph) {
            vec.len()
        } else {
            0
        }
    }
}

impl Deref for IndexedMovingObjectSet {
    type Target = MovingObjectSet;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(test)]
mod test {
    use crate::graph::Graph;
    use crate::loader::{export_moving_objects, load_graph, load_moving_objects};
    use crate::object::{MovingObject, MovingObjectSet};

    #[test]
    fn test_generate_moving_objects_1() {
        let mut graph = Graph::new();

        /* 在 3 个候选位置中插入 3 个移动点. */
        graph.insert(1);
        graph.insert(2);
        graph.insert(3);
        graph.connect(1, 2, 3);
        graph.connect(3, 2, 2);

        let moving_objects = MovingObjectSet::random(&graph, 3).unwrap();
        for MovingObject { edge, offset, .. } in moving_objects.iter() {
            match edge {
                (1, 2) => assert!(*offset == 1 || *offset == 2),
                (2, 3) => assert_eq!(*offset, 1),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_generate_moving_objects_2() {
        let mut graph = Graph::new();

        /* 在 3 个候选位置中插入 4 个移动点. */
        graph.insert(1);
        graph.insert(2);
        graph.insert(3);
        graph.connect(1, 2, 3);
        graph.connect(3, 2, 2);

        let _ = MovingObjectSet::random(&graph, 4).unwrap();
    }

    #[test]
    fn test_generate_moving_objects_3() {
        let mut graph = Graph::new();

        /* 在 4 个候选位置中插入 2 个移动点. */
        graph.insert(1);
        graph.insert(2);
        graph.insert(3);
        graph.connect(1, 2, 3);
        graph.connect(3, 2, 3);

        let moving_objects = MovingObjectSet::random(&graph, 2).unwrap();
        for MovingObject { edge, offset, .. } in moving_objects.iter() {
            match edge {
                (1, 2) => assert!((1..3).contains(offset)),
                (2, 3) => assert!((1..3).contains(offset)),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_objects_generation() {
        let graph = load_graph("../dataset/USA-road-d.NY.gr").unwrap();
        let moving_objects = MovingObjectSet::random(&graph, 10000).unwrap();

        assert_eq!(moving_objects.len(), 10000);
        for MovingObject { edge, offset, .. } in moving_objects.iter() {
            let &(v1, v2) = edge;
            assert!(graph.has_edge(v1, v2));

            let w = graph.get_weight(v1, v2).unwrap();
            assert!((1..w).contains(offset));
        }
    }


    #[test]
    fn test_objects_save_load() {
        const GRAPH: &str = "../dataset/USA-road-d.NY.gr";
        const TEMP_FILE: &str = "/tmp/USA-road-d-test.NY.mos.gr";
        const LEN: usize = 1;

        let graph = load_graph(GRAPH).unwrap();
        let set1 = MovingObjectSet::random(&graph, LEN).unwrap();

        assert_eq!(set1.len(), LEN);
        println!("{:?}", set1.iter().collect::<Vec<_>>());

        let set2 = {
            export_moving_objects(TEMP_FILE, &set1).unwrap();
            let result = load_moving_objects(TEMP_FILE).unwrap();
            std::fs::remove_file(TEMP_FILE).unwrap();

            result
        };
        assert_eq!(set2.len(), LEN);
        println!("{:?}", set2.iter().collect::<Vec<_>>());

        for obj in set2.iter() {
            println!("{:?}", obj);
            assert!(set1.contains_obj(obj));
        }
    }
}
