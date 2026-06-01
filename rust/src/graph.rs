use std::fmt::{Debug, Formatter};

use bincode::de::{BorrowDecoder, Decoder};
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{BorrowDecode, Decode};

use crate::transaction::{TNestedMap, TSet, Transaction};

pub type VertexId = u32;

/// 描述边的权重
pub type EdgeWeight = u64;

/// 描述可能出现小数的路径的长度. 如果 bool = true, 则表示值后面有 0.5, 否则是一个整数.
pub type MayHalfPathWeight = (EdgeWeight, bool);

pub const INF_WEIGHT: EdgeWeight = EdgeWeight::MAX;
pub const INVALID_VERTEX: VertexId = 0;
pub const MAX_VERTEX: VertexId = VertexId::MAX;

pub type Graph = DirectedGraph;
#[derive(Default)]
/// 有向图
pub struct DirectedGraph {
    /// 顶点集合
    vertices: TSet<VertexId>,
    /// 边集.
    edge_map: TNestedMap<VertexId, VertexId, EdgeWeight>,

    /// 边数
    edge_count: usize,
    /// 开启事务前的边数
    edge_count_before: usize,

    /// 事务状态
    transaction_enabled: bool,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            vertices: Default::default(),
            edge_map: Default::default(),
            edge_count: 0,
            edge_count_before: 0,
            transaction_enabled: false,
        }
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    pub fn add_edge(
        &mut self,
        v1: VertexId,
        v2: VertexId,
        weight: EdgeWeight,
    ) -> Option<EdgeWeight> {
        assert_ne!(v1, v2);
        debug_assert!(self.vertices.contains(&v1));
        debug_assert!(self.vertices.contains(&v2));

        match self.edge_map.insert(v1, v2, weight) {
            None => {
                self.edge_count += 1;
                None
            }
            Some(old_weight) => Some(old_weight),
        }
    }

    /// 连接两个顶点（添加一条无向边）
    pub fn connect(&mut self, v1: VertexId, v2: VertexId, weight: EdgeWeight) {
        self.add_edge(v1, v2, weight);
        self.add_edge(v2, v1, weight);
    }

    /// 删除一条边
    pub fn remove_edge(&mut self, v1: VertexId, v2: VertexId) {
        self.edge_map.remove(&v1, &v2)
    }

    /// 删除连接两点的无向边
    pub fn disconnect(&mut self, v1: VertexId, v2: VertexId) {
        self.remove_edge(v1, v2);
        self.remove_edge(v2, v1);
    }

    /// 判断两顶点间是否有边
    pub fn has_edge(&self, v1: VertexId, v2: VertexId) -> bool {
        assert_ne!(v1, v2);

        self.get_weight(v1, v2).is_some() || self.get_weight(v2, v1).is_some()
    }

    /// 查询边长度, 若不存在则返回 None
    pub fn get_weight(&self, v1: VertexId, v2: VertexId) -> Option<EdgeWeight> {
        assert_ne!(v1, v2);

        self.edge_map.get(&v1, &v2).copied()
    }

    /// 替换边的权值
    pub fn replace(
        &mut self,
        v1: VertexId,
        v2: VertexId,
        new_weight: EdgeWeight,
    ) -> Option<EdgeWeight> {
        assert_ne!(v1, v2);

        if let Some(old_value) = self.get_weight(v1, v2) {
            self.disconnect(v1, v2);
            self.connect(v1, v2, new_weight);

            Some(old_value)
        } else {
            None
        }
    }

    /// 获取某一顶点的边, 只返回出方向的数据.
    pub fn get_out_adjacent_edges<'a>(
        &'a self,
        v: &'a VertexId,
    ) -> impl Iterator<Item=(VertexId, EdgeWeight)> + 'a {
        self.edge_map
            .get_tier2_map_iter(v)
            .map(move |(v, w)| (*v, *w))
    }

    /// 得到图中的顶点数
    pub fn size(&self) -> usize {
        self.vertices.len()
    }

    /// 得到图中的边数. 注意, 这个 Graph 被视为一个有向图, 当其存储无向图时, 返回的边数可能介于 |E| 和 2|E| 之间.
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// 添加顶点
    pub fn insert(&mut self, v: VertexId) -> bool {
        self.vertices.insert(v)
    }

    /// 得到所有的边
    pub fn edges(&self) -> impl Iterator<Item=((VertexId, VertexId), EdgeWeight)> + '_ {
        self.edge_map.iter().map(|(v1, v2, w)| ((*v1, *v2), *w))
    }

    /// 得到顶点的迭代器
    pub fn vertices(&self) -> impl Iterator<Item=VertexId> + '_ {
        self.vertices.iter()
    }

    /// 判断图中是否包含某个顶点
    pub fn contains(&self, v: VertexId) -> bool {
        self.vertices.contains(&v)
    }
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        assert!(
            !self.transaction_enabled,
            "Graph can only be cloned outside transaction mode."
        );
        Self {
            vertices: self.vertices.clone(),
            edge_map: self.edge_map.clone(),
            edge_count: self.edge_count,
            edge_count_before: 0,
            transaction_enabled: false,
        }
    }
}

impl Debug for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let vertex_count = self.size();
        let edge_count = self.edge_count();

        writeln!(
            f,
            "Graph (with {vertex_count} vertices, {edge_count} edges)"
        )?;
        write!(f, " V = {{", )?;
        let mut vertices = self.vertices.iter().collect::<Vec<_>>();
        vertices.sort();
        for (index, vertex) in vertices.iter().enumerate() {
            if index != 0 {
                write!(f, ", {vertex}")?;
            } else {
                write!(f, "{vertex}")?;
            }
        }
        writeln!(f, "}}")?;

        write!(f, " E = {{", )?;
        let mut edges = self.edges().collect::<Vec<_>>();
        edges.sort();
        for (index, ((v1, v2), _)) in edges.iter().enumerate() {
            // 为了简化输出结果, 一条边对应的一组数据（v1-v2, v2-v1）只输出一个.
            if v1 > v2 {
                continue;
            }
            if index != 0 {
                write!(f, ", ({v1}, {v2})")?;
            } else {
                write!(f, "({v1}, {v2})")?;
            }
        }
        writeln!(f, "}}")?;

        Ok(())
    }
}

impl Transaction for Graph {
    fn begin(&mut self) {
        debug_assert!(!self.transaction_enabled);
        self.edge_map.begin();
        self.vertices.begin();
        self.edge_count_before = self.edge_count;
        self.transaction_enabled = true;
    }

    fn rollback(&mut self) {
        debug_assert!(self.transaction_enabled);
        self.edge_map.rollback();
        self.vertices.rollback();
        self.edge_count = self.edge_count_before;
        self.edge_count_before = 0;
        self.transaction_enabled = false;
    }

    fn commit(&mut self) {
        debug_assert!(self.transaction_enabled);
        self.edge_map.commit();
        self.vertices.commit();
        self.edge_count_before = 0;
        self.transaction_enabled = false;
    }
}

impl bincode::Encode for Graph {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        // 1. 写入顶点列表
        self.vertices.encode(encoder)?;
        // 2. 写入边列表. Graph 是有向图, 直接存
        self.edge_map.encode(encoder)
    }
}

impl<Context> bincode::Decode<Context> for Graph {
    fn decode<D: Decoder<Context=Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        // 1. 读取顶点列表
        let vertices = TSet::decode(decoder)?;
        // 2. 读取边列表
        let edge_map = TNestedMap::decode(decoder)?;
        // 3. 边数
        let edge_count = edge_map.len();

        Ok(Self {
            vertices,
            edge_map,
            edge_count,
            edge_count_before: 0,
            transaction_enabled: false,
        })
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Graph {
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Graph::decode(decoder)
    }
}

#[cfg(test)]
mod test {
    use crate::graph::Graph;
    use crate::transaction::Transaction;

    #[test]
    fn test_graph_transaction() {
        let mut g = Graph::new();

        g.insert(1);
        g.insert(2);
        g.insert(3);
        g.connect(1, 2, 1);
        g.connect(3, 2, 1);

        assert_eq!(g.size(), 3);
        assert_eq!(g.edges().count(), 4);
        assert_eq!(g.edge_count, 4);
        g.begin();
        g.connect(1, 3, 1);
        assert_eq!(g.edges().count(), 6);
        assert_eq!(g.edge_count, 6);

        g.rollback();
        assert_eq!(g.edges().count(), 4);
        assert_eq!(g.edge_count, 4);
    }

    #[test]
    fn test_graph_replace_edge() {
        let mut g = Graph::new();

        g.insert(1);
        g.insert(2);
        g.insert(3);
        g.connect(1, 2, 1);
        g.connect(3, 2, 1);
        assert_eq!(g.get_weight(1, 2), Some(1));
        assert_eq!(g.get_weight(3, 2), Some(1));

        let old_weight = g.replace(1, 3, 5);
        assert_eq!(old_weight, None);
        assert_eq!(g.get_weight(1, 3), None);

        g.replace(1, 2, 3);
        assert_eq!(g.get_weight(1, 2), Some(3));
        assert_eq!(g.get_weight(2, 1), Some(3));
    }
}
