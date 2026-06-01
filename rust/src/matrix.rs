use std::collections::hash_map::Entry;
use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut, Index, IndexMut};

use ahash::{HashMap, HashMapExt};
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{BorrowDecode, Decode, Encode};

use crate::graph::{EdgeWeight, VertexId, INF_WEIGHT};
use crate::transaction::Transaction;

const DEFAULT_ARRAY_CAPACITY: usize = 20;

#[derive(Default)]
pub struct ArrayBuilder {
    source: Option<VertexId>,
    capacity: Option<usize>,
}

impl ArrayBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from(mut self, source: VertexId) -> Self {
        self.source = Some(source);
        self
    }

    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }

    pub fn build(self) -> Array {
        let source = self
            .source
            .expect("source has not been set yet, please call ArrayBuilder::from() before build.");
        let capacity = self.capacity.unwrap_or(DEFAULT_ARRAY_CAPACITY);

        Array {
            source,
            map: HashMap::with_capacity(capacity),
        }
    }
}

/// 从单顶点到其他顶点的距离（一维数组）
pub struct Array {
    /// 源点
    source: VertexId,
    /// 记录顶点到源点的距离.
    map: HashMap<VertexId, EdgeWeight>,
}

impl Array {
    pub fn contains(&self, target: &VertexId) -> bool {
        self.map.contains_key(target)
    }

    /// 设置 source 到 target 之间的最短距离.
    pub fn set(&mut self, target: VertexId, weight: EdgeWeight) {
        assert_ne!(self.source, target);

        self.map.insert(target, weight);
    }

    /// 取 source 到 target 的距离.
    pub fn get(&self, target: VertexId) -> Option<EdgeWeight> {
        if self.source == target {
            return Some(0);
        }

        self.map.get(&target).copied()
    }

    /// 取 source 到 target 的距离, 若不存在, 返回 InfWeight.
    pub fn get_or_inf(&self, target: VertexId) -> EdgeWeight {
        self.get(target).unwrap_or(INF_WEIGHT)
    }
}

impl Debug for Array {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Distance array (from v_{})", self.source)?;
        write!(f, "[")?;
        for (i, (&target, &distance)) in self.map.iter().enumerate() {
            if i == 0 {
                write!(f, "{{v_{}: {}}}", target, distance)?;
            } else {
                write!(f, ", {{v_{}: {}}}", target, distance)?;
            }
        }
        write!(f, "]")
    }
}

impl Index<VertexId> for Array {
    type Output = EdgeWeight;

    fn index(&self, index: VertexId) -> &Self::Output {
        if self.source == index {
            return &0;
        }

        self.map
            .get(&index)
            .expect("A valid, contained vertex is expected.")
    }
}

impl IndexMut<VertexId> for Array {
    fn index_mut(&mut self, index: VertexId) -> &mut Self::Output {
        if self.source == index {
            panic!("Could not modify distance from one vertex to itself, which is always zero.");
        }

        match self.map.entry(index) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(INF_WEIGHT),
        }
    }
}

const DEFAULT_MATRIX_SIZE: usize = 30;

/// 这里的 Matrix 是论文 4.3 中提到的最短距离矩阵.
/// > The shortest distances between vertices within 𝑆𝐺𝑒 are stored in a matrix D of size 𝑚 × 𝑚, where 𝑚 is the
/// > number of vertices in 𝑆𝐺𝑒 , which increases as new vertices are added to 𝑆𝐺𝑒.
///
/// 对于 AIP 算法，由于该矩阵的大小是不确定的，需要考虑动态扩容.
/// 对于 VFIP 算法，可以按照 theta 的大小预留空间.
/// 注意，子图中的顶点终究远少于整体的顶点，所以 Matrix 内部会记录当前加入的顶点与 array 下标的对应关系.
///
/// 在实现上, 由于该最短距离矩阵是一个对称矩阵, 因此可以压缩为一个行优先存储的下三角矩阵. 且, 由于主对角线上的元素表示顶点到自身的距离,
/// 值始终为 0, 还可以省略.
///
/// 对于这个数据结构, 需要注意的是: 我们区分了顶点编号（VertexId）、邻接矩阵的下标（usize）以及矩阵在一维 Vec 上存储时的下标 pos（usize）.
#[derive(Clone)]
pub struct Matrix {
    /// 记录顶点与矩阵下标的对应关系. VertexId 从 1 开始, usize 也是从 1 开始的.
    map: HashMap<VertexId, usize>,
    /// 矩阵数据的存储空间
    array: Vec<EdgeWeight>,
    /// 矩阵（方阵）当前存储的顶点数量
    n: usize,
    /// 最大可以存储的顶点数量
    capacity: usize,
}

impl Default for Matrix {
    fn default() -> Self {
        Self::new()
    }
}

impl Matrix {
    /// 新建一个空的最短距离矩阵
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MATRIX_SIZE)
    }

    /// 新建一个 n * n 的矩阵, 并立即分配空间.
    pub fn with_capacity(capacity: usize) -> Self {
        let size = capacity * (capacity - 1) / 2;
        Self {
            array: vec![INF_WEIGHT; size],
            map: Default::default(),
            n: 0,
            capacity,
        }
    }

    #[inline]
    /// 计算矩阵中下标 `(i, j)` 的元素在一维空间上的实际位置
    fn pos(i: usize, j: usize) -> usize {
        // 保证 i >= j
        let (i, j) = if i < j { (j, i) } else { (i, j) };

        // i, j 从 1 开始计算, 而 pos 从 0 开始计算.
        (i - 2) * (i - 1) / 2 + j - 1
    }

    pub fn contains(&self, v: VertexId) -> bool {
        self.map.contains_key(&v)
    }

    /// 查询 v_i 和 v_j 间的最短距离
    ///
    /// 如果 v_i 或 v_j 不存在于矩阵中, 函数将 panic.
    pub fn get(&self, v_i: VertexId, v_j: VertexId) -> EdgeWeight {
        if v_i == v_j {
            debug_assert!(self.map.contains_key(&v_i));
            0
        } else {
            let i = *self.map.get(&v_i).unwrap();
            let j = *self.map.get(&v_j).unwrap();
            self.array[Self::pos(i, j)]
        }
    }

    /// 查询 v_i 和 v_j 间的最短距离
    ///
    /// 如果 v_i 或 v_j 不存在于矩阵中, 函数将返回 INF_WEIGHT.
    pub fn get_or_inf(&self, v_i: VertexId, v_j: VertexId) -> EdgeWeight {
        if !self.map.contains_key(&v_i) || !self.map.contains_key(&v_j) {
            return INF_WEIGHT;
        }

        if v_i == v_j {
            0
        } else {
            let (i, j) = (self.map.get(&v_i).unwrap(), self.map.get(&v_j).unwrap());
            self.array[Self::pos(*i, *j)]
        }
    }

    fn get_index_or_insert(&mut self, v: VertexId) -> usize {
        match self.map.entry(v) {
            Entry::Occupied(v) => *v.get(),
            Entry::Vacant(entry) => {
                let index = self.n + 1;
                self.n += 1;

                entry.insert(index);
                index
            }
        }
    }

    /// 扩大矩阵的存储空间.
    ///
    /// 函数将自动完成旧数据的拷贝, 并将新的区域用 INF_WEIGHT 填充.
    fn try_resize(&mut self) {
        if self.n + 1 < self.capacity {
            return;
        }

        // 指数扩张, 支持 2n 个顶点的存储. 对于 n 个顶点所需空间: n * (n - 1) / 2
        let new_size = (self.n * 2) * (self.n * 2 - 1) / 2;
        // 由于当前使用矩阵压缩存储, 所以只需要在后面追加空间 （扩大三角形的大小）.
        self.array.resize(new_size, INF_WEIGHT);
        self.capacity = self.n << 1;
    }

    /// 返回一个迭代器, 迭代访问给定顶点的所有距离以及权重.
    pub fn distance_from(
        &self,
        vertex: VertexId,
    ) -> impl Iterator<Item=(VertexId, EdgeWeight)> + '_ {
        debug_assert!(self.map.contains_key(&vertex));

        // 解决一个 Rust 中的语法限制: vertex 需要 move 到闭包中, 因此必须先 Copy 一份.
        let vertex_shadow = vertex;
        self.map
            .iter()
            .filter(
                move |(v_j, _j)| **v_j != vertex, /* 过滤掉顶点自身 */
            )
            .filter_map(move |(v_j, _j)| {
                let d = self.get(vertex_shadow, *v_j);
                if d != INF_WEIGHT {
                    Some((*v_j, d))
                } else {
                    None
                }
            })
    }
}

impl Index<(VertexId, VertexId)> for Matrix {
    type Output = EdgeWeight;

    fn index(&self, index: (VertexId, VertexId)) -> &Self::Output {
        let (v_i, v_j) = index;

        debug_assert!(self.map.contains_key(&v_i));
        if v_i == v_j {
            &0
        } else {
            debug_assert!(self.map.contains_key(&v_j));

            let i = *self.map.get(&v_i).unwrap();
            let j = *self.map.get(&v_j).unwrap();
            &self.array[Self::pos(i, j)]
        }
    }
}

impl IndexMut<(VertexId, VertexId)> for Matrix {
    fn index_mut(&mut self, index: (VertexId, VertexId)) -> &mut Self::Output {
        let (v_i, v_j) = index;

        self.try_resize();
        let i = self.get_index_or_insert(v_i);
        self.try_resize();
        let j = self.get_index_or_insert(v_j);

        // 取得这个可变引用后, 需要马上使用. 若 self.array 扩容, 将导致该地址无效
        // (compiler 应该可以保证)
        &mut self.array[Self::pos(i, j)]
    }
}


impl Debug for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let reversed_map =
            self.map
                .iter()
                .fold(vec![0 as VertexId; self.n], |mut vec, (vertex, index)| {
                    vec[*index - 1] = *vertex;
                    vec
                });

        write!(f, "\t")?;
        for e in reversed_map.iter() {
            write!(f, "{e}\t")?;
        }
        writeln!(f)?;

        for i in 0..self.n {
            write!(f, "{}\t", reversed_map[i])?;
            for j in 0..self.n {
                let value = self.get_or_inf(reversed_map[i], reversed_map[j]);
                if value == EdgeWeight::MAX {
                    write!(f, "Inf\t")?;
                } else {
                    write!(f, "{}\t", value)?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct TMatrix {
    matrix: Matrix,
    new_matrix: Option<Matrix>,

    enable: bool,
}

impl TMatrix {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            matrix: Matrix::with_capacity(n),
            new_matrix: None,
            enable: false,
        }
    }
}

impl Deref for TMatrix {
    type Target = Matrix;

    fn deref(&self) -> &Self::Target {
        if self.enable {
            self.new_matrix.as_ref().unwrap()
        } else {
            &self.matrix
        }
    }
}

impl DerefMut for TMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.enable {
            self.new_matrix.as_mut().unwrap()
        } else {
            &mut self.matrix
        }
    }
}

impl Debug for TMatrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.enable {
            write!(
                f,
                "Matrix in transaction mode: \n{:?}",
                self.new_matrix.as_ref().unwrap()
            )
        } else {
            write!(f, "Matrix in normal: \n{:?}", self.matrix)
        }
    }
}

impl Transaction for TMatrix {
    fn begin(&mut self) {
        debug_assert!(!self.enable);
        debug_assert!(self.new_matrix.is_none());

        self.enable = true;
        self.new_matrix = Some(self.matrix.clone());
    }

    fn rollback(&mut self) {
        debug_assert!(self.enable);
        debug_assert!(self.new_matrix.is_some());

        self.new_matrix = None;
        self.enable = false;
    }

    fn commit(&mut self) {
        debug_assert!(self.enable);
        debug_assert!(self.new_matrix.is_some());

        self.matrix = self.new_matrix.take().unwrap();
        self.enable = false;
    }
}

impl Encode for Matrix {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        // 1. 当前顶点数
        self.n.encode(encoder)?;
        // 2. 写入 “顶点-下标” 映射表.
        self.map.encode(encoder)?;
        // 3. 写入矩阵.
        let actual_count = self.n * (self.n - 1) / 2;
        self.array[..actual_count].encode(encoder)
    }
}

impl<Context> Decode<Context> for Matrix {
    fn decode<D: Decoder<Context=Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        // 1. 当前顶点数
        let n = usize::decode(decoder)?;
        // 2. 读取 “顶点-下标” 映射表.
        let map = HashMap::<VertexId, usize>::decode(decoder)?;
        // 3. 读取矩阵.
        let array = Vec::<EdgeWeight>::decode(decoder)?;

        Ok(Matrix { n, map, array, capacity: n })
    }
}

impl Encode for TMatrix {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.matrix.encode(encoder)
    }
}

impl<Context> Decode<Context> for TMatrix {
    fn decode<D: Decoder<Context=Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let matrix = Matrix::decode(decoder)?;
        Ok(TMatrix {
            matrix,
            ..TMatrix::default()
        })
    }
}

impl<Context> BorrowDecode<'_, Context> for TMatrix {
    fn borrow_decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let matrix = Matrix::decode(decoder)?;
        Ok(TMatrix {
            matrix,
            ..TMatrix::default()
        })
    }
}

#[cfg(test)]
mod test {
    use crate::graph::{VertexId, INF_WEIGHT};
    use crate::matrix::{ArrayBuilder, Matrix, TMatrix};
    use crate::transaction::Transaction;

    #[test]
    fn test_array_basic() {
        let mut array = ArrayBuilder::new().from(7).build();

        assert_eq!(array.get(10), None);
        array[10] = 5;
        assert_eq!(array.get(10), Some(5));
    }

    #[test]
    fn test_matrix_basic() {
        let mut matrix = Matrix::with_capacity(5);

        matrix[(1, 2)] = 1;
        matrix[(1, 4)] = 5;
        assert_eq!(matrix[(1, 1)], 0);
        assert_eq!(matrix[(2, 1)], 1);
        assert_eq!(matrix[(4, 1)], 5);
    }

    #[test]
    fn test_matrix_expansion() {
        let mut matrix = Matrix::with_capacity(5);
        matrix[(1, 2)] = 1;
        matrix[(1, 4)] = 5;

        assert_eq!(matrix[(1, 1)], 0);
        assert_eq!(matrix[(2, 1)], 1);
        assert_eq!(matrix[(4, 1)], 5);

        matrix[(8, 9)] = 10;
        assert_eq!(matrix[(1, 9)], INF_WEIGHT);
        assert_eq!(matrix[(1, 8)], INF_WEIGHT);
        assert_eq!(matrix[(8, 9)], 10);
    }

    #[test]
    fn test_matrix_edge_iteration() {
        let mut matrix = Matrix::with_capacity(5);

        matrix[(1, 2)] = 1;
        matrix[(1, 3)] = 2;
        matrix[(1, 4)] = 5;
        matrix[(2, 3)] = 5;
        matrix[(3, 4)] = 2;
        fn get_edges(matrix: &Matrix, v: VertexId) -> Vec<VertexId> {
            let mut result = matrix.distance_from(v).map(|(v, _)| v).collect::<Vec<_>>();

            result.sort();
            result
        }

        assert_eq!(get_edges(&matrix, 3), vec![1, 2, 4]);
        assert_eq!(get_edges(&matrix, 1), vec![2, 3, 4]);
        assert_eq!(get_edges(&matrix, 4), vec![1, 3]);
    }

    #[test]
    fn print_matrix() {
        let mut matrix = Matrix::with_capacity(5);

        matrix[(1, 3)] = 2;
        matrix[(1, 4)] = 5;
        matrix[(7, 3)] = 5;
        matrix[(3, 4)] = 2;
        matrix[(1, 7)] = 1;
        println!("{:?}", matrix);
    }

    #[test]
    fn test_matrix_transaction() {
        let mut matrix = TMatrix::with_capacity(5);

        matrix[(1, 2)] = 1;
        matrix[(1, 4)] = 5;
        assert_eq!(matrix[(1, 1)], 0);
        assert_eq!(matrix[(2, 1)], 1);
        assert_eq!(matrix[(4, 1)], 5);

        matrix.begin();

        matrix[(1, 2)] = 9;
        assert_eq!(matrix[(1, 2)], 9);

        matrix.rollback();
        assert_eq!(matrix[(1, 2)], 1);

        matrix.begin();
        matrix[(1, 3)] = 0;
        assert_eq!(matrix[(1, 3)], 0);
        matrix.commit();
        assert_eq!(matrix[(1, 3)], 0);
    }
}
