use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;

use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use bincode::de::{BorrowDecoder, Decoder};
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{BorrowDecode, Decode, Encode};

#[derive(Default, Clone)]
pub struct TMap<K, V>
where
    K: Ord + Copy + Hash,
    V: Clone,
{
    /// 存储主要数据
    main: HashMap<K, V>,
    /// 上一次 `begin()` 后, 到下一次 `commit()` 前的增量数据. 记住: 该集合是相对于 `main` 的操作而言的.
    increment: HashMap<K, V>,
    /// 上一次 `begin()` 后, 到下一次 `commit()` 前删除的数据. 记住: 该集合是相对于 `main` 的操作而言的.
    /// 其实根据当前需求, 不太用得到删除操作, 但为了功能的完备性实现了该功能.
    decrement: HashSet<K>,

    /// 是否使用 increment/decrement 集合 （事务开关）
    enabled: bool,
    /// 字典中的元素数量. 因为要保证 `len()` 方法的时间复杂度为 `O(1)`, 所以在此处存储大小.
    ///
    /// 保证 `len` 的值为 `main.len() + increment.len() - duplicates.len() - decrement.len()`,
    /// 其中 `duplicates` 为 `main` 和 `increment` 重复项的数量.
    /// 另外注意 `decrement` 和 `increment` 不存在交集.
    len: usize,
}

pub trait Transaction {
    /// 开始一个事务, 这将改变 insert 和 remove 的行为.
    fn begin(&mut self);

    /// 结束并回滚当前事务中的修改
    fn rollback(&mut self);

    /// 提交当前修改
    fn commit(&mut self);
}

impl<K: Ord + Copy + Hash, V: Clone> Transaction for TMap<K, V> {
    fn begin(&mut self) {
        debug_assert!(!self.enabled);
        debug_assert!(self.increment.is_empty());
        debug_assert!(self.decrement.is_empty());
        self.enabled = true;
    }

    fn rollback(&mut self) {
        debug_assert!(self.enabled);
        // 注意注意 increment 和 decrement 没有交集, 但 main 和 increment 或 main 和 decrement 可能有.
        self.len += self.decrement.len();
        self.decrement.clear();

        for (k, _) in self.increment.iter() {
            // 对 main 和 increment 非重叠部分计算长度.
            if !self.main.contains_key(k) {
                self.len -= 1;
            }
        }
        self.increment.clear();
        self.enabled = false;
    }

    fn commit(&mut self) {
        debug_assert!(self.enabled);
        // 处理当前事务中新增的条目
        for (k, v) in self.increment.iter() {
            self.main.insert(*k, v.clone());
        }
        self.increment.clear();

        // 处理当前事务中删除的条目
        for e in self.decrement.iter() {
            self.main.remove(e);
        }
        self.decrement.clear();
        // 结束事务.
        self.enabled = false;
    }
}

impl<K: Ord + Copy + Hash, V: Clone> TMap<K, V> {
    pub fn new() -> Self {
        Self {
            main: HashMap::new(),
            increment: HashMap::new(),
            decrement: HashSet::new(),
            enabled: false,
            len: 0,
        }
    }

    /// 插入新记录.
    ///
    /// 若此时处于事务操作模式,
    /// 1. 记录存在于 main, 则在 increment 集合中覆盖它;
    /// 2. 记录存在于 increment, 直接替换掉;
    /// 3. 记录存在于 decrement, 移除旧的 remove 操作并添加记录.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.enabled {
            // 删除后再添加 K-V: 消除此条 decrement 操作.
            if self.decrement.contains(&key) {
                self.decrement.remove(&key);
                // 由于 decrement 是相对于 main 而言的减量操作, 此时 main 一定有对应的 key.
                // 从 decrement 删除项目后, 长度 +1.
                self.len += 1;
            }

            match self.increment.insert(key, value) {
                Some(old) => Some(old),
                None => {
                    if !self.main.contains_key(&key) {
                        self.len += 1;
                    }
                    None
                }
            }
        } else {
            match self.main.insert(key, value) {
                Some(v) => Some(v),
                None => {
                    self.len += 1;
                    None
                }
            }
        }
    }

    pub fn contains(&self, key: &K) -> bool {
        // 要求 decrement 不包含对应元素 (即, 元素没有被标记为删除)
        !self.decrement.contains(key)
            && (self.increment.contains_key(key) || self.main.contains_key(key))
    }

    /// 删除记录
    ///
    /// 如果 `main` 和 `increment` 都包含 `key` 项目, 那么自然地返回 `increment` 中的 value.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if !self.enabled {
            // 如果未开启事务, 从主集合中删除
            return match self.main.remove(key) {
                Some(value) => {
                    self.len -= 1;
                    Some(value)
                }
                None => None,
            };
        }

        /*
             main    increment   操作
             -----------------------------
             有      有           len -= 1
             有      无           len -= 1
             无      有           len -= 1
             无      无           nothing
        */
        // 如果目标记录是新增的, 尚未提交, 直接从 increment 删除
        let deleted_value_from_increment = self.increment.remove(key);

        // 无论 increment 是否包含要删除的项, 都要检查 main 中是否包含旧的项
        let deleted_value_from_main = if self.main.contains_key(key) {
            // 如果目标记录在 main 中, 标记删除
            self.decrement.insert(*key);
            self.main.get(key)
        } else {
            None
        };

        match deleted_value_from_increment.or(deleted_value_from_main.cloned()) {
            Some(v) => {
                self.len -= 1;
                Some(v)
            }
            None => None,
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item=(&K, &V)> + '_> {
        if !self.enabled {
            return Box::new(self.main.iter());
        }

        // 由于 map 存储的是键值对, 所以当 main 和 increment 都包含某一 key 但 value 不同时，应使用后添加的版本.
        let iter = self
            .main
            .iter()
            .filter(|(k, _)| {
                !self.enabled || (!self.decrement.contains(k) && !self.increment.contains_key(k))
            })
            .chain(self.increment.iter());
        Box::new(iter)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        if self.decrement.contains(key) {
            return None;
        }

        self.increment.get(key).or_else(|| self.main.get(key))
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if self.decrement.contains(key) {
            return None;
        }

        self.increment
            .get_mut(key)
            .or_else(|| self.main.get_mut(key))
    }
}

impl<K, V> Encode for TMap<K, V>
where
    K: Encode + Ord + Copy + Hash,
    V: Encode + Clone,
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        assert_eq!(self.enabled, false);
        self.main.encode(encoder)
    }
}

impl<K, V, Context> Decode<Context> for TMap<K, V>
where
    K: Ord + Copy + Hash + Decode<Context>,
    V: Clone + Decode<Context>,
{
    fn decode<D: Decoder<Context=Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let main = HashMap::<K, V>::decode(decoder)?;
        let len = main.len();
        Ok(Self {
            main,
            len,
            ..TMap::new()
        })
    }
}

impl<T: Debug + Ord + Copy + Hash> Debug for TSet<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (index, element) in self.iter().enumerate() {
            if index != 0 {
                write!(f, ", {element:?}")?;
            } else {
                write!(f, "{element:?}")?;
            }
        }
        write!(f, "}}")
    }
}

#[derive(Default, Clone)]
pub struct TSet<T: Ord + Copy + Hash> {
    inner: TMap<T, ()>,
}

impl<T: Ord + Copy + Hash> TSet<T> {
    pub fn new() -> Self {
        Self { inner: TMap::new() }
    }

    /// 插入新记录.
    ///
    /// 若此时处于事务操作模式,
    /// 1. 记录存在于 main, 则在 increment 集合中覆盖它;
    /// 2. 记录存在于 increment, 直接替换掉;
    /// 3. 记录存在于 decrement, 移除旧的 remove 操作并添加记录.
    pub fn insert(&mut self, key: T) -> bool {
        self.inner.insert(key, ()).is_none()
    }

    pub fn contains(&self, key: &T) -> bool {
        self.inner.contains(key)
    }

    /// 删除记录
    pub fn remove(&mut self, key: &T) -> bool {
        self.inner.remove(key).is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item=T> + '_ {
        self.inner.iter().map(|(k, _)| *k)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<T: Ord + Copy + Hash> Transaction for TSet<T> {
    fn begin(&mut self) {
        self.inner.begin()
    }

    fn rollback(&mut self) {
        self.inner.rollback()
    }

    fn commit(&mut self) {
        self.inner.commit()
    }
}

impl<T> Encode for TSet<T>
where
    T: Ord + Copy + Hash + Encode,
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.inner.encode(encoder)
    }
}

impl<T, Context> Decode<Context> for TSet<T>
where
    T: Ord + Copy + Hash + Decode<Context>,
{
    fn decode<D: Decoder<Context=Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let inner = TMap::<T, ()>::decode(decoder)?;
        Ok(Self { inner })
    }
}

impl<'de, T, Context> BorrowDecode<'de, Context> for TSet<T>
where
    T: Ord + Copy + Hash + Decode<Context>,
{
    fn borrow_decode<D: BorrowDecoder<'de, Context=Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Self::decode(decoder)
    }
}

#[derive(Default, Clone, Encode, Decode)]
pub struct NestedMap<K1, K2, V>
where
    K1: Ord + Copy + Hash,
    K2: Ord + Copy + Hash,
{
    inner: HashMap<K1, HashMap<K2, V>>,
}

impl<K1, K2, V> NestedMap<K1, K2, V>
where
    K1: Ord + Copy + Hash,
    K2: Ord + Copy + Hash,
{
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key1: K1, key2: K2, value: V) -> Option<V> {
        if let Some(map2) = self.inner.get_mut(&key1) {
            map2.insert(key2, value)
        } else {
            let mut map2 = HashMap::<K2, V>::new();
            map2.insert(key2, value);

            self.inner.insert(key1, map2);
            None
        }
    }

    pub fn remove(&mut self, key1: &K1, key2: &K2) -> Option<V> {
        if let Some(map2) = self.inner.get_mut(key1) {
            let value = map2.remove(key2);
            if map2.is_empty() {
                self.inner.remove(key1);
            }
            value
        } else {
            None
        }
    }

    pub fn get(&self, key1: &K1, key2: &K2) -> Option<&V> {
        self.inner
            .get(key1)
            .and_then(|map2| map2.get(key2))
            .or(None)
    }

    pub fn contains_key(&self, key1: &K1, key2: &K2) -> bool {
        self.get(key1, key2).is_some()
    }

    fn iter(&self) -> impl Iterator<Item=(&K1, &K2, &V)> + '_ {
        self.inner
            .iter()
            .flat_map(|(key1, map2)| map2.iter().map(move |(key2, value)| (key1, key2, value)))
    }

    fn iter_map2(&self, key1: &K1) -> Box<dyn Iterator<Item=(&K2, &V)> + '_> {
        if let Some(map2) = self.inner.get(key1) {
            Box::new(map2.iter())
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn merge_from(&mut self, other: &mut Self) {
        for (new_key1, new_map2) in other.inner.drain() {
            if let Some(old_map2) = self.inner.get_mut(&new_key1) {
                old_map2.extend(new_map2);
            } else {
                self.inner.insert(new_key1, new_map2);
            }
        }
    }

    fn sub_from<_V>(&mut self, other: &mut NestedMap<K1, K2, _V>) {
        for (there_key1, there_map2) in other.inner.drain() {
            if let Some(here_map2) = self.inner.get_mut(&there_key1) {
                // Delete items from here_map2 referring to there_map2
                for k in there_map2.keys() {
                    here_map2.remove(k);
                }
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct TNestedMap<K1, K2, V>
where
    K1: Ord + Copy + Hash,
    K2: Ord + Copy + Hash,
{
    main: NestedMap<K1, K2, V>,
    increment: NestedMap<K1, K2, V>,
    decrement: NestedMap<K1, K2, ()>,

    enabled: bool,
}

impl<K1, K2, V> TNestedMap<K1, K2, V>
where
    K1: Ord + Copy + Hash,
    K2: Ord + Copy + Hash,
{
    pub fn new() -> Self {
        Self {
            main: NestedMap::new(),
            increment: NestedMap::new(),
            decrement: NestedMap::new(),
            enabled: false,
        }
    }

    pub fn insert(&mut self, key1: K1, key2: K2, value: V) -> Option<V> {
        let map = if self.enabled {
            &mut self.increment
        } else {
            &mut self.main
        };

        if self.enabled && self.decrement.contains_key(&key1, &key2) {
            self.decrement.remove(&key1, &key2);
        }
        map.insert(key1, key2, value)
    }

    pub fn remove(&mut self, key1: &K1, key2: &K2) {
        if self.enabled {
            if self.increment.contains_key(key1, key2) {
                self.increment.remove(key1, key2);
            }

            if self.main.contains_key(key1, key2) {
                self.decrement.insert(*key1, *key2, ());
            }
        } else {
            self.main.remove(key1, key2);
        }
    }

    pub fn contains_key(&self, key1: &K1, key2: &K2) -> bool {
        (self.enabled
            && !self.decrement.contains_key(key1, key2)
            && self.increment.contains_key(key1, key2))
            || self.main.contains_key(key1, key2)
    }

    pub fn get<'a>(&'a self, key1: &'a K1, key2: &'a K2) -> Option<&'a V> {
        if self.enabled {
            if self.decrement.contains_key(key1, key2) {
                return None;
            } else {
                self.increment.get(key1, key2)
            }
        } else {
            None
        }
            .or_else(|| self.main.get(key1, key2))
    }

    pub fn get_tier2_map_iter<'a>(&'a self, key1: &'a K1) -> impl Iterator<Item=(&K2, &V)> {
        self.main
            .iter_map2(key1)
            .filter(|(key2, _)| !self.increment.contains_key(key1, key2))
            .chain(self.increment.iter_map2(key1))
            .filter(|(key2, _)| !self.decrement.contains_key(key1, key2))
    }

    pub fn iter(&self) -> impl Iterator<Item=(&K1, &K2, &V)> + '_ {
        self.main
            .iter()
            .filter(|(key1, key2, _)| !self.increment.contains_key(key1, key2))
            .chain(self.increment.iter())
            .filter(|(key1, key2, _)| !self.decrement.contains_key(key1, key2))
    }

    // TODO: len() now only supported when not in transaction mode.
    pub fn len(&self) -> usize {
        assert_eq!(self.enabled, false);
        self.main.inner.len()
    }
}

impl<K1, K2, V> Transaction for TNestedMap<K1, K2, V>
where
    K1: Ord + Copy + Hash,
    K2: Ord + Copy + Hash,
{
    fn begin(&mut self) {
        debug_assert!(!self.enabled);
        self.enabled = true;
    }

    fn rollback(&mut self) {
        debug_assert!(self.enabled);
        self.enabled = false;

        self.increment.inner.clear();
        self.decrement.inner.clear();
    }

    fn commit(&mut self) {
        debug_assert!(self.enabled);
        self.enabled = false;

        self.main.merge_from(&mut self.increment);
        /* Now self.increment is empty. */

        self.main.sub_from(&mut self.decrement);
        /* Now self.decrement is empty. */
    }
}

impl<K1, K2, V> Encode for TNestedMap<K1, K2, V>
where
    K1: Ord + Copy + Hash + Encode,
    K2: Ord + Copy + Hash + Encode,
    V: Encode,
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        assert_eq!(self.enabled, false);
        self.main.encode(encoder)
    }
}


impl<K1, K2, V, Context> Decode<Context> for TNestedMap<K1, K2, V>
where
    K1: Ord + Copy + Hash + Decode<Context>,
    K2: Ord + Copy + Hash + Decode<Context>,
    V: Decode<Context>,
{
    fn decode<D: Decoder<Context=Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let main = NestedMap::<K1, K2, V>::decode(decoder)?;
        Ok(Self {
            main,
            increment: NestedMap::new(),
            decrement: NestedMap::new(),
            enabled: false,
        })
    }
}

#[derive(Default)]
pub struct TQueue<T> {
    queue: VecDeque<T>,
    new: VecDeque<T>,

    enabled: bool,
}

impl<T> TQueue<T> {
    pub fn push(&mut self, value: T) {
        if self.enabled {
            self.new.push_back(value);
        } else {
            self.queue.push_back(value);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.enabled {
            self.new.pop_front()
        } else {
            self.queue.pop_front()
        }
    }
}

impl<T> Transaction for TQueue<T> {
    fn begin(&mut self) {
        debug_assert!(!self.enabled);
        self.new.clear();
        self.new.append(&mut self.queue);
        self.enabled = true;
    }

    fn rollback(&mut self) {
        debug_assert!(self.enabled);
        self.new.clear();
        self.enabled = false;
    }

    fn commit(&mut self) {
        debug_assert!(self.enabled);
        self.queue.clear();
        self.queue.append(&mut self.new);
        self.enabled = false;
    }
}

#[cfg(test)]
mod test {
    use crate::transaction::{TMap, TNestedMap, TSet, Transaction};

    #[test]
    fn test_transactional_set() {
        let mut set = TSet::new();

        /* 测试普通情况下的插入 */
        set.insert(1);
        set.insert(2); // Now, there are 1, 2 in set.
        debug_assert!(set.contains(&1));
        debug_assert!(set.contains(&2));

        /* 测试事务模式下的插入 */
        set.begin();
        set.insert(3);
        set.insert(4);
        set.commit(); // Now, there are 1, 2, 3, 4 in set.
        debug_assert!(set.contains(&3));
        debug_assert!(set.contains(&4));

        /* 测试回滚插入 */
        set.begin();
        set.insert(5);
        debug_assert!(set.contains(&5));
        set.rollback();

        // 5 is rolled back.
        debug_assert!(!set.contains(&5));
        // 3, 4 are not rolled back.
        debug_assert!(set.contains(&3));
        debug_assert!(set.contains(&4));

        /* 测试回滚删除 */
        debug_assert!(set.contains(&1));
        set.begin();
        set.remove(&1);
        debug_assert!(!set.contains(&1));
        set.rollback();
        debug_assert!(set.contains(&1));
    }

    #[test]
    fn test_transactional_map() {
        let mut map = TMap::new();
        assert_eq!(map.len(), 0);

        map.insert(1, 1);
        assert_eq!(map.len(), 1);
        map.insert(1, 2); // Update value to 2.
        assert_eq!(map.len(), 1);

        /* As for now, map = {1: 2} */
        map.begin();
        map.insert(1, 3);
        assert_eq!(map.len(), 1);
        map.remove(&1);
        assert_eq!(map.len(), 0);

        map.rollback();
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_transactional_nested_map() {
        let mut map = TNestedMap::<u32, u32, usize>::new();

        map.insert(1, 2, 0);
        map.insert(3, 4, 9);
        map.insert(7, 5, 1);

        assert_eq!(map.get(&1, &2), Some(&0));
        assert_eq!(map.get(&3, &4), Some(&9));
        assert_eq!(map.get(&7, &5), Some(&1));
        assert_eq!(map.get(&1, &5), None);
        assert_eq!(map.get(&0, &0), None);

        map.begin();
        map.insert(1, 5, 3);
        map.insert(0, 0, 0);
        assert_eq!(map.get(&1, &5), Some(&3));
        assert_eq!(map.get(&0, &0), Some(&0));

        map.rollback();
        assert_eq!(map.get(&1, &5), None);
        assert_eq!(map.get(&0, &0), None);

        map.begin();
        map.insert(1, 5, 3);
        map.insert(0, 0, 0);
        map.commit();
        assert_eq!(map.get(&1, &5), Some(&3));
        assert_eq!(map.get(&0, &0), Some(&0));

        map.remove(&0, &0);
        assert_eq!(map.get(&0, &0), None);

        map.begin();
        map.remove(&1, &5);
        assert_eq!(map.get(&1, &5), None);
        map.rollback();
        assert_eq!(map.get(&1, &5), Some(&3));

        map.begin();
        map.remove(&1, &5);
        assert_eq!(map.get(&1, &5), None);
        map.commit();
        assert_eq!(map.get(&1, &5), None);
    }
}
