use bap::loader::load_graph;
use bap::partition::VfipPartition;
use clap::Parser;
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

/// 计算函数的执行时间, 返回函数返回值及执行时间（微秒）.
pub fn calculate_cost_time<F, T>(f: F) -> (T, u128)
where
    F: FnOnce() -> T,
{
    let start_time = Instant::now();
    let result = f();
    let elapsed = start_time.elapsed();

    (result, elapsed.as_micros())
}

/// 将不同子图的内点数量按 group_size 一档进行统计.
fn count_by_group(
    iter: impl Iterator<Item = usize>,
    group_size: usize,
) -> Vec<((usize, usize), usize)> {
    use std::collections::btree_map::Entry;

    let mut result = BTreeMap::<usize, usize>::new();
    iter.for_each(|value| {
        let key: usize = value / group_size;

        match result.entry(key) {
            Entry::Occupied(e) => {
                *e.into_mut() += 1;
            }
            Entry::Vacant(e) => {
                e.insert(1);
            }
        };
    });

    result
        .into_iter()
        .map(|(start, count)| ((start * group_size, start * group_size + group_size), count))
        .collect()
}

#[derive(Debug, clap::Parser)]
struct Args {
    #[arg(short, long)]
    /// 图文件路径
    path: String,

    #[arg(short, long, default_value = "5")]
    /// 算法执行次数
    round: u32,
    #[arg(short, long, default_value = "vfip")]
    /// 划分算法
    algorithm: String,
    #[arg(short, long, default_value = "1")]
    /// 初始顶点
    seed: u32,
    #[arg(short, long, default_value = "30")]
    /// 划分时单个子图的最大点的数量, 仅在 vfip 算法下有效.
    theta: u32,
}

#[derive(Debug, serde::Serialize)]
struct SingleResult {
    /// 执行时间（微秒）
    cost: u64,

    /// 子图数量
    subgraph_count: usize,
    /// 子图的平均大小
    avg_size: usize,
    /// 骨架图大小
    skeleton_size: usize,
    /// 分布情况
    distribution: Vec<((usize, usize), usize)>,
}

#[derive(Debug, serde::Serialize)]
struct BenchResult {
    /// 当前划分的算法
    algorithm: String,
    /// 算法执行次数
    round: u32,
    /// 初始顶点
    seed: u32,
    /// 划分时单个子图的最大点的数量.
    theta: u32,
    /// 原始图的顶点数量
    global_size: usize,

    tests: Vec<SingleResult>,
}

fn main() {
    let args = Args::parse();

    let (graph, _) =
        calculate_cost_time(|| load_graph(&args.path).expect("unable to load graph file."));
    let global_size = graph.size();

    let mut tests = vec![];
    for _ in 0..args.round {
        let (subgraphs, partition_time) = calculate_cost_time(|| {
            let vfip = VfipPartition::new(&graph, args.theta as usize);
            vfip.run(args.theta)
        });

        let subgraph_sizes: Vec<_> = subgraphs.iter().map(|sg| sg.graph.size()).collect();
        let avg_size = subgraph_sizes.iter().sum::<usize>() / subgraph_sizes.len();
        let distribution = count_by_group(subgraph_sizes.into_iter(), 10);

        let mut skeleton_vertices = HashSet::new();
        subgraphs
            .iter()
            .for_each(|sg| skeleton_vertices.extend(sg.bound_vertices.iter()));

        let result = SingleResult {
            cost: partition_time as u64,
            subgraph_count: subgraphs.len(),
            avg_size,
            skeleton_size: skeleton_vertices.len(),
            distribution,
        };
        tests.push(result);
    }

    let result = BenchResult {
        global_size,
        algorithm: args.algorithm,
        round: args.round,
        seed: args.seed,
        theta: args.theta,
        tests,
    };

    let json_result = serde_json::to_string(&result).unwrap();
    println!("{}", json_result);
}
