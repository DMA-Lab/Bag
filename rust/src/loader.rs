use std::io::Write;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use bincode::{Decode, Encode};

use crate::graph::{EdgeWeight, Graph, VertexId};
use crate::object::{MovingObjectSet, ObjId};
use crate::partition::Subgraph;

enum Operation {
    Add {
        source: VertexId,
        destination: VertexId,
        weight: EdgeWeight,
    },
    Comment {
        comment: String,
    },
    Problem {
        vertex_count: usize,
        edge_count: usize,
    },
}

fn next_int<T: std::str::FromStr>(s: &str) -> Option<(&str, T)> {
    let (remain, this) = next_str(s)?;
    let result = str::parse::<T>(this).ok()?;

    Some((remain, result))
}

fn next_str(s: &str) -> Option<(&str, &str)> {
    if let Some((this, remain)) = s.split_once(|ch: char| ch.is_whitespace() || ch.is_control()) {
        Some((remain, this))
    } else {
        Some(("", s))
    }
}

fn map_line_to_operation(line: &str) -> Option<Operation> {
    let (command, parameters) = line.split_once(char::is_whitespace)?;

    match command {
        "a" => {
            let s = parameters;

            let (s, source): (_, VertexId) = next_int(s)?;
            let (s, destination): (_, VertexId) = next_int(s)?;
            let (_, weight): (_, EdgeWeight) = next_int(s)?;
            Some(Operation::Add {
                source,
                destination,
                weight,
            })
        }
        "c" => Some(Operation::Comment {
            comment: parameters.to_string(),
        }),
        "p" => {
            let s = parameters;

            let (s, _sp) = next_str(s)?;
            let (s, vertex_count) = next_int(s)?;
            let (_, edge_count) = next_int(s)?;
            Some(Operation::Problem {
                vertex_count,
                edge_count,
            })
        }
        _ => None,
    }
}

fn load_original_graph<P: AsRef<Path>>(path: P) -> std::io::Result<Graph> {
    let file = std::fs::read_to_string(path)?;
    let mut graph = Graph::new();

    for op in file.lines().filter_map(map_line_to_operation) {
        match op {
            Operation::Comment { comment } => println!("comment: {comment}"),
            Operation::Problem {
                vertex_count,
                edge_count,
            } => {
                println!("problem scale: |V| = {vertex_count}, |E| = {edge_count}");
            }
            Operation::Add {
                source,
                destination,
                weight,
            } => {
                for v in [source, destination] {
                    if !graph.contains(v) {
                        graph.insert(v);
                    }
                }
                if let Some(old_weight) = graph.add_edge(source, destination, weight) {
                    if old_weight != weight {
                        println!("warning: weight updated from {old_weight} to {weight}, between v_{source} and v_{destination}.")
                    }
                }
            }
        }
    }

    Ok(graph)
}

fn load_cached_graph<P: AsRef<Path>>(path: P) -> Result<Graph> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to read file: {}", path.display()))?;
    let cached_graph = bincode::decode_from_std_read(&mut file, bincode::config::standard())
        .with_context(|| format!("failed to deserialize graph: {}", path.display()))?;
    Ok(cached_graph)
}

fn export_cached_graph<P: AsRef<Path>>(path: P, graph: &Graph) -> Result<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to create file: {}", path.display()))?;
    bincode::encode_into_std_write(graph, &mut file, bincode::config::standard())
        .with_context(|| format!("failed to serialize graph: {}", path.display()))?;
    Ok(())
}

pub fn path_prefix<P: AsRef<Path>>(path: P) -> String {
    let path = path.as_ref();
    let binding = path.to_string_lossy();
    for suffix in &[".mos.gr", ".gr"] {
        if let Some(prefix) = binding.strip_suffix(suffix) {
            return prefix.to_string();
        }
    }
    binding.to_string()
}

pub fn load_graph<P: AsRef<Path>>(path: P) -> Result<Graph> {
    let path = path.as_ref();

    let new_path = path_prefix(path);
    let cached_path = format!("{new_path}/cached.cgr");
    if Path::new(&cached_path).exists() {
        load_cached_graph(cached_path)
    } else {
        let graph = load_original_graph(path)?;
        if !Path::new(&new_path).exists() && std::fs::create_dir_all(&new_path).is_err() {
            eprintln!("failed to create directory: {}, ignored.", new_path);
        }
        if let Err(e) = export_cached_graph(cached_path, &graph) {
            eprintln!("failed to export cached graph: {e}, ignored.");
        }
        Ok(graph)
    }
}

type MovingObjectRecord = (ObjId, VertexId, VertexId, EdgeWeight);
#[derive(Encode, Decode)]
struct CachedMovingObjects(Vec<MovingObjectRecord>);

pub fn load_moving_objects<P: AsRef<Path>>(path: P) -> Result<MovingObjectSet> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to read file: {}", path.display()))?;
    let cached: CachedMovingObjects = bincode::decode_from_std_read(&mut file, bincode::config::standard())
        .with_context(|| format!("decode failed: {}", path.display()))?;

    let mut set = MovingObjectSet::default();
    for (id, src, dst, weight) in cached.0 {
        set.emplace(id, (src, dst), weight);
    }

    Ok(set)
}

pub fn export_moving_objects<P: AsRef<Path>>(path: P, set: &MovingObjectSet) -> Result<()> {
    let path = path.as_ref();
    let records: Vec<MovingObjectRecord> = set.iter().map(|obj| (obj.unique_id, obj.edge.0, obj.edge.1, obj.offset)).collect();
    let cached = CachedMovingObjects(records);

    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to create file: {}", path.display()))?;
    bincode::encode_into_std_write(&cached, &mut file, bincode::config::standard())
        .with_context(|| format!("failed to serialize moving objects: {}", path.display()))?;
    Ok(())
}

/// Loads a MovingObjectSet from a text file created by `export_moving_objects_text`.
///
/// Since the text format doesn't include a unique ID, this function assigns a new
/// sequential ID to each object upon loading.
pub fn load_moving_objects_text<P: AsRef<Path>>(path: P) -> Result<MovingObjectSet> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open file: {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut set = MovingObjectSet::default();
    let mut current_id_counter: ObjId = 0; // Counter to generate new unique IDs.

    for (line_num, line_result) in reader.lines().enumerate() {
        let line =
            line_result.with_context(|| format!("failed to read line {} from file", line_num + 1))?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        // Expecting format: "a <u_id> <v_id> <offset>"
        if parts.len() != 4 || parts[0] != "a" {
            anyhow::bail!("invalid format on line {}: '{}'", line_num + 1, line);
        }

        let u: VertexId = parts[1].parse().with_context(|| {
            format!("failed to parse vertex u on line {}: '{}'", line_num + 1, line)
        })?;
        let v: VertexId = parts[2].parse().with_context(|| {
            format!("failed to parse vertex v on line {}: '{}'", line_num + 1, line)
        })?;
        let offset: EdgeWeight = parts[3].parse().with_context(|| {
            format!("failed to parse offset on line {}: '{}'", line_num + 1, line)
        })?;

        set.emplace(current_id_counter, (u, v), offset);
        current_id_counter += 1;
    }

    Ok(set)
}

pub fn export_moving_objects_text<P: AsRef<Path>>(path: P, set: &MovingObjectSet) -> Result<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to create file: {}", path.display()))?;

    for obj in set.iter() {
        writeln!(&mut file, "a {} {} {}", obj.unique_id, obj.edge.0, obj.offset)
            .with_context(|| format!("failed to write moving object to file: {}", path.display()))?;
    }

    Ok(())
}

pub fn load_or_generate_moving_objects<P: AsRef<Path>>(
    graph_path: P,
    graph: &Graph,
    count: usize,
) -> MovingObjectSet {
    let path = graph_path.as_ref();

    let prefix = path_prefix(path);
    let mos_path = format!("{prefix}/cached.{count}.mos");
    match load_moving_objects(&mos_path) {
        Ok(set) => set,
        Err(_) => {
            let set = MovingObjectSet::random(graph, count)
                .expect("could not generate moving objects.");
            export_moving_objects(&mos_path, &set).expect("failed to save moving object set.");
            set
        }
    }
}

pub fn load_subgraphs<P: AsRef<Path>>(path: P) -> Result<Vec<Subgraph>> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to read file: {}", path.display()))?;
    let cached: Vec<Subgraph> = bincode::decode_from_std_read(&mut file, bincode::config::standard())
        .with_context(|| format!("decode failed: {}", path.display()))?;
    Ok(cached)
}

pub fn export_subgraphs<P: AsRef<Path>>(path: P, subgraphs: &Vec<Subgraph>) -> Result<()> {
    let path = path.as_ref();

    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to create file: {}", path.display()))?;
    bincode::encode_into_std_write(&subgraphs, &mut file, bincode::config::standard())
        .with_context(|| format!("failed to serialize subgraphs: {}", path.display()))?;
    Ok(())
}

pub fn load_or_partition<P: AsRef<Path>>(
    graph_path: P,
    graph: &Graph,
    start_vertex: VertexId,
    theta: usize,
    partition_func: impl Fn(&Graph, usize, VertexId) -> Vec<Subgraph>,
) -> Vec<Subgraph> {
    let path = graph_path.as_ref();

    let prefix = path_prefix(path);
    let subgraph_path = format!("{prefix}/cached.{theta}.{start_vertex}.subgraphs");
    match load_subgraphs(&subgraph_path) {
        Ok(subgraphs) => subgraphs,
        Err(_) => {
            let subgraphs = partition_func(graph, theta, start_vertex);
            export_subgraphs(&subgraph_path, &subgraphs).expect("failed to save subgraphs.");
            subgraphs
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_graph_loader() {
        use crate::loader::load_graph;

        let graph = load_graph("../dataset/on-paper.gr").expect("failed to load graph file.");
        assert_eq!(graph.size(), 25);
        // 对于基于有向图的无向图而言有 70 条边.
        assert_eq!(graph.edge_count(), 70);

        assert_eq!(graph.get_weight(1, 2), Some(2));
        assert_eq!(graph.get_weight(1, 25), None);

        let graph = load_graph("../dataset/USA-road-d.BAY.gr").expect("failed to load graph file.");
        assert_eq!(graph.size(), 321270);
        assert_eq!(graph.edge_count(), 794830);

        assert_eq!(graph.get_weight(1, 2), Some(1988));
        assert_eq!(graph.get_weight(2, 1), Some(1988));
    }

    #[test]
    fn test_graph_display() {
        use crate::loader::load_graph;

        let graph = load_graph("../dataset/on-paper.gr").expect("failed to load graph file.");
        assert_eq!(graph.size(), 25);
        // 对于基于有向图的无向图而言有 70 条边.
        assert_eq!(graph.edge_count(), 70);

        println!("{graph:?}");
    }
}
