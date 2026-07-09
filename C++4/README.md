# BAG C++4 — final improved build

Drop-in replacement for the paper repo's `C++2/` variant. This is the authoritative
improved source (supersedes the earlier `C++3/` snapshot). It contains, over the pristine
`Bag-main.zip`:

- **5 correctness fixes** — the default path is now **exact** (verified per query vs
  whole-graph Dijkstra); the pristine baseline is not. See `../IMPROVEMENTS.md`.
- **Query-engine speedups** — CSR skeleton/clique rows, deferred subgraph-grouped
  refinement, per-subgraph kNN-order SoA, monotone radix heap, b0 tight stopping radius
  (kNN), border-cost pc-subgraph evaluation (range).
- **Border-minimized partitions by default** (`--border-min` defaults to **true**): demotes
  conservatively-marked border vertices back to internal when they have no external edge and
  the BR-property still holds — ~26% fewer borders, ~15% faster queries, no extra build cost,
  still exact. Pass `--border-min false` to reproduce the paper's partition. The partition
  algorithm itself (`partition.cpp`) is unchanged — this only flips the default.

Net on NY θ=30 vs the (wrong) baseline: **kNN ~2.6–2.8×, range ~1.8–2.0×**, exact.

## Which files changed vs pristine C++2

Only these five, all in `src/`:

- `index.cpp`, `index.h`, `object.cpp`, `object.h` — the query engine + data layout
- `main.cpp` — CLI conveniences + the `--border-min` default flip

Everything else in `src/` is byte-identical to pristine and included so the tree builds
standalone. `../C++2_improvements.patch` is the full unified diff. If you only want to patch
your own tree, the four `index`/`object` files are the interdependent core (see
`../minimal-4-files/`); add `main.cpp` (or pass `--border-min true`) for the partition win.

## Build

```bash
python3 -m ziglang c++ -std=c++20 -O2 -DNDEBUG -DBAG_ENABLE_KNN_FINE_TIMING=0 \
  -Isrc src/*.cpp -o bag_cpp          # or use CMakeLists.txt with a normal clang/g++
```

## Run

```bash
./bag_cpp partition   --path NY_dimacs.gr --theta 30 --partition-cache-mode auto
./bag_cpp batch       --path NY_dimacs.gr --theta 30 --partition-cache-mode read \
  --query-count 500 --objects 200000 --range-radius 50000 --knn-k 200
# verify the production kNN/range paths are exact:
./bag_cpp query-sweep --path NY_dimacs.gr --theta 30 --partition-cache-mode read \
  --query-count 30 --objects 200000 --radii 12500,25000,50000,100000 \
  --ks 50,200,400,800 --knn-pc-dijkstra-mode skeleton --verify true
```

`--knn-sound-termination` defaults to true (exact); `--border-min` defaults to true.
