# BAG

Official source code for **BAG**, a partition-based framework for exact spatial query processing on road networks with moving objects.

This repository contains implementation code only. Datasets and generated artifacts are intentionally excluded.

## Repository Layout

- `src/`: C++ implementation
- `rust/`: Rust implementation
- `CMakeLists.txt`: build entry for the C++ implementation

## Build

### C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

This produces the `bag_cpp` executable.

### Rust

```bash
cd rust
cargo build --release
```

## C++ Entry Points

The C++ executable exposes the following main commands:

- `batch`
- `knn-batch`
- `knn-build-diagnostics`
- `hierarchy-probe`
- `diagnostics`
- `maint-bench`
- `memory-report`

Run the executable without arguments to print the full command-line interface.

## Notes

- The repository is intended for code release and reproducible builds.
- External road-network inputs should be supplied by the user at runtime.
