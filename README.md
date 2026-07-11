# SCORE

SCORE is an in-memory subspace collision index for high-dimensional approximate
nearest-neighbor (ANN) search. This repository contains the source code and
experiment helpers used by the paper artifact. Datasets, generated indexes,
benchmark outputs, paper sources, and external baseline repositories are not
tracked.

## Quick Start

The steps below show the general SCORE workflow using GIST1M as a concrete
example. For another dataset, replace the file paths and dataset-specific index
settings.

### 1. Build

Requirements:

- CMake 3.16 or newer
- A C++17 compiler (tested with GCC 13.3.0)
- OpenMP and pthreads
- Eigen3 headers, or the fallback vendored with RaBitQ-Library
- RapidJSON headers, or network access during CMake configuration

Build SCORE and run its tests:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/score_test
```

CMake enables native and AVX-512 code generation when supported. For a portable
binary, configure with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSCORE_ENABLE_NATIVE=OFF \
  -DSCORE_ENABLE_AVX512=OFF
cmake --build build -j
```

BLAS or AOCL-BLIS is optional and accelerates brute-force ground-truth
generation. tcmalloc is also optional. liburing is needed only with
`-DSCORE_USE_LIBURING=ON`.

### 2. Prepare a Dataset

Create local data and output directories. In this GIST1M example, place the
downloaded files under `dataset/gist/`. Both directories are ignored by Git.

```bash
mkdir -p dataset/gist results/score_indices
```

The streaming builder consumes `.fbin` base vectors. For GIST1M, convert the
downloaded base file once:

```bash
python3 tools/experiments/fvecs_to_fbin.py \
  dataset/gist/gist_base.fvecs \
  dataset/gist/gist_base.fbin \
  --overwrite
```

Queries remain in `.fvecs` format and ground truth in `.ivecs` format.

### 3. Build an Index

The following quick-start uses the GIST1M settings `M=8` and `ivf_k=1024`, plus
a 128 MiB source block. Other builder options keep their binary defaults.

```bash
./build/score_stream_build_reorder \
  dataset/gist/gist_base.fbin \
  results/score_indices/gist_score \
  -M 8 \
  -ivf_k 1024 \
  -block_mib 128
```

The output index directory must not already exist. The builder creates it and
fails rather than overwriting an existing index.

### 4. Search

This GIST1M point uses one query thread and a larger-than-default candidate
ratio for higher recall:

```bash
./build/score_search \
  results/score_indices/gist_score \
  dataset/gist/gist_query.fvecs \
  dataset/gist/gist_groundtruth.ivecs \
  -M 8 \
  -alpha 0.05 \
  -beta 0.01 \
  -threads 1 \
  -max_w 3
```

`-M` must match the value used to build the index. In this example, `M=8`,
`beta=0.01`, and `threads=1` are GIST experiment settings rather than binary
defaults. The default value of `max_w` is `3`.

## Default Parameters

The primary index builder, `score_stream_build_reorder`, uses these defaults:

| Option | Default |
| --- | --- |
| `-M` | `16` subspaces |
| `-rabitq_k` | `200` top-level clusters |
| `-rabitq_seed` | `42` |
| `-ivf_k` | `2500` cells per subspace |
| `-sample_rows` | `65536` training rows |
| `-train_iters` | `10` |
| `-subspace_workers` | `0`, meaning all subspaces |
| Input block | `65536` rows; `-block_mib` overrides it by size |
| Reorder key | depth `2`, `cluster-first` |
| Reorder I/O | `io_rows=65536`, `bucket_buffer_rows=64`, `rewrite_workers=4` |

`score_search` and `score_update_search` share these query defaults:

| Option | Default |
| --- | --- |
| `-topk` | `10` |
| `-M` | `16` |
| `-alpha` | `0.05` |
| `-beta` | `0.001` |
| `-query_runs` | `1` |
| `-threads` | `0`, meaning the OpenMP default |
| `-max_w` | `3` |
| `-score_prefetch` | `32` (`score_search` only) |

The paper experiments use `rabitq_k=200`, a 512 MiB training sample, 128 MiB
streaming blocks, and 10 K-means iterations. For float32 vectors of dimension
`d`, set `sample_rows=floor(512 * 1024 * 1024 / (4 * d))`. They use
`M=8, ivf_k=1024` for the million-scale datasets and `M=16, ivf_k=2500` for
Cohere10M and Wikipedia.

## Recall-QPS Sweeps

Run the default SCORE query grid on an existing GIST1M index:

```bash
python3 tools/experiments/score_curve_sweep.py \
  --dataset gist1m \
  --index-dir results/score_indices/gist_score \
  --build-backend stream-reorder \
  --M 8 \
  --ivf-k 1024
```

The default grid uses `alpha={0.03,0.04,0.05,0.06}`,
`beta={0.001,0.002,0.005,0.01,0.02}`, `max_w=3`, five runs per point, and one
query thread. If the index directory is absent, the script builds it first; an
existing directory is treated as a fixed index. Results are written under
`docs/evaluation/`. Use `--overwrite-logs` to replace results with the same
label and `--force-rebuild` to rebuild an index under `results/`.

To compute exact ground truth for a base/query pair:

```bash
./build/gt_compute \
  dataset/gist/gist_base.fvecs \
  dataset/gist/gist_query.fvecs \
  dataset/gist/gist_groundtruth.ivecs \
  100
```

## Updates

SCORE accepts inserts, deletions, and upserts against a built index. Insert and
upsert vectors use `.fbin`; each accompanying ID file contains one original
`uint32` vector ID per row in the same order. ID files may be
whitespace/comma-separated text or raw `uint32` binary.

- Insert IDs must not already be live.
- Upserts insert missing IDs and replace existing IDs.
- Deletions require only an ID file.

For example:

```bash
./build/score_update_index \
  results/score_indices/gist_score \
  -upsert_fbin dataset/gist/gist_update.fbin \
  -upsert_ids dataset/gist/gist_update_ids.txt
```

The defaults are buffered WAL durability, one commit for the supplied batch,
the OpenMP default thread count, an L0 flush fraction of `0.05`, and a
compaction fraction of `0.20`. Override them with `-durability`, `-batch_rows`,
`-threads`, `-flush_fraction`, and `-compaction_fraction` as needed.

Use the update-aware query executable after applying updates:

```bash
./build/score_update_search \
  results/score_indices/gist_score \
  dataset/gist/gist_query.fvecs \
  dataset/gist/gist_groundtruth.ivecs \
  -M 8 \
  -alpha 0.05 \
  -beta 0.01 \
  -threads 1 \
  -max_w 3
```

Ground truth must describe the live vectors represented by the updated index.

## Paper Datasets

The repository does not redistribute datasets. The experiment scripts expect
the following paths:

| Dataset | Script key and expected files | Source |
| --- | --- | --- |
| GIST1M | `gist1m`: `dataset/gist/gist_{base,query}.fvecs`, `gist_groundtruth.ivecs` | [TEXMEX ANN GIST1M](http://corpus-texmex.irisa.fr/) |
| LAION1M | `laion1m`: `dataset/laion1m/laion1m_{base,query}.fvecs`, `laion1m_groundtruth.ivecs` | [SISAP 2024 LAION challenge](https://sisap-challenges.github.io/2024/datasets/) |
| Cohere1M | `cohere1m`: `dataset/cohere1m/cohere1m_{base,query}.fvecs`, `cohere1m_groundtruth.ivecs` | [makneeee/cohere_medium_1m](https://huggingface.co/datasets/makneeee/cohere_medium_1m) |
| ImageNet1M | `imagenet1m_2048`: `dataset/imagenet1m_2048/imagenet1m_2048_{base,query}.fvecs`, `imagenet1m_2048_groundtruth.ivecs` | [ILSVRC 2012 images](https://image-net.org/challenges/LSVRC/2012/), evaluated as 2048-D features |
| Cohere10M | `cohere10m`: `dataset/cohere10m/cohere10m_{base,query}.fvecs`, `cohere10m_groundtruth.ivecs` | [makneeee/cohere_large_10m](https://huggingface.co/datasets/makneeee/cohere_large_10m) |
| Wikipedia | `wikipedia`: `dataset/wikipedia_bge_m3/wikipedia_bge_m3_{base,query}.fvecs`, `wikipedia_bge_m3_groundtruth.ivecs` | [SISAP 2026 development datasets](https://huggingface.co/datasets/SISAP-Challenges/SISAP2026) |

For streaming builds, convert each base file to the same path with an `.fbin`
extension.

## Baseline Sources

Baseline repositories are not vendored. The paper uses the following upstream
sources and configurations:

| Baseline | Upstream source | Paper configuration |
| --- | --- | --- |
| SuCo | [WeiJiuQi/SuCo](https://github.com/WeiJiuQi/SuCo) | `N_s=8`, `K=50` |
| TaCo | [WeiJiuQi/TaCo](https://github.com/WeiJiuQi/TaCo) | `N_s=8`, `s=10`, `K=50` |
| CRISP | [Preprint](https://arxiv.org/abs/2603.05180) | 16 dimensions per subspace, `K=100` |
| HNSW | [nmslib/hnswlib](https://github.com/nmslib/hnswlib) | `M=48`, `efC=200` |
| alpha-CNG | [Binglbh/Fast-Convergent-Proximity-Graphs](https://github.com/Binglbh/Fast-Convergent-Proximity-Graphs-for-Approximate-Nearest-Neighbor-Search) | `L=60`, `R=70`, `C=500` |
| IVF-RaBitQ | [gaoj0017/RaBitQ](https://github.com/gaoj0017/RaBitQ), [VectorDB-NTU/RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library) | 4096 coarse clusters |
| DET-LSH | [WeiJiuQi/DET-LSH](https://github.com/WeiJiuQi/DET-LSH) | `K=16`, `L=4` |

## Repository Layout

- `include/`: SCORE index construction, query, rotation, and update code.
- `src/`: command-line executables.
- `tests/`: C++ smoke and regression tests.
- `tools/experiments/`: SCORE data conversion and Recall-QPS sweep helpers.
- `external/RaBitQ-Library/`: vendored header-only RaBitQ subset.
- `external/SuperKMeans/`: vendored header-only SuperKMeans subset.

The vendored dependencies retain their upstream licenses. Their snapshots are
documented in
[`external/RaBitQ-Library/VENDOR.md`](external/RaBitQ-Library/VENDOR.md) and
[`external/SuperKMeans/VENDOR.md`](external/SuperKMeans/VENDOR.md).
