# SCORE Artifact

This repository contains the source code needed to build and evaluate SCORE for
the paper submission. It intentionally excludes local build products, large
datasets, generated indexes, raw benchmark logs, paper sources, and third-party
baseline repositories.

## Contents

- `include/` and `src/`: SCORE C++17 implementation and command-line tools.
- `external/RaBitQ-Library/`: vendored header-only subset used by SCORE.
- `external/SuperKMeans/`: vendored header-only subset used by SCORE.
- `tests/`: C++ smoke and regression tests.
- `tools/experiments/`: lightweight scripts for data conversion, SCORE sweeps,
  baseline metric capture, and plotting/parsing benchmark results.

## Build

Required system packages:

- CMake 3.16 or newer
- A C++17 compiler, tested with GCC 13.3.0
- OpenMP and pthreads
- Eigen3 headers, or the vendored Eigen fallback under `external/RaBitQ-Library`
- RapidJSON headers, or network access during CMake configure so CMake can fetch
  RapidJSON v1.1.0

Optional packages:

- BLAS or AOCL-BLIS for faster brute-force ground-truth generation
- tcmalloc for slightly faster serving builds
- liburing only when configuring with `-DSCORE_USE_LIBURING=ON`

Build commands:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For a portable build on machines that do not match the evaluation CPU, disable
native and AVX-512 code generation:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSCORE_ENABLE_NATIVE=OFF \
  -DSCORE_ENABLE_AVX512=OFF
cmake --build build -j
```

Run tests:

```bash
./build/score_test
```

## Data Formats

SCORE uses standard ANN binary vector files:

- `.fvecs`: repeated rows of `[dim:int32][dim float32 values]`.
- `.ivecs`: repeated rows of `[dim:int32][dim int32 values]`, used for ground
  truth.
- `.fbin`: `[n:int32][dim:int32][reserved:int32=0][n * dim float32 values]`.

The optimized streaming builder consumes `.fbin` base vectors. Convert `.fvecs`
base files with:

```bash
python3 tools/experiments/fvecs_to_fbin.py \
  dataset/gist/gist_base.fvecs \
  dataset/gist/gist_base.fbin \
  --overwrite
```

Queries and ground truth are read as `.fvecs` and `.ivecs`.

## Basic Usage

The low-memory paper build path is the streaming reordered builder:

```bash
./build/score_stream_build_reorder \
  dataset/gist/gist_base.fbin \
  results/score_indices/gist_score \
  -M 8 \
  -rabitq_k 200 \
  -ivf_k 1024 \
  -sample_rows 65536 \
  -block_mib 128 \
  -train_iters 10
```

The output index directory must not already exist. Use a fresh path or remove an
old generated index before rebuilding.

Search the built index:

```bash
./build/score_search \
  results/score_indices/gist_score \
  dataset/gist/gist_query.fvecs \
  dataset/gist/gist_groundtruth.ivecs \
  -topk 10 \
  -M 8 \
  -alpha 0.05 \
  -beta 0.01 \
  -threads 1 \
  -max_w 15
```

For larger datasets in the paper, use `-M 16 -ivf_k 2500`. For million-scale
datasets, use `-M 8 -ivf_k 1024`.

Compute brute-force ground truth when needed:

```bash
./build/gt_compute \
  dataset/gist/gist_base.fvecs \
  dataset/gist/gist_query.fvecs \
  dataset/gist/gist_groundtruth.ivecs \
  100
```

Run a SCORE curve sweep:

```bash
python3 tools/experiments/score_curve_sweep.py \
  --dataset gist1m \
  --index-dir results/score_indices/gist_score \
  --build-backend stream-reorder \
  --M 8 \
  --ivf-k 1024 \
  --alphas 0.03,0.04,0.05,0.06 \
  --betas 0.001,0.002,0.005,0.01,0.02 \
  --max-ws 15 \
  --runs 5 \
  --threads 1
```

The script expects dataset files under the paths listed below and writes result
logs under `docs/evaluation/`.

## Updates

SCORE also includes the update path used by the paper. Updates use `.fbin`
vectors and explicit original vector ids. The update files below are workload
inputs, not files distributed with GIST: the id file contains one uint32 original
vector id per update row. Use new ids for inserts, existing ids for replacements,
or `-delete_ids` for deletions.

```bash
./build/score_update_index \
  results/score_indices/gist_score \
  -upsert_fbin dataset/gist/gist_update.fbin \
  -upsert_ids dataset/gist/gist_update_ids.txt \
  -durability buffered \
  -batch_rows 1000 \
  -threads 48 \
  -flush_fraction 0.05 \
  -compaction_fraction 0.20
```

Query an updated index:

```bash
./build/score_update_search \
  results/score_indices/gist_score \
  dataset/gist/gist_query.fvecs \
  dataset/gist/gist_groundtruth.ivecs \
  -topk 10 \
  -M 8 \
  -alpha 0.05 \
  -beta 0.01 \
  -threads 1 \
  -max_w 15
```

## Paper Datasets

The paper uses the following datasets. The repository does not redistribute
datasets; place converted files under the expected paths if using the provided
scripts.

| Dataset | Expected paths | Source |
| --- | --- | --- |
| GIST1M | `dataset/gist/gist_base.fvecs`, `gist_query.fvecs`, `gist_groundtruth.ivecs`, optional `gist_base.fbin` | TEXMEX `ANN_GIST1M`: http://corpus-texmex.irisa.fr/ |
| LAION1M | `dataset/laion1m/laion1m_base.fvecs`, `laion1m_query.fvecs`, `laion1m_groundtruth.ivecs`, optional `laion1m_base.fbin` | SISAP 2024 LAION challenge data: https://sisap-challenges.github.io/2024/datasets/ |
| Cohere1M | `dataset/cohere1m/cohere1m_base.fvecs`, `cohere1m_query.fvecs`, `cohere1m_groundtruth.ivecs`, optional `cohere1m_base.fbin` | Hugging Face `makneeee/cohere_medium_1m`: https://huggingface.co/datasets/makneeee/cohere_medium_1m |
| ImageNet | `dataset/imagenet1m_2048/imagenet1m_2048_base.fvecs`, `imagenet1m_2048_query.fvecs`, `imagenet1m_2048_groundtruth.ivecs`, optional `imagenet1m_2048_base.fbin` | ILSVRC 2012 source images, evaluated as 2048-D features: https://image-net.org/challenges/LSVRC/2012/ |
| Cohere10M | `dataset/cohere10m/cohere10m_base.fvecs`, `cohere10m_query.fvecs`, `cohere10m_groundtruth.ivecs`, optional `cohere10m_base.fbin` | Hugging Face `makneeee/cohere_large_10m`: https://huggingface.co/datasets/makneeee/cohere_large_10m |
| Wikipedia | `dataset/wikipedia_bge_m3/wikipedia_bge_m3_base.fvecs`, `wikipedia_bge_m3_query.fvecs`, `wikipedia_bge_m3_groundtruth.ivecs`, optional `wikipedia_bge_m3_base.fbin` | SISAP 2026 development datasets: https://huggingface.co/datasets/SISAP-Challenges/SISAP2026 |

## Baseline Sources

Baseline source code is not vendored in this artifact. The paper compares SCORE
with the following systems and uses the listed upstream/source references.

| Baseline | Source used for reproduction | Paper/config note |
| --- | --- | --- |
| SuCo | https://github.com/WeiJiuQi/SuCo | Subspace Collision baseline; paper runs use `N_s=8`, `K=50`. |
| TaCo | https://github.com/WeiJiuQi/TaCo | Data-adaptive/query-aware subspace collision baseline; paper runs use `N_s=8`, `s=10`, `K=50`. |
| CRISP | Paper/preprint: https://arxiv.org/abs/2603.05180 | Correlation-resilient subspace collision baseline; paper runs use 16 dimensions per subspace and `K=100`. |
| HNSW | hnswlib: https://github.com/nmslib/hnswlib | Graph baseline; paper runs use `M=48`, `efC=200`. |
| alpha-CNG | https://github.com/Binglbh/Fast-Convergent-Proximity-Graphs-for-Approximate-Nearest-Neighbor-Search | Graph baseline; paper runs use `L=60`, `R=70`, `C=500`. |
| IVF-RaBitQ | https://github.com/gaoj0017/RaBitQ and RaBitQ-Library: https://github.com/VectorDB-NTU/RaBitQ-Library | IVF plus RaBitQ quantization; paper runs use 4096 coarse clusters. |
| DET-LSH | https://github.com/WeiJiuQi/DET-LSH | Theory-guaranteed LSH baseline; paper runs use `K=16`, `L=4`. |

The helper `tools/experiments/baseline_benchmark.py` records baseline build and
search metrics when given a command line for an externally checked-out baseline.

## Third-Party Code Included Here

This artifact vendors only header subsets that are direct SCORE build
dependencies:

- RaBitQ-Library, upstream https://github.com/VectorDB-NTU/RaBitQ-Library,
  local snapshot documented in `external/RaBitQ-Library/VENDOR.md`.
- SuperKMeans, upstream https://github.com/cwida/SuperKMeans, local snapshot
  documented in `external/SuperKMeans/VENDOR.md`.

Their upstream licenses are preserved in the respective `external/*/LICENSE`
files.
