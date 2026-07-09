# RaBitQ-Library vendor snapshot

This directory vendors the header-only subset of RaBitQ-Library that is required by
the SCORE C++ build. It lives under `external/` because it is a core build dependency,
not a benchmark implementation.

- Upstream: <https://github.com/VectorDB-NTU/RaBitQ-Library>
- Local upstream checkout observed when vendored: `c8290476283307ca004ea50a1c92aa328319c272`
- Included files: `LICENSE`, `README.md`, `include/rabitqlib/defines.hpp`,
  `include/rabitqlib/fastscan/{fastscan.hpp,highacc_fastscan.hpp}`,
  `include/rabitqlib/index/{query.hpp,lut.hpp}`, `include/rabitqlib/quantization/**`,
  `include/rabitqlib/utils/{rotator.hpp,fht_avx.hpp,space.hpp,tools.hpp}`, and
  `include/rabitqlib/third/Eigen/**` for the existing Eigen fallback path.
- Excluded files: upstream `.git`, build outputs, binaries, docs, tests, Python code,
  samples, unused index implementations, hnswlib, and unrelated utilities.

SCORE directly uses RaBitQ-Library's `FhtKacRotator` through
`rabitqlib/utils/rotator.hpp`, compact quantization through
`rabitqlib/quantization/rabitq.hpp`, and query helpers through
`rabitqlib/index/query.hpp`. The dependency is header-only for SCORE; no RaBitQ
library binary is linked.

The excluded local checkout had uncommitted changes outside this vendored header
closure. Those changes were not part of SCORE's direct build dependency and should not
be restored into the vendor snapshot without a new include-closure review and smoke
tests.
