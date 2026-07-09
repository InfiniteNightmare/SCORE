# SuperKMeans vendor snapshot

This repository vendors the header-only subset of SuperKMeans that is required by
the SCORE C++ build.

- Upstream: <https://github.com/cwida/SuperKMeans>
- Local upstream checkout observed when vendored: `bb63fb560283de2759f74ea0f00386c42ece9b64`
- Included files: `LICENSE`, `README.md`, and `include/superkmeans/**`
- Excluded files: upstream `.git`, benchmarks, examples, Python packaging, tests, and scripts

The vendored headers include SCORE-local build-memory and assignment-state patches in
`common.h`, `superkmeans.h`, and `hierarchical_superkmeans.h`. Do not replace this
directory with a clean upstream checkout without reapplying those patches and rerunning
the SCORE build and unit tests.
