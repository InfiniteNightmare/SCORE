#!/usr/bin/env bash
# SCORE local regression gate.
#
# There is no remote CI on this repo, so this script is the gate that protects
# the 117/117 score_test state from drift. Run it before merges / pushes / after
# any change to include/ or src/. Exits non-zero on any build or test failure.
#
#   ./tools/ci.sh              # build + run score_test
#   ./tools/ci.sh --tsan       # also run the TSan build + score_test (concurrency)
#
# The --tsan path uses docs/tsan_validation.md + tools/tsan_suppressions.txt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Match build_clean.sh: strip conda env vars that poison the toolchain.
UNSET_VARS="CONDA_PREFIX CONDA_EXE CONDA_TOOLCHAIN_BUILD CONDA_TOOLCHAIN_HOST \
_CONDA_PYTHON_SYSCONFIGDATA_NAME CMAKE_ARGS CXX CPP CXXFLAGS LDFLAGS DEBUG_CFLAGS \
DEBUG_CXXFLAGS GCC_RANLIB GPROF STRINGS NVCC_PREPEND_FLAGS build_alias \
GSETTINGS_SCHEMA_DIR GSETTINGS_SCHEMA_DIR_CONDA_BACKUP TESSDATA_PREFIX LIBRARY_PATH \
LD_LIBRARY_PATH CMAKE_PREFIX_PATH"
ENV_FLAGS=""
for v in $UNSET_VARS; do ENV_FLAGS="$ENV_FLAGS -u $v"; done
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

BUILD="$ROOT/build"
if [ ! -f "$BUILD/Makefile" ]; then
    echo "== ci: configuring build/ =="
    mkdir -p "$BUILD"
    (cd "$BUILD" && env $ENV_FLAGS cmake .. -DCMAKE_CXX_COMPILER=g++)
fi

echo "== ci: building score_test =="
(cd "$BUILD" && env $ENV_FLAGS make -j"$(nproc)" score_test)

echo "== ci: running score_test =="
"$BUILD/score_test"

if [ "${1:-}" = "--tsan" ]; then
    TSAN_BUILD="$ROOT/build_tsan"
    if [ ! -f "$TSAN_BUILD/Makefile" ]; then
        echo "== ci: configuring build_tsan/ =="
        cmake -S . -B "$TSAN_BUILD" -DCMAKE_CXX_COMPILER=g++ \
            -DSCORE_SANITIZER_FRIENDLY=ON \
            -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1 -fno-omit-frame-pointer" \
            -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
    fi
    echo "== ci: building TSan score_test =="
    cmake --build "$TSAN_BUILD" -j"$(nproc)" --target score_test
    echo "== ci: running TSan score_test (ASLR off, OMP noise suppressed) =="
    setarch "$(uname -m)" -R env \
        TSAN_OPTIONS="suppressions=$ROOT/tools/tsan_suppressions.txt halt_on_error=0" \
        "$TSAN_BUILD/score_test"
fi

echo "== ci: PASS =="
