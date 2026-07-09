#!/bin/bash
# Build script that strips conda environment variables
# Usage: ./build_clean.sh [make targets...]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# Conda vars to strip
UNSET_VARS="CONDA_PREFIX CONDA_EXE CONDA_TOOLCHAIN_BUILD CONDA_TOOLCHAIN_HOST _CONDA_PYTHON_SYSCONFIGDATA_NAME CMAKE_ARGS CXX CPP CXXFLAGS LDFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS GCC_RANLIB GPROF STRINGS NVCC_PREPEND_FLAGS build_alias GSETTINGS_SCHEMA_DIR GSETTINGS_SCHEMA_DIR_CONDA_BACKUP TESSDATA_PREFIX LIBRARY_PATH LD_LIBRARY_PATH CMAKE_PREFIX_PATH"

# Build the env -u string
ENV_FLAGS=""
for v in $UNSET_VARS; do
    ENV_FLAGS="$ENV_FLAGS -u $v"
done

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

if [ ! -f "$BUILD_DIR/Makefile" ]; then
    echo "=== Running cmake ==="
    mkdir -p "$BUILD_DIR"
    (cd "$BUILD_DIR" && env $ENV_FLAGS cmake .. -DCMAKE_CXX_COMPILER=g++)
fi

echo "=== Building ==="
(cd "$BUILD_DIR" && env $ENV_FLAGS make -j$(nproc) "$@")
