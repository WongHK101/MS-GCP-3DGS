#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT=${SRC_ROOT:-/root/autodl-tmp/src/ms-gcp-3dgs}
OPT_ROOT=${OPT_ROOT:-/root/autodl-tmp/opt/ms-gcp-3dgs}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COLMAP_TAG=${COLMAP_TAG:-4.0.4}
CERES_COMMIT=${CERES_COMMIT:-8a566fcc156322160b96f8ca5f0ff755241c2d33}
CUDSS_VERSION=${CUDSS_VERSION:-0.8.0.10}
CUDSS_ARCHIVE="libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz"
CUDSS_URL="https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/${CUDSS_ARCHIVE}"

COLMAP_SRC="$SRC_ROOT/colmap-${COLMAP_TAG}"
CERES_SRC="$SRC_ROOT/ceres-gpu-ba"
CUDSS_PREFIX="$OPT_ROOT/cudss-${CUDSS_VERSION}"
CERES_PREFIX="$OPT_ROOT/ceres-gpu-ba"
COLMAP_PREFIX="$OPT_ROOT/colmap-${COLMAP_TAG}-gpu-ba"
BUILD_ROOT="$SRC_ROOT/build"

mkdir -p "$SRC_ROOT" "$OPT_ROOT" "$BUILD_ROOT"

if [[ ! -d "$COLMAP_SRC/.git" ]]; then
  git clone --branch "$COLMAP_TAG" --depth 1 --recursive \
    --shallow-submodules https://github.com/colmap/colmap.git "$COLMAP_SRC"
fi

if [[ ! -d "$CERES_SRC/.git" ]]; then
  git clone --depth 1 --recursive --shallow-submodules \
    https://github.com/ceres-solver/ceres-solver.git "$CERES_SRC"
fi
git -C "$CERES_SRC" fetch --depth 1 origin "$CERES_COMMIT"
git -C "$CERES_SRC" checkout --detach "$CERES_COMMIT"
git -C "$CERES_SRC" submodule update --init --recursive --depth 1

# Ceres tries to download a Gerrit commit hook during CMake configure. That
# development-only network access can stall on restricted server networks.
mkdir -p "$CERES_SRC/.git/hooks"
touch "$CERES_SRC/.git/hooks/commit-msg"
chmod +x "$CERES_SRC/.git/hooks/commit-msg"

CERES_ARCH_PATCH="$REPO_ROOT/patches/ceres_respect_explicit_cuda_architectures.patch"
if git -C "$CERES_SRC" apply --check "$CERES_ARCH_PATCH" 2>/dev/null; then
  git -C "$CERES_SRC" apply "$CERES_ARCH_PATCH"
elif ! git -C "$CERES_SRC" apply --reverse --check "$CERES_ARCH_PATCH" 2>/dev/null; then
  echo "Ceres CUDA architecture patch neither applies nor appears applied." >&2
  exit 2
fi

if [[ ! -d "$CUDSS_PREFIX/include" ]]; then
  archive="$SRC_ROOT/$CUDSS_ARCHIVE"
  [[ -f "$archive" ]] || curl -L --fail --retry 5 -o "$archive" "$CUDSS_URL"
  mkdir -p "$CUDSS_PREFIX"
  tar -xJf "$archive" --strip-components=1 -C "$CUDSS_PREFIX"
fi

if [[ ! -f "$CUDSS_PREFIX/lib/cmake/cudss/cudss-config.cmake" ]]; then
  echo "cuDSS CMake package not found under $CUDSS_PREFIX" >&2
  exit 2
fi

rm -rf "$BUILD_ROOT/ceres"
cmake -S "$CERES_SRC" -B "$BUILD_ROOT/ceres" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CERES_PREFIX" \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=100 \
  -DCMAKE_PREFIX_PATH="$CUDSS_PREFIX" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_BUILD_RPATH="$CUDSS_PREFIX/lib;$CUDA_ROOT/lib64" \
  -DCMAKE_INSTALL_RPATH="$CUDSS_PREFIX/lib;$CUDA_ROOT/lib64" \
  -DUSE_CUDA=ON \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF
cmake --build "$BUILD_ROOT/ceres" --parallel "$(nproc)"
cmake --install "$BUILD_ROOT/ceres"

rm -rf "$BUILD_ROOT/colmap"
cmake -S "$COLMAP_SRC" -B "$BUILD_ROOT/colmap" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$COLMAP_PREFIX" \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=100 \
  -DCMAKE_PREFIX_PATH="$CERES_PREFIX;$CUDSS_PREFIX" \
  -DCeres_DIR="$CERES_PREFIX/lib/cmake/Ceres" \
  -DCMAKE_BUILD_RPATH="$CERES_PREFIX/lib;$CUDSS_PREFIX/lib;$CUDA_ROOT/lib64" \
  -DCMAKE_INSTALL_RPATH="$CERES_PREFIX/lib;$CUDSS_PREFIX/lib;$CUDA_ROOT/lib64" \
  -DCUDA_ENABLED=ON \
  -DGUI_ENABLED=OFF \
  -DONNX_ENABLED=OFF \
  -DCGAL_ENABLED=OFF \
  -DTESTS_ENABLED=OFF
cmake --build "$BUILD_ROOT/colmap" --parallel "$(nproc)"
cmake --install "$BUILD_ROOT/colmap"

{
  echo "colmap_tag=$COLMAP_TAG"
  echo "colmap_commit=$(git -C "$COLMAP_SRC" rev-parse HEAD)"
  echo "ceres_commit=$(git -C "$CERES_SRC" rev-parse HEAD)"
  echo "cudss_version=$CUDSS_VERSION"
  echo "cuda=$("$CUDA_ROOT/bin/nvcc" --version | tail -n 1)"
  echo "built_at=$(date --iso-8601=seconds)"
} > "$COLMAP_PREFIX/BUILD_MANIFEST.txt"

"$COLMAP_PREFIX/bin/colmap" -h
"$COLMAP_PREFIX/bin/colmap" mapper -h | grep -E \
  'ba_use_gpu|min_num_images_gpu_solver|max_num_images_direct'
