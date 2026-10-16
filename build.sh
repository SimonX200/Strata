#!/bin/sh
# Strata: compile the engine from this checkout, from scratch.
#
#   ./build.sh                                      fresh CUDA build for this PC's GPUs -> engine/strata
#   ./build.sh --vision                             also the image encoder -> engine/strata-vision
#   ./build.sh --tests                              the whole test suite too, then ctest
#   ./build.sh --arch 86,89                         other compute capabilities than the cards in this PC
#   ./build.sh --arch 86-real                       SASS only, no PTX: an 18 % smaller binary for this one card
#   ./build.sh --keep                               keep build/: compile only what changed
#   ./build.sh --no-install                         leave the binaries in build/, write nothing to engine/
#   ./build.sh --jobs 8                             build threads (default: half the cores, as setup.py)
#   ./build.sh --nvcc /opt/cuda-12/bin/nvcc         the toolkit to build with
#   ./build.sh --dir build-dev                      another build folder (its encoder: build-dev-vision)
#   ./build.sh --configure-only                     stop after CMake's configure (to check the setup)
#   ./build.sh --extra -DSTRATA_NATIVE_EXPERTS=ON   more CMake definitions for the engine
#
# The same steps `setup.py --build` runs, without its questions: llama.cpp at the pinned commit
# (third_party/llama.cpp, a zip - no git), Ninja into build/ (build-vision/ for the encoder) with
# CMAKE_BUILD_TYPE=Release, the engine target `strata`, then a copy into engine/ with the BUILD.json stamp
# setup.py reads to decide whether the engine there was built from this source.  A folder that is still
# there is removed first, so nothing of an earlier configure (an old arch, an old ggml option) survives:
# that is what "komplett neu" means here.  --keep keeps the folder and compiles only what changed.
#
# --arch takes CMake's architecture suffixes: a plain 86 embeds PTX and SASS (what setup.py and the ready-made
# engine do, and what engine/BUILD.json records), 86-real only the SASS of that one card.  Measured on the
# heaviest kernel file (src/kernels/cuda/iq_kernels.cu, CUDA 13.4, RTX 3090 Ti, one nvcc at a time): the object
# went 6.14 MB -> 4.94 MB and the compile 42.6 s -> 41.7 s, so the win is size, not time.  What is given up is
# the driver's JIT: a plain-86 engine still runs on a newer card (its PTX is compiled there on the first run),
# an 86-real engine on any other architecture stops with "no kernel image is available for execution".
#
# A failed compile is tried once more: CUDA 13.0's ptxas sometimes cannot parse a PTX file it just wrote
# (issue #45), and the second attempt gets past it.
#
# Needs cmake 3.24+, ninja, a CUDA toolkit (nvcc: 13.x for RTX 20 or newer, 12.x for Pascal/Volta) and a
# C++ compiler.  cmake and ninja come from PATH or from .venv/bin, where setup's pip puts them; the
# toolkit from $STRATA_NVCC, PATH, /usr/local/cuda or /opt/cuda.  A full engine compile takes 10-20
# minutes on a 32-thread desktop CPU (docs/INSTALL.md).  AMD (HIP) and Intel Arc (SYCL) build differently:
# docs/AMD_HIP.md, docs/INTEL_ARC.md.
set -eu

cd "$(dirname "$0")" || exit 1
ROOT=$PWD
ARCHS=""                                  # empty: read the cards in this PC
VISION=0 TESTS=OFF KEEP=0 INSTALL=1 CONF=0 JOBS="" BDIR="" VDIR="" EXTRA=""

say() { printf '%s\n' "$*"; }
die() { printf 'build.sh: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --vision) VISION=1 ;;
    --tests) TESTS=ON ;;
    --arch|-a) ARCHS=$(printf '%s' "${2-}" | tr ',' ';' | tr -d ' '); shift ;;
    --keep) KEEP=1 ;;
    --no-install) INSTALL=0 ;;
    --jobs|-j) JOBS="${2-}"; shift ;;
    --extra) EXTRA="$EXTRA ${2-}"; shift ;;
    --nvcc) STRATA_NVCC="${2-}"; shift ;;
    --dir) BDIR="${2-}"; shift ;;
    --configure-only) CONF=1 ;;
    -h|--help) sed -n '2,/^set -eu/p' "$0" | sed '$d'; exit 0 ;;
    *) die "unknown option '$1' (./build.sh --help)" ;;
  esac
  shift
done
# the build folders setup.py uses (build/, build-vision/); --dir takes another pair
BDIR=${BDIR:-build}; VDIR=${VDIR:-${BDIR}-vision}
case "$BDIR" in /*) ;; *) BDIR=$ROOT/$BDIR ;; esac
case "$VDIR" in /*) ;; *) VDIR=$ROOT/$VDIR ;; esac

# ---- the tools: PATH first, then what setup's pip installed next to the project's Python
find_tool() {                             # setup.py's find_tool, in sh
  command -v "$1" 2>/dev/null && return 0
  for d in "$ROOT/.venv/bin" "$HOME/.local/bin"; do
    if [ -x "$d/$1" ]; then printf '%s\n' "$d/$1"; return 0; fi
  done
  return 1
}
CMAKE=$(find_tool cmake) || die "cmake not found: sudo apt install cmake, or .venv/bin/python -m pip install cmake"
NINJA=$(find_tool ninja) || die "ninja not found: sudo apt install ninja-build, or .venv/bin/python -m pip install ninja"
PY=$ROOT/.venv/bin/python
[ -x "$PY" ] || PY=$(find_tool python3) || die "no Python 3 to run setup.py's helpers with"
NVCC=${STRATA_NVCC:-}
if [ -z "$NVCC" ]; then
  NVCC=$(find_tool nvcc || true)
  for c in /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
    if [ -z "$NVCC" ] && [ -x "$c" ]; then NVCC=$c; fi
  done
fi
[ -n "$NVCC" ] || die "nvcc not found: install the CUDA Toolkit, or point --nvcc / STRATA_NVCC at it"

# ---- the CUDA architectures: the cards in this PC, or the ones --arch names
if [ -z "$ARCHS" ]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi to read the compute capability: pass --arch 86"
  ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' \r' | sed 's/\.//g' \
          | sort -un | paste -sd ';')
  [ -n "$ARCHS" ] || die "nvidia-smi reported no GPU: pass --arch 86"
fi
# the plain numbers behind them: the guard below and engine/BUILD.json want ints, CMake takes the suffixes
NUM=$(printf '%s\n' "$ARCHS" | tr ';' '\n' | sed 's/-.*$//; s/[af]$//' | grep -E '^[0-9]+$' | sort -un | paste -sd ';')
if [ -z "$NUM" ]; then      # native / all: the stamp still wants the numbers of the cards in this PC
  NUM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' \r' | sed 's/\.//g' \
        | sort -un | paste -sd ';' || true)
fi
[ -n "$NUM" ] || die "no architecture number to record in engine/BUILD.json in '$ARCHS': pass --arch 86"
MIN=$(printf '%s\n' "$NUM" | tr ';' '\n' | sort -n | head -1)
CUDA_V=$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1.\2/p')
say "building the engine for sm_$(printf '%s' "$ARCHS" | tr ';' ',') with CUDA ${CUDA_V:-?} ($NVCC)"
if [ "$MIN" -lt 75 ]; then
  EXTRA="$EXTRA -DSTRATA_EXPERIMENTAL_SM60=ON"     # the experimental Pascal / Volta build, docs/OLDER_GPUS.md
  if [ "${CUDA_V%%.*}" -ge 13 ] 2>/dev/null; then
    say "  note: sm_$MIN needs a CUDA 12.x toolkit (CUDA 13 dropped Pascal and Volta): --nvcc /opt/cuda-12/bin/nvcc"
  fi
fi

# ---- llama.cpp at the pinned commit (setup.py's get_llama_cpp: a zip, no git)
LLAMA=$ROOT/third_party/llama.cpp
if [ ! -f "$LLAMA/ggml/CMakeLists.txt" ]; then
  say "fetching llama.cpp at the pinned commit (a few hundred MB, once) ..."
  "$PY" -c 'import setup; setup.get_llama_cpp()' >/dev/null
fi
[ -f "$LLAMA/ggml/CMakeLists.txt" ] || die "$LLAMA has no ggml: run .venv/bin/python -c 'import setup; setup.get_llama_cpp()'"

# ---- from scratch: a configured folder keeps its CMakeCache (an old arch, an old ggml option, an old floor)
if [ "$KEEP" = 0 ]; then
  say "removing $(basename "$BDIR")$(if [ "$VISION" = 1 ]; then printf '/%s' "$(basename "$VDIR")"; fi) - a fresh build (--keep compiles only what changed)"
  rm -rf "$BDIR"
  if [ "$VISION" = 1 ]; then rm -rf "$VDIR"; fi
fi

# ---- configure and compile: the definitions setup.py's build_engine passes
TOOLKIT=$(CDPATH= cd -- "$(dirname "$NVCC")/.." && pwd)    # CUDAToolkit_ROOT: CMake must not mix two toolkits
set -- -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" -S "$ROOT" -B "$BDIR" -DCMAKE_BUILD_TYPE=Release \
       -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=$TESTS -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
       -DCMAKE_CUDA_COMPILER="$NVCC" -DSTRATA_GGML_DIR="$LLAMA"
if [ "$TOOLKIT" != "/usr" ]; then set -- "$@" -DCUDAToolkit_ROOT="$TOOLKIT"; fi
if [ -n "$EXTRA" ]; then set -- "$@" $EXTRA; fi
"$CMAKE" "$@"
if [ "$CONF" = 1 ]; then
  say "configured: $BDIR (cmake --build $BDIR --target strata -j <n>)"
  exit 0
fi

JOBS=${JOBS:-$(( $(nproc 2>/dev/null || echo 4) / 2 ))}
[ "$JOBS" -ge 2 ] 2>/dev/null || JOBS=2
build() {   # $1 build folder, $2 target; a CUDA 13.0 ptxas hiccup (issue #45) is retried, as setup.py does
  if ! "$CMAKE" --build "$1" --target "$2" -j "$JOBS"; then
    say "  (the build stopped - trying it once more)"
    "$CMAKE" --build "$1" --target "$2" -j "$JOBS"
  fi
}
if [ "$TESTS" = ON ]; then build "$BDIR" all; else build "$BDIR" strata; fi
if [ "$TESTS" = ON ]; then
  CTEST=$(find_tool ctest || true)
  if [ -n "$CTEST" ]; then
    "$CTEST" --test-dir "$BDIR" -j "$JOBS" --output-on-failure || say "  (some tests failed - they are listed above)"
  else
    say "ctest not found: run the tests with .venv/bin/python -m pip install cmake"
  fi
fi

# ---- the image encoder (optional): tools/vision is its own project, as in setup.py
if [ "$VISION" = 1 ]; then
  set -- -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" -S "$ROOT/tools/vision" -B "$VDIR" -DCMAKE_BUILD_TYPE=Release \
         -DLLAMA_DIR="$LLAMA" -DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=OFF \
         -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DCMAKE_CUDA_COMPILER="$NVCC"
  if [ "$TOOLKIT" != "/usr" ]; then set -- "$@" -DCUDAToolkit_ROOT="$TOOLKIT"; fi
  "$CMAKE" "$@"
  build "$VDIR" strata-vision
fi

if [ "$INSTALL" = 0 ]; then
  say "not installed: run it with LD_LIBRARY_PATH=$TOOLKIT/lib64 $BDIR/strata --version"
  exit 0
fi

# ---- install into engine/, with the BUILD.json stamp setup.py reads (engine/BUILD.json, source_hash)
mkdir -p "$ROOT/engine"
install_one() { cp "$1" "$ROOT/engine/$2.new"; mv -f "$ROOT/engine/$2.new" "$ROOT/engine/$2"; }  # mv: a running engine cannot be written to
install_one "$BDIR/strata" strata
if [ "$VISION" = 1 ]; then install_one "$VDIR/bin/strata-vision" strata-vision; fi
"$PY" - "$NVCC" "$NUM" "$VISION" <<'PY'
import json, pathlib, sys
import setup
nvcc, archs, vision = sys.argv[1], sys.argv[2].split(";"), sys.argv[3] == "1"
bindir = pathlib.Path(nvcc).resolve().parent
dirs = [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()]
stamp, enc = setup.ROOT / "engine" / "BUILD.json", setup.ROOT / "engine" / setup.VEXE
old = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
stamp.write_text(json.dumps({
    "source": "local", "version": setup.source_version(), "archs": [int(a) for a in archs],
    "vision": "gpu" if vision else "none", "cuda_dirs": dirs,
    "src": setup.source_hash(setup.ENGINE_SOURCES),
    # an encoder that was not rebuilt here keeps the stamp it had, so setup does not compile it again
    "vision_src": setup.source_hash(setup.VISION_SOURCES) if vision
                  else (old.get("vision_src") if enc.exists() else None),
}, indent=1), encoding="utf-8")
PY

say "engine: $ROOT/engine/strata $(LD_LIBRARY_PATH=$TOOLKIT/lib64 "$ROOT/engine/strata" --version 2>/dev/null || echo '(it did not run - check the CUDA libraries above)')"
if [ "$VISION" = 1 ]; then say "encoder: $ROOT/engine/strata-vision"; fi
say "next: ./run-<model>.sh, or .venv/bin/python serve/server.py --engine strata --config strata-<model>.json"
