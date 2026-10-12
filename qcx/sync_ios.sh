#!/usr/bin/env bash
# Sync the qcx champion (C++ search + parameters) into the iOS app.
#
#   bash qcx/sync_ios.sh              # copy sources, regenerate params + parity fixtures
#   bash qcx/sync_ios.sh --test       # ...then run the iOS unit tests on a simulator
#   bash qcx/sync_ios.sh --install    # ...then build for device + install on the iPhone
#   bash qcx/sync_ios.sh --test --install
#
# Options:
#   --no-fixtures   skip regenerating Python parity fixtures (only if the
#                   engine/search did not change; stale fixtures fail tests)
#   --check         verify the app copies are in sync with qcx/ and exit
#
# Env:
#   QCX_PYTHON      python with pyspiel (default: <main checkout>/.venv/bin/python)
#   SIM_DEST        xcodebuild simulator destination for --test
#   IPHONE_DEVICE_ID  forwarded to QuantumCatIOS/Tools/install_on_iphone.sh
#
# The app compiles these files directly; nothing in Native/qc_* may be edited
# by hand. See qcx/IOS_DEPLOY.md.
set -euo pipefail

QCX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$QCX_DIR/.." && pwd)"
NATIVE_DIR="$ROOT_DIR/QuantumCatIOS/QuantumCatIOS/Native"
SOURCES=(qc_core.h qc_agents.h qc_capi.cpp)

DO_TEST=0; DO_INSTALL=0; DO_FIXTURES=1; CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --test) DO_TEST=1 ;;
    --install) DO_INSTALL=1 ;;
    --no-fixtures) DO_FIXTURES=0 ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

die() { echo "sync_ios: $*" >&2; exit 1; }

# ---------------------------------------------------------------- contract checks
py_view_len="$(sed -n 's/^VIEW_LEN *= *\([0-9][0-9]*\).*/\1/p' "$QCX_DIR/qc_py.py")"
hdr_view_len="$(sed -n 's/^#define QCX_VIEW_LEN *\([0-9][0-9]*\).*/\1/p' "$NATIVE_DIR/qcx_capi.h")"
json_view_len="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["view_len"])' "$QCX_DIR/qcx_champion.json")"
[[ -n "$py_view_len" ]] || die "could not read VIEW_LEN from qcx/qc_py.py"
if [[ "$py_view_len" != "$hdr_view_len" || "$py_view_len" != "$json_view_len" ]]; then
  die "view layout changed: qc_py.py VIEW_LEN=$py_view_len, Native/qcx_capi.h QCX_VIEW_LEN=$hdr_view_len, qcx_champion.json view_len=$json_view_len.
Update QCX_VIEW_LEN, qcx_champion.json and QCXView.build (QuantumCatIOS/QuantumCatIOS/Models/QCXChampion.swift) to mirror qc_py.build_view, then re-run."
fi
for fn in "int qc_decide(const int\* view, int worlds, int util, unsigned long long seed, double\* values_out)" \
          "int qc_legal_from_view(const int\* view, int\* out)"; do
  grep -q "$fn" "$QCX_DIR/qc_capi.cpp" || die "qc_capi.cpp no longer defines: ${fn//\\/} -- update Native/qcx_capi.h and the Swift bridge"
done

if [[ "$CHECK_ONLY" == 1 ]]; then
  status=0
  for f in "${SOURCES[@]}"; do
    cmp -s "$QCX_DIR/$f" "$NATIVE_DIR/$f" || { echo "OUT OF SYNC: $f"; status=1; }
  done
  [[ $status == 0 ]] && echo "in sync"
  exit $status
fi

# ---------------------------------------------------------------- copy sources
mkdir -p "$NATIVE_DIR"
for f in "${SOURCES[@]}"; do
  if ! cmp -s "$QCX_DIR/$f" "$NATIVE_DIR/$f"; then
    cp "$QCX_DIR/$f" "$NATIVE_DIR/$f"
    echo "synced $f"
  fi
done

# ---------------------------------------------------------------- parameters
commit="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if ! git -C "$ROOT_DIR" diff --quiet -- "$QCX_DIR" 2>/dev/null; then commit="$commit+dirty"; fi
/usr/bin/python3 - "$QCX_DIR" "$NATIVE_DIR" "$commit" <<'PY'
import hashlib, json, os, sys
qcx, native, commit = sys.argv[1:4]
doc = json.load(open(os.path.join(qcx, "qcx_champion.json")))
h = hashlib.sha256()
for f in ("qc_core.h", "qc_agents.h", "qc_capi.cpp"):
  h.update(open(os.path.join(qcx, f), "rb").read())
doc["source_commit"] = commit
doc["source_sha256"] = h.hexdigest()
out = os.path.join(native, "qcx_champion.json")
new = json.dumps(doc, indent=2, sort_keys=True) + "\n"
old = open(out).read() if os.path.exists(out) else ""
# Only rewrite when something other than the commit stamp changed.
strip = lambda s: {k: v for k, v in (json.loads(s) if s else {}).items() if k != "source_commit"}
if strip(old) != strip(new):
  open(out, "w").write(new)
  print("wrote Native/qcx_champion.json")
PY

# ---------------------------------------------------------------- parity fixtures
if [[ "$DO_FIXTURES" == 1 ]]; then
  if [[ -z "${QCX_PYTHON:-}" ]]; then
    common="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    for cand in "$ROOT_DIR/.venv/bin/python" "${common%/.git}/.venv/bin/python"; do
      [[ -x "$cand" ]] && { QCX_PYTHON="$cand"; break; }
    done
  fi
  [[ -n "${QCX_PYTHON:-}" && -x "$QCX_PYTHON" ]] || die "no python with pyspiel found; set QCX_PYTHON or pass --no-fixtures"
  echo "building qcx/libqc.dylib"
  clang++ -O3 -std=c++17 -shared -fPIC "$QCX_DIR/qc_capi.cpp" -o "$QCX_DIR/libqc.dylib"
  (cd "$ROOT_DIR" && OMP_NUM_THREADS=1 "$QCX_PYTHON" "$QCX_DIR/export_ios_fixtures.py" "$NATIVE_DIR/qcx_parity_fixtures.json")
fi

for f in "${SOURCES[@]}"; do
  cmp -s "$QCX_DIR/$f" "$NATIVE_DIR/$f" || die "copy verification failed for $f"
done
echo "qcx champion synced into $NATIVE_DIR (source $commit)"

# ---------------------------------------------------------------- optional test / install
if [[ "$DO_TEST" == 1 ]]; then
  SIM_DEST="${SIM_DEST:-platform=iOS Simulator,name=iPhone 17 Pro Max}"
  (cd "$ROOT_DIR" && xcodebuild test \
    -project QuantumCatIOS/QuantumCatIOS.xcodeproj \
    -scheme QuantumCatIOS \
    -destination "$SIM_DEST" \
    -only-testing:QuantumCatIOSTests \
    -derivedDataPath "${TEST_DERIVED_DATA:-/tmp/QuantumCatIOSTestDD}" \
    COMPILER_INDEX_STORE_ENABLE=NO)
fi

if [[ "$DO_INSTALL" == 1 ]]; then
  APP="$(bash "$ROOT_DIR/QuantumCatIOS/Tools/build_device.sh" | tail -n 1)"
  export APP
  attempts="${INSTALL_ATTEMPTS:-6}"
  for i in $(seq 1 "$attempts"); do
    if bash "$ROOT_DIR/QuantumCatIOS/Tools/install_on_iphone.sh"; then
      echo "installed $APP"
      exit 0
    fi
    echo "install attempt $i/$attempts failed (phone locked or unreachable?); retrying in 30s" >&2
    sleep 30
  done
  die "install failed after $attempts attempts: unlock the iPhone, keep it awake on the same Wi-Fi, and re-run with --install"
fi
