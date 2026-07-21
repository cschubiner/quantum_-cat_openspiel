#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-az_runs/best_3p_policy_checkpoint.pt}"
DEVICE_ID="${IPHONE_DEVICE_ID:-${IPHONE_UDID:-}}"
APP="${APP:-/tmp/QuantumCatIOSDeviceDD/Build/Products/Debug-iphoneos/QuantumCatIOS.app}"
BUNDLE_ID="${BUNDLE_ID:-com.canal.quantumcat}"
BENCHMARK_GAMES="${BENCHMARK_GAMES:-5}"
BENCHMARK_SEATS="${BENCHMARK_SEATS:-3}"
INSTALL="${INSTALL:-1}"
NORMAL_LAUNCH="${NORMAL_LAUNCH:-1}"
REQUIRE_EXACT_MOVE_PARITY="${REQUIRE_EXACT_MOVE_PARITY:-1}"
LOG_DIR="${LOG_DIR:-az_runs/ios_bridge_check_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -d "$APP" ]]; then
  APP="$(bash QuantumCatIOS/Tools/build_device.sh)"
fi

if [[ ! -f "$APP/champion_belief_policy.pt" ]]; then
  echo "Signed app is missing champion_belief_policy.pt: $APP" >&2
  exit 1
fi

expected_sha="$(shasum -a 256 "$CHECKPOINT" | awk '{print $1}')"
app_sha="$(shasum -a 256 "$APP/champion_belief_policy.pt" | awk '{print $1}')"
echo "checkpoint_sha256=$expected_sha"
echo "app_champion_sha256=$app_sha"
if [[ "$expected_sha" != "$app_sha" ]]; then
  echo "App champion checkpoint hash does not match CHECKPOINT." >&2
  exit 2
fi

echo "== Device list =="
xcrun devicectl list devices | tee "$LOG_DIR/devices.txt"

if [[ -z "$DEVICE_ID" ]]; then
  DEVICE_ID="$(
    xcrun devicectl list devices \
      | perl -ne 'if (/iPhone/ && /available \(paired\)/ && /([0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12})/) { print $1; exit }'
  )"
fi

if [[ -z "$DEVICE_ID" ]]; then
  echo "No available paired iPhone was found. Set IPHONE_DEVICE_ID to override." >&2
  exit 1
fi

if [[ "$INSTALL" == "1" ]]; then
  echo "== Installing $APP on $DEVICE_ID =="
  xcrun devicectl device install app --device "$DEVICE_ID" "$APP" \
    | tee "$LOG_DIR/install.txt"
fi

echo "== On-device CoreML benchmark =="
set +e
DEVICECTL_CHILD_DEVICE_PARADOX_BENCHMARK=1 \
DEVICECTL_CHILD_DEVICE_PARADOX_BENCHMARK_GAMES="$BENCHMARK_GAMES" \
DEVICECTL_CHILD_DEVICE_PARADOX_BENCHMARK_SEATS="$BENCHMARK_SEATS" \
xcrun devicectl -t 120 device process launch \
  --device "$DEVICE_ID" \
  --terminate-existing \
  --console \
  "$BUNDLE_ID" \
  >"$LOG_DIR/device_benchmark.txt" 2>&1
benchmark_status=$?
set -e
cat "$LOG_DIR/device_benchmark.txt"
if [[ "$benchmark_status" -ne 0 ]]; then
  if rg -q "Locked|could not be, unlocked" "$LOG_DIR/device_benchmark.txt"; then
    echo "Device is locked; unlock the iPhone and rerun this script." >&2
  fi
  exit "$benchmark_status"
fi

rg -q "DEVICE_PARADOX_BENCHMARK" "$LOG_DIR/device_benchmark.txt"
rg -q "coreml_failures=0" "$LOG_DIR/device_benchmark.txt"
rg -q "shared_bid_check=true" "$LOG_DIR/device_benchmark.txt"
rg -q "shared_discard_check=true" "$LOG_DIR/device_benchmark.txt"

if [[ "$REQUIRE_EXACT_MOVE_PARITY" == "1" ]]; then
  if rg -q "DEVICE_MODEL_MOVE_PARITY|MODEL_MOVE_PARITY" QuantumCatIOS/QuantumCatIOS; then
    echo "Exact move-parity hook detected, but this wrapper must be extended to invoke it." >&2
    exit 4
  fi
  echo "Exact PyTorch-to-CoreML move-parity mode is not present in the iOS app." >&2
  echo "Do not claim exact same-move parity on physical iPhone until that hook exists and passes." >&2
  exit 4
fi

if [[ "$NORMAL_LAUNCH" == "1" ]]; then
  echo "== Normal launch =="
  xcrun devicectl -t 60 device process launch \
    --device "$DEVICE_ID" \
    --terminate-existing \
    "$BUNDLE_ID" \
    | tee "$LOG_DIR/normal_launch.txt"
fi

echo "ios_bridge_check: PASS"
