#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
APP="${APP:-/tmp/QuantumCatIOSDeviceDD/Build/Products/Debug-iphoneos/QuantumCatIOS.app}"
LOG_DIR="${LOG_DIR:-research/mobile1052/device}"
DEVICE_ID="${IPHONE_DEVICE_ID:?Set IPHONE_DEVICE_ID to the paired iPhone identifier}"
mkdir -p "$LOG_DIR"
expected="$(shasum -a 256 research/run1052/champion.pt | awk '{print $1}')"
fixture="$(plutil -extract checkpointSHA256 raw -o - "$APP/champion_1052_search_parity.json")"
neural_fixture="$(plutil -extract checkpointSha256 raw -o - "$APP/champion_5p_move_parity.json")"
test "$expected" = "$fixture"
test "$expected" = "$neural_fixture"
test -d "$APP/champion_1052_search.mlmodelc"
codesign --verify --deep --strict "$APP"
printf 'checkpoint_sha256=%s\n' "$expected" | tee "$LOG_DIR/checkpoint.txt"
if [[ "${INSTALL:-1}" == "1" ]]; then
  xcrun devicectl device install app --device "$DEVICE_ID" "$APP" | tee "$LOG_DIR/install.txt"
fi
xcrun devicectl -t 180 device process launch --device "$DEVICE_ID" --terminate-existing --console \
  com.canal.quantumcat -deviceModelMoveParity -deviceModelMoveParityPlayers 5 \
  > "$LOG_DIR/neural_parity.txt" 2>&1
rg 'DEVICE_MODEL_MOVE_PARITY' "$LOG_DIR/neural_parity.txt"
rg -q 'mismatches=0.*live_input_mismatches=0.*passed=true' "$LOG_DIR/neural_parity.txt"
xcrun devicectl -t 300 device process launch --device "$DEVICE_ID" --terminate-existing --console \
  com.canal.quantumcat -deviceChampionSearchParity \
  > "$LOG_DIR/search_parity.txt" 2>&1
rg 'ROOT |SEARCH_.*PARITY|DEVICE_CHAMPION' "$LOG_DIR/search_parity.txt"
rg -q 'DEVICE_CHAMPION_1052_SEARCH_PARITY passed=true enabled=true' "$LOG_DIR/search_parity.txt"
if [[ "${NORMAL_LAUNCH:-1}" == "1" ]]; then
  xcrun devicectl -t 60 device process launch --device "$DEVICE_ID" --terminate-existing \
    com.canal.quantumcat | tee "$LOG_DIR/normal_launch.txt"
fi
echo 'champion_search_bridge: PASS'
