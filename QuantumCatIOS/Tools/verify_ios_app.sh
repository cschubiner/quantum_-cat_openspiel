#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIM_DERIVED_DATA="${SIM_DERIVED_DATA:-/tmp/QuantumCatIOSSimDD}"
DEVICE_BUNDLE="${DEVICE_BUNDLE:-/tmp/QuantumCatIOSDeviceDD/Build/Products/Debug-iphoneos/QuantumCatIOS.app}"
SIM_BUNDLE="$SIM_DERIVED_DATA/Build/Products/Debug-iphonesimulator/QuantumCatIOS.app"

cd "$ROOT_DIR"

echo "== Swift engine smoke =="
xcrun swiftc -parse-as-library \
  QuantumCatIOS/QuantumCatIOS/Models/BotCatalog.swift \
  QuantumCatIOS/QuantumCatIOS/Models/QuantumCatMLPolicy.swift \
  QuantumCatIOS/QuantumCatIOS/Models/QuantumCatGame.swift \
  QuantumCatIOS/Tools/engine_smoke.swift \
  -o /tmp/quantum-cat-engine-smoke
/tmp/quantum-cat-engine-smoke

echo "== Simulator build =="
xcodebuild \
  -project QuantumCatIOS/QuantumCatIOS.xcodeproj \
  -scheme QuantumCatIOS \
  -configuration Debug \
  -sdk iphonesimulator \
  -derivedDataPath "$SIM_DERIVED_DATA" \
  CODE_SIGNING_ALLOWED=NO \
  COMPILER_INDEX_STORE_ENABLE=NO \
  -quiet \
  build

echo "== Simulator bundle artifacts =="
test -d "$SIM_BUNDLE/champion_belief_policy.mlmodelc"
test -d "$SIM_BUNDLE/setpool_distill.mlmodelc"
test -d "$SIM_BUNDLE/raw_policy_league.mlmodelc"
test -d "$SIM_BUNDLE/strict_q_head.mlmodelc"
test -f "$SIM_BUNDLE/champion_belief_policy.pt"
test -f "$SIM_BUNDLE/setpool_distill.pt"
test -f "$SIM_BUNDLE/raw_policy_league.pt"
test -f "$SIM_BUNDLE/strict_q_head.pt"
test -f "$SIM_BUNDLE/BotModels.json"
plutil -extract CFBundleDisplayName raw -o - "$SIM_BUNDLE/Info.plist"
echo
plutil -extract CFBundleIdentifier raw -o - "$SIM_BUNDLE/Info.plist"
echo

if [[ -d "$DEVICE_BUNDLE" ]]; then
  echo "== Existing signed device bundle artifacts =="
  test -d "$DEVICE_BUNDLE/champion_belief_policy.mlmodelc"
  test -d "$DEVICE_BUNDLE/setpool_distill.mlmodelc"
  test -d "$DEVICE_BUNDLE/raw_policy_league.mlmodelc"
  test -d "$DEVICE_BUNDLE/strict_q_head.mlmodelc"
  test -f "$DEVICE_BUNDLE/champion_belief_policy.pt"
  test -f "$DEVICE_BUNDLE/setpool_distill.pt"
  test -f "$DEVICE_BUNDLE/raw_policy_league.pt"
  test -f "$DEVICE_BUNDLE/strict_q_head.pt"
  plutil -extract CFBundleDisplayName raw -o - "$DEVICE_BUNDLE/Info.plist"
  echo
  plutil -extract CFBundleIdentifier raw -o - "$DEVICE_BUNDLE/Info.plist"
  echo
fi

echo "verify_ios_app.sh: PASS"
