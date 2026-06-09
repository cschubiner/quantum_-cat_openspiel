#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DERIVED_DATA="${DERIVED_DATA:-/tmp/QuantumCatIOSDeviceDD}"

cd "$ROOT_DIR"

xcodebuild \
  -project QuantumCatIOS/QuantumCatIOS.xcodeproj \
  -scheme QuantumCatIOS \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  -derivedDataPath "$DERIVED_DATA" \
  DEVELOPMENT_TEAM=94NPBVCHV4 \
  PRODUCT_BUNDLE_IDENTIFIER=com.canal.quantumcat \
  CODE_SIGN_STYLE=Manual \
  PROVISIONING_PROFILE_SPECIFIER='Quantum Cat Development' \
  CODE_SIGN_IDENTITY='Apple Development' \
  COMPILER_INDEX_STORE_ENABLE=NO \
  -quiet \
  build

APP="$DERIVED_DATA/Build/Products/Debug-iphoneos/QuantumCatIOS.app"
test -d "$APP/champion_belief_policy.mlmodelc"
test -d "$APP/setpool_distill.mlmodelc"
test -d "$APP/raw_policy_league.mlmodelc"
test -d "$APP/strict_q_head.mlmodelc"
test -f "$APP/champion_belief_policy.pt"
test -f "$APP/setpool_distill.pt"
test -f "$APP/raw_policy_league.pt"
test -f "$APP/strict_q_head.pt"
test "$(plutil -extract CFBundleIdentifier raw -o - "$APP/Info.plist")" = "com.canal.quantumcat"

echo "$APP"
