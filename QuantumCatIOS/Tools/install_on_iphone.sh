#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEVICE_ID="${IPHONE_DEVICE_ID:-${IPHONE_UDID:-}}"
APP="${APP:-/tmp/QuantumCatIOSDeviceDD/Build/Products/Debug-iphoneos/QuantumCatIOS.app}"
BUNDLE_ID="${BUNDLE_ID:-com.canal.quantumcat}"

cd "$ROOT_DIR"

if [[ ! -d "$APP" ]]; then
  APP="$(bash QuantumCatIOS/Tools/build_device.sh)"
fi

echo "== Device list =="
xcrun devicectl list devices

if [[ -z "$DEVICE_ID" ]]; then
  DEVICE_ID="$(
    xcrun devicectl list devices \
      | perl -ne 'if (/iPhone/ && /available \(paired\)/ && /([0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12})/) { print $1; exit }'
  )"
fi

if [[ -z "$DEVICE_ID" ]]; then
  DEVICE_ID="$(
    xcrun xctrace list devices 2>/dev/null \
      | perl -ne 'if (/^\S/ .. /^== Devices Offline ==/) { if (!/Mac|Simulator|Offline|^==/ && /([0-9A-F]{8}-[0-9A-F]{16})/) { print $1; exit } }'
  )"
fi

if [[ -z "$DEVICE_ID" ]]; then
  echo "No available paired iPhone was found. Set IPHONE_DEVICE_ID to override." >&2
  exit 1
fi

echo "== Installing $APP on $DEVICE_ID =="
xcrun devicectl device install app --device "$DEVICE_ID" "$APP"

echo "== Launching $BUNDLE_ID on $DEVICE_ID =="
xcrun devicectl device process launch --device "$DEVICE_ID" "$BUNDLE_ID"
