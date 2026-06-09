#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE_PATH="${ARCHIVE_PATH:-/tmp/QuantumCatIOSArchive/QuantumCatIOS.xcarchive}"
EXPORT_PATH="${EXPORT_PATH:-/tmp/QuantumCatIOSArchive/export}"

cd "$ROOT_DIR"

xcodebuild archive \
  -project QuantumCatIOS/QuantumCatIOS.xcodeproj \
  -scheme QuantumCatIOS \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  -archivePath "$ARCHIVE_PATH" \
  DEVELOPMENT_TEAM=94NPBVCHV4 \
  PRODUCT_BUNDLE_IDENTIFIER=com.canal.quantumcat \
  CODE_SIGN_STYLE=Manual \
  PROVISIONING_PROFILE_SPECIFIER='Quantum Cat App Store' \
  CODE_SIGN_IDENTITY='Apple Distribution' \
  COMPILER_INDEX_STORE_ENABLE=NO

test -d "$ARCHIVE_PATH/Products/Applications/QuantumCatIOS.app"
test "$(plutil -extract CFBundleIdentifier raw -o - "$ARCHIVE_PATH/Products/Applications/QuantumCatIOS.app/Info.plist")" = "com.canal.quantumcat"

rm -rf "$EXPORT_PATH"
xcodebuild -exportArchive \
  -archivePath "$ARCHIVE_PATH" \
  -exportPath "$EXPORT_PATH" \
  -exportOptionsPlist QuantumCatIOS/ExportOptions-AppStore.plist

test -f "$EXPORT_PATH/QuantumCatIOS.ipa"

echo "$ARCHIVE_PATH"
echo "$EXPORT_PATH/QuantumCatIOS.ipa"
