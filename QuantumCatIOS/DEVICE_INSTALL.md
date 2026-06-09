# Quantum Cat iOS Device Install Notes

## App Identity

- Display name: `Quantum Cat`
- Bundle identifier: `com.canal.quantumcat`
- Apple Developer team: `94NPBVCHV4` / Clayton Schubiner
- App ID registered in Apple Developer on 2026-06-06:
  - `Quantum Cat`
  - `94NPBVCHV4.com.canal.quantumcat`

## Development Provisioning Profile

The real development profile is installed locally:

- Profile name: `Quantum Cat Development`
- UUID: `01482de4-8ee9-4446-88db-11c3259ce50b`
- App ID: `94NPBVCHV4.com.canal.quantumcat`
- Type: iOS Development
- Expires: 2027-06-06
- Certificate: `Clayton Schubiner (Clay's MacBook Pro) (Development)`
- Devices: the registered iPhone and iPad test devices in this Apple Developer team

The downloaded source file is:

```text
/Users/canal/Downloads/Quantum_Cat_Development.mobileprovision
```

The installed profile path is:

```text
/Users/canal/Library/MobileDevice/Provisioning Profiles/01482de4-8ee9-4446-88db-11c3259ce50b.mobileprovision
```

## App Store Provisioning Profile

The real App Store Connect profile is installed locally. It was regenerated on
2026-06-06 to use the modern Apple Distribution certificate so archives can be
exported without keychain or certificate mismatches.

- Profile name: `Quantum Cat App Store`
- UUID: `3c031bd1-0378-4ba1-b275-c9eb14feb4c7`
- App ID: `94NPBVCHV4.com.canal.quantumcat`
- Type: App Store
- Expires: 2027-06-06
- Certificate: `Apple Distribution: Clayton Schubiner (94NPBVCHV4)`
- Certificate SHA-1: `104ACB0986C67B7D264D28D1408580D9DE04F9DD`

The downloaded source file is:

```text
/Users/canal/Downloads/Quantum_Cat_App_Store (1).mobileprovision
```

The installed profile path is:

```text
/Users/canal/Library/MobileDevice/Provisioning Profiles/3c031bd1-0378-4ba1-b275-c9eb14feb4c7.mobileprovision
```

## Devices

- iPhone 17 Pro Max
  - CoreDevice identifier: `7F494DB0-B84F-5ECE-8B15-52C8F081AEC8`
  - Hardware UDID / Xcode destination id used successfully for install: `00008122-000C58D11446801C`
- iPad Air 11-inch (M3)
  - CoreDevice identifier: `F0688612-A069-5A2B-9F4A-A05434261256`
  - Hardware UDID / Xcode destination id: use `xcrun devicectl list devices` to confirm before installing

## Build And Install

Build a signed device app:

```bash
bash QuantumCatIOS/Tools/build_device.sh
```

Install and launch on the iPhone:

```bash
bash QuantumCatIOS/Tools/install_on_iphone.sh
```

If CoreDevice lists the iPhone by Wi-Fi but not as `available (paired)`, the
installer falls back to the online hardware UDID reported by `xctrace`. You can
still override selection explicitly:

```bash
IPHONE_DEVICE_ID=00008122-000C58D11446801C bash QuantumCatIOS/Tools/install_on_iphone.sh
```

Create an App Store/TestFlight archive:

```bash
bash QuantumCatIOS/Tools/archive_appstore.sh
```

The expected installed bundle id is:

```text
com.canal.quantumcat
```

## Verification

Run the verifier:

```bash
bash QuantumCatIOS/Tools/verify_ios_app.sh
```

The verifier covers:

- Swift engine smoke tests for supported human/bot mixes.
- JSON encode/decode persistence smoke for terminal game states.
- Simulator bundle build and artifact checks when Xcode's build service is healthy.
- Existing canonical device bundle artifact checks when a signed device bundle exists.

## Current Device Launch Note

The iPhone install path has been verified with:

```text
App installed:
bundleID: com.canal.quantumcat
```

Readback after uninstalling the old borrowed bundle shows:

```text
Quantum Cat        com.canal.quantumcat       0.1       1
```

Remote launch requires the physical phone to be unlocked and awake. When it is
locked, SpringBoard rejects launch with:

```text
Unable to launch com.canal.quantumcat because the device was not, or could not be, unlocked.
```
