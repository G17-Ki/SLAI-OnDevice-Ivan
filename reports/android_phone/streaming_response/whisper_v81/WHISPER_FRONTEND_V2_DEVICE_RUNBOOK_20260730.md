# Whisper frontend-v2 sequential device runbook

Date: 2026-07-30

## Status and authorization boundary

This is a prepared runbook, not an execution record. None of the ADB commands
below were run while writing or validating this document.

The required order is:

1. five-language frontend verify-all;
2. ZH semantic, two runs from one APK;
3. KO semantic, two runs from one APK;
4. RU semantic, two runs from one APK.

Never rebuild between the two repetitions of one semantic language. Never
proceed to install if the clean rebuild does not reproduce the pinned whole-APK
identity or the independent local verifier does not pass. The collector is the
only command below that installs an APK; it verifies the exact v2 profile and
whole-APK trust root before invoking ADB.

## Frozen identities

| Lane | APK bytes | APK SHA-256 | Spec bytes | Spec SHA-256 | Committed local-report SHA-256 |
|---|---:|---|---:|---|---|
| v2 verify-all | 414,786,946 | `aad2f5959de6d9f9ae77db028e50ea1e51fe545fee2bb2f206fddf018452bcc9` | 7,635 | `1bf369d7efa093d664c3e4f29a75723cd3de77936ae00e98212c9504fc16a291` | `a34f610225c86d70aad32d3994577214a58628b0dc81b156bb2432208901e617` |
| ZH semantic | 414,786,946 | `b33bf51b988c1def433d42d37fc00d7df1a2de788c5758efab0be68ef0331d3c` | 7,932 | `1a77a5804df42ee58d38d93220e98929696896c9d780249d990fb45be1ac78b4` | `cb00f9e51b4c53081243f66ae3d5e3c27e928f8c09d8bc9cbf719fe87420f913` |
| KO semantic | 414,786,946 | `81b75cdb237d00acffe9696f1b463d841a036f2fd8208917af6b072be47325e7` | 7,933 | `8564002d4e2ddb624410f40b53afd7e9ca0dd70794d2c046fbe7b07f7ad3b54a` | `16e960fbb004c5da83416978397cccbafc8101c9e16efdfa54372f4d335fc3ac` |
| RU semantic | 414,786,946 | `e058489e6946e21a32f28aff458330cddc686f43a61d4f939f5cada5ceaebb9e` | 7,942 | `07b2f69f60a2d78dc9f6fc07a02a3249ab90300333b33523b6aa1a643a7fef51` | `a378e305a77cc376bb421f2c13fcbe7353f6188c880651ad9263afc8827323cb` |

The unchanged base spec is 9,332 bytes with SHA-256
`af7859d8d3cb69facd202b39fceab11e71a7d2e068b11bf0ea32575c562444bd`.
The protected package hashes pinned by every lane are:

- `ai.liquid.lfmdemo`:
  `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`
- `ai.liquid.lfmdemo.runtime`:
  `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`

## Common shell variables and host preflight

Run from a fresh shell. Replace only `LMF_DEVICE_SERIAL`, using the exact
authorized device serial. Do not derive it by running ADB until the device phase
is explicitly authorized.

```sh
LMF_REPO="$(git rev-parse --show-toplevel)"
LMF_ANDROID_APP="$LMF_REPO/android-app"
LMF_APK="$LMF_REPO/android-app/whisper-probe/build/outputs/apk/debug/whisper-probe-debug.apk"
LMF_BASE_SPEC="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/nubia_whisper_ort_qnn_probe_spec.json"
LMF_PAYLOAD=/private/tmp/lmf-whisper-ort-qnn-probe-payload-en29c-v4
LMF_FRONTEND_V2=/private/tmp/lmf-whisper-frontend-v2-assets-20260730-a1
LMF_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo"
LMF_ANDROID_SDK="$LMF_CACHE_ROOT/android-sdk"
LMF_ADB="$LMF_ANDROID_SDK/platform-tools/adb"
LMF_DEVICE_SERIAL='REPLACE_WITH_EXACT_AUTHORIZED_SERIAL'

test -d "$LMF_PAYLOAD"
test -f "$LMF_PAYLOAD/assets/whisper_probe/payload_manifest.json"
test -f "$LMF_FRONTEND_V2/assets/whisper_frontend/v2/manifest.json"
test -x "$LMF_ADB"

shasum -a 256 "$LMF_BASE_SPEC"
shasum -a 256 "$LMF_FRONTEND_V2/assets/whisper_frontend/v2/manifest.json"
```

Expected hashes are the base-spec hash above and v2 manifest SHA-256
`acb5c46f3bbbe3c11620bd662242157b30695500c1a0e8b6e7510acdc3157736`.

The device must already be in the accepted safety state: airplane mode on, no
unexpected ADB forwards, protected main/runtime APK hashes exact, and no other
candidate process active. The collector rechecks protected hashes before and
after its install/run and fails closed on any drift.

## 1. Build, authenticate, and collect v2 verify-all

### Clean rebuild and lint

```sh
cd "$LMF_ANDROID_APP"
env \
  JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  ANDROID_HOME="$LMF_ANDROID_SDK" \
  ANDROID_SDK_ROOT="$LMF_ANDROID_SDK" \
  WHISPER_PROBE_PAYLOAD_DIR="$LMF_PAYLOAD" \
  WHISPER_FRONTEND_ASSET_DIR="$LMF_FRONTEND_V2" \
  WHISPER_FRONTEND_PROFILE=v2 \
  WHISPER_SEMANTIC_DECODE_ENABLED=false \
  WHISPER_SEMANTIC_LANGUAGE=en \
  ./gradlew :whisper-probe:clean :whisper-probe:assembleDebug \
    :whisper-probe:lintDebug --max-workers=2 --no-daemon
cd "$LMF_REPO"

stat -f '%z %N' "$LMF_APK"
shasum -a 256 "$LMF_APK"
```

Expected APK: 414,786,946 bytes and SHA-256
`aad2f5959de6d9f9ae77db028e50ea1e51fe545fee2bb2f206fddf018452bcc9`.

### Independent local gate

```sh
LMF_SPEC="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/nubia_whisper_android_frontend_multilang_v2_probe_spec.json"
LMF_COMMITTED_LOCAL="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/whisper_android_frontend_multilang_v2_local_verify.json"
LMF_PREINSTALL_LOCAL=/private/tmp/whisper-v2-verifyall-preinstall-local-verify.json

python3 "$LMF_REPO/scripts/verify_whisper_android_frontend_apk.py" \
  --apk "$LMF_APK" \
  --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" \
  --frontend-profile v2 \
  --report "$LMF_PREINSTALL_LOCAL"
cmp "$LMF_PREINSTALL_LOCAL" "$LMF_COMMITTED_LOCAL"
```

### One isolated target run

This collector command locally authenticates the v2 trust root again, installs
only `ai.liquid.lfmdemo.whisperprobe`, runs all five frontend fixtures, captures
raw process evidence, and finalizes through the hardened device verifier.

```sh
LMF_RUN_ID=frontend-v2-20260730-aad2f5-r1
LMF_RUN_DIR="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_frontend_v2_verifyall_run_20260730_aad2f5_r1"

python3 "$LMF_REPO/scripts/run_whisper_android_frontend_device_probe.py" \
  --serial "$LMF_DEVICE_SERIAL" \
  --adb "$LMF_ADB" \
  --apk "$LMF_APK" \
  --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" \
  --frontend-profile v2 \
  --output-dir "$LMF_RUN_DIR" \
  --run-id "$LMF_RUN_ID" \
  --timeout-seconds 300

"$LMF_ADB" -s "$LMF_DEVICE_SERIAL" shell am force-stop ai.liquid.lfmdemo.whisperprobe
```

The output directory must be absent or empty before collection. If collection
fails, retain its archived evidence, run the force-stop command anyway, and do
not reuse or delete the directory.

## 2. Build, authenticate, and collect ZH semantic twice

### Clean rebuild and local gate

```sh
cd "$LMF_ANDROID_APP"
env \
  JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  ANDROID_HOME="$LMF_ANDROID_SDK" \
  ANDROID_SDK_ROOT="$LMF_ANDROID_SDK" \
  WHISPER_PROBE_PAYLOAD_DIR="$LMF_PAYLOAD" \
  WHISPER_FRONTEND_ASSET_DIR="$LMF_FRONTEND_V2" \
  WHISPER_FRONTEND_PROFILE=v2 \
  WHISPER_SEMANTIC_DECODE_ENABLED=true \
  WHISPER_SEMANTIC_LANGUAGE=zh \
  ./gradlew :whisper-probe:clean :whisper-probe:assembleDebug \
    :whisper-probe:lintDebug --max-workers=2 --no-daemon
cd "$LMF_REPO"

stat -f '%z %N' "$LMF_APK"
shasum -a 256 "$LMF_APK"

LMF_SPEC="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/nubia_whisper_android_frontend_multilang_v2_semantic_zh_probe_spec.json"
LMF_COMMITTED_LOCAL="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/whisper_android_frontend_multilang_v2_semantic_zh_local_verify.json"
LMF_PREINSTALL_LOCAL=/private/tmp/whisper-v2-semantic-zh-preinstall-local-verify.json

python3 "$LMF_REPO/scripts/verify_whisper_android_frontend_apk.py" \
  --apk "$LMF_APK" \
  --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" \
  --frontend-profile v2 \
  --report "$LMF_PREINSTALL_LOCAL"
cmp "$LMF_PREINSTALL_LOCAL" "$LMF_COMMITTED_LOCAL"
```

Expected APK SHA-256:
`b33bf51b988c1def433d42d37fc00d7df1a2de788c5758efab0be68ef0331d3c`.

### Repetition 1

```sh
LMF_RUN_ID=semantic-zh-v2-20260730-b33bf5-r1
LMF_RUN_DIR="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_zh_v2_run_20260730_b33bf5_r1"

python3 "$LMF_REPO/scripts/run_whisper_android_frontend_device_probe.py" \
  --serial "$LMF_DEVICE_SERIAL" --adb "$LMF_ADB" \
  --apk "$LMF_APK" --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" --frontend-profile v2 \
  --output-dir "$LMF_RUN_DIR" --run-id "$LMF_RUN_ID" \
  --timeout-seconds 300

python3 "$LMF_REPO/scripts/verify_whisper_semantic_device.py" \
  --semantic-spec "$LMF_SPEC" --base-spec "$LMF_BASE_SPEC" \
  --local-verify "$LMF_RUN_DIR/local_apk_verify.json" \
  --device-report "$LMF_RUN_DIR/device_report_${LMF_RUN_ID}.json" \
  --evidence "$LMF_RUN_DIR/device_evidence_${LMF_RUN_ID}.json" \
  --run-id "$LMF_RUN_ID" \
  --output "$LMF_RUN_DIR/semantic_verify_${LMF_RUN_ID}.json"

"$LMF_ADB" -s "$LMF_DEVICE_SERIAL" shell am force-stop ai.liquid.lfmdemo.whisperprobe
```

### Repetition 2, without rebuilding

```sh
LMF_RUN_ID=semantic-zh-v2-20260730-b33bf5-r2
LMF_RUN_DIR="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_zh_v2_run_20260730_b33bf5_r2"

python3 "$LMF_REPO/scripts/run_whisper_android_frontend_device_probe.py" \
  --serial "$LMF_DEVICE_SERIAL" --adb "$LMF_ADB" \
  --apk "$LMF_APK" --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" --frontend-profile v2 \
  --output-dir "$LMF_RUN_DIR" --run-id "$LMF_RUN_ID" \
  --timeout-seconds 300

python3 "$LMF_REPO/scripts/verify_whisper_semantic_device.py" \
  --semantic-spec "$LMF_SPEC" --base-spec "$LMF_BASE_SPEC" \
  --local-verify "$LMF_RUN_DIR/local_apk_verify.json" \
  --device-report "$LMF_RUN_DIR/device_report_${LMF_RUN_ID}.json" \
  --evidence "$LMF_RUN_DIR/device_evidence_${LMF_RUN_ID}.json" \
  --run-id "$LMF_RUN_ID" \
  --output "$LMF_RUN_DIR/semantic_verify_${LMF_RUN_ID}.json"

"$LMF_ADB" -s "$LMF_DEVICE_SERIAL" shell am force-stop ai.liquid.lfmdemo.whisperprobe
```

## 3. Build, authenticate, and collect KO semantic twice

### Clean rebuild and local gate

```sh
cd "$LMF_ANDROID_APP"
env \
  JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  ANDROID_HOME="$LMF_ANDROID_SDK" \
  ANDROID_SDK_ROOT="$LMF_ANDROID_SDK" \
  WHISPER_PROBE_PAYLOAD_DIR="$LMF_PAYLOAD" \
  WHISPER_FRONTEND_ASSET_DIR="$LMF_FRONTEND_V2" \
  WHISPER_FRONTEND_PROFILE=v2 \
  WHISPER_SEMANTIC_DECODE_ENABLED=true \
  WHISPER_SEMANTIC_LANGUAGE=ko \
  ./gradlew :whisper-probe:clean :whisper-probe:assembleDebug \
    :whisper-probe:lintDebug --max-workers=2 --no-daemon
cd "$LMF_REPO"

stat -f '%z %N' "$LMF_APK"
shasum -a 256 "$LMF_APK"

LMF_SPEC="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/nubia_whisper_android_frontend_multilang_v2_semantic_ko_probe_spec.json"
LMF_COMMITTED_LOCAL="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/whisper_android_frontend_multilang_v2_semantic_ko_local_verify.json"
LMF_PREINSTALL_LOCAL=/private/tmp/whisper-v2-semantic-ko-preinstall-local-verify.json

python3 "$LMF_REPO/scripts/verify_whisper_android_frontend_apk.py" \
  --apk "$LMF_APK" --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" --frontend-profile v2 \
  --report "$LMF_PREINSTALL_LOCAL"
cmp "$LMF_PREINSTALL_LOCAL" "$LMF_COMMITTED_LOCAL"
```

Expected APK SHA-256:
`81b75cdb237d00acffe9696f1b463d841a036f2fd8208917af6b072be47325e7`.

### Repetition 1

```sh
LMF_RUN_ID=semantic-ko-v2-20260730-81b75c-r1
LMF_RUN_DIR="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_ko_v2_run_20260730_81b75c_r1"

python3 "$LMF_REPO/scripts/run_whisper_android_frontend_device_probe.py" \
  --serial "$LMF_DEVICE_SERIAL" --adb "$LMF_ADB" \
  --apk "$LMF_APK" --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" --frontend-profile v2 \
  --output-dir "$LMF_RUN_DIR" --run-id "$LMF_RUN_ID" \
  --timeout-seconds 300

python3 "$LMF_REPO/scripts/verify_whisper_semantic_device.py" \
  --semantic-spec "$LMF_SPEC" --base-spec "$LMF_BASE_SPEC" \
  --local-verify "$LMF_RUN_DIR/local_apk_verify.json" \
  --device-report "$LMF_RUN_DIR/device_report_${LMF_RUN_ID}.json" \
  --evidence "$LMF_RUN_DIR/device_evidence_${LMF_RUN_ID}.json" \
  --run-id "$LMF_RUN_ID" \
  --output "$LMF_RUN_DIR/semantic_verify_${LMF_RUN_ID}.json"

"$LMF_ADB" -s "$LMF_DEVICE_SERIAL" shell am force-stop ai.liquid.lfmdemo.whisperprobe
```

### Repetition 2, without rebuilding

```sh
LMF_RUN_ID=semantic-ko-v2-20260730-81b75c-r2
LMF_RUN_DIR="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_ko_v2_run_20260730_81b75c_r2"

python3 "$LMF_REPO/scripts/run_whisper_android_frontend_device_probe.py" \
  --serial "$LMF_DEVICE_SERIAL" --adb "$LMF_ADB" \
  --apk "$LMF_APK" --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" --frontend-profile v2 \
  --output-dir "$LMF_RUN_DIR" --run-id "$LMF_RUN_ID" \
  --timeout-seconds 300

python3 "$LMF_REPO/scripts/verify_whisper_semantic_device.py" \
  --semantic-spec "$LMF_SPEC" --base-spec "$LMF_BASE_SPEC" \
  --local-verify "$LMF_RUN_DIR/local_apk_verify.json" \
  --device-report "$LMF_RUN_DIR/device_report_${LMF_RUN_ID}.json" \
  --evidence "$LMF_RUN_DIR/device_evidence_${LMF_RUN_ID}.json" \
  --run-id "$LMF_RUN_ID" \
  --output "$LMF_RUN_DIR/semantic_verify_${LMF_RUN_ID}.json"

"$LMF_ADB" -s "$LMF_DEVICE_SERIAL" shell am force-stop ai.liquid.lfmdemo.whisperprobe
```

## 4. Build, authenticate, and collect RU semantic twice

### Clean rebuild and local gate

```sh
cd "$LMF_ANDROID_APP"
env \
  JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  ANDROID_HOME="$LMF_ANDROID_SDK" \
  ANDROID_SDK_ROOT="$LMF_ANDROID_SDK" \
  WHISPER_PROBE_PAYLOAD_DIR="$LMF_PAYLOAD" \
  WHISPER_FRONTEND_ASSET_DIR="$LMF_FRONTEND_V2" \
  WHISPER_FRONTEND_PROFILE=v2 \
  WHISPER_SEMANTIC_DECODE_ENABLED=true \
  WHISPER_SEMANTIC_LANGUAGE=ru \
  ./gradlew :whisper-probe:clean :whisper-probe:assembleDebug \
    :whisper-probe:lintDebug --max-workers=2 --no-daemon
cd "$LMF_REPO"

stat -f '%z %N' "$LMF_APK"
shasum -a 256 "$LMF_APK"

LMF_SPEC="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/nubia_whisper_android_frontend_multilang_v2_semantic_ru_probe_spec.json"
LMF_COMMITTED_LOCAL="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/whisper_android_frontend_multilang_v2_semantic_ru_local_verify.json"
LMF_PREINSTALL_LOCAL=/private/tmp/whisper-v2-semantic-ru-preinstall-local-verify.json

python3 "$LMF_REPO/scripts/verify_whisper_android_frontend_apk.py" \
  --apk "$LMF_APK" --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" --frontend-profile v2 \
  --report "$LMF_PREINSTALL_LOCAL"
cmp "$LMF_PREINSTALL_LOCAL" "$LMF_COMMITTED_LOCAL"
```

Expected APK SHA-256:
`e058489e6946e21a32f28aff458330cddc686f43a61d4f939f5cada5ceaebb9e`.

### Repetition 1

```sh
LMF_RUN_ID=semantic-ru-v2-20260730-e05848-r1
LMF_RUN_DIR="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_ru_v2_run_20260730_e05848_r1"

python3 "$LMF_REPO/scripts/run_whisper_android_frontend_device_probe.py" \
  --serial "$LMF_DEVICE_SERIAL" --adb "$LMF_ADB" \
  --apk "$LMF_APK" --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" --frontend-profile v2 \
  --output-dir "$LMF_RUN_DIR" --run-id "$LMF_RUN_ID" \
  --timeout-seconds 300

python3 "$LMF_REPO/scripts/verify_whisper_semantic_device.py" \
  --semantic-spec "$LMF_SPEC" --base-spec "$LMF_BASE_SPEC" \
  --local-verify "$LMF_RUN_DIR/local_apk_verify.json" \
  --device-report "$LMF_RUN_DIR/device_report_${LMF_RUN_ID}.json" \
  --evidence "$LMF_RUN_DIR/device_evidence_${LMF_RUN_ID}.json" \
  --run-id "$LMF_RUN_ID" \
  --output "$LMF_RUN_DIR/semantic_verify_${LMF_RUN_ID}.json"

"$LMF_ADB" -s "$LMF_DEVICE_SERIAL" shell am force-stop ai.liquid.lfmdemo.whisperprobe
```

### Repetition 2, without rebuilding

```sh
LMF_RUN_ID=semantic-ru-v2-20260730-e05848-r2
LMF_RUN_DIR="$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_ru_v2_run_20260730_e05848_r2"

python3 "$LMF_REPO/scripts/run_whisper_android_frontend_device_probe.py" \
  --serial "$LMF_DEVICE_SERIAL" --adb "$LMF_ADB" \
  --apk "$LMF_APK" --base-spec "$LMF_BASE_SPEC" \
  --frontend-spec "$LMF_SPEC" --frontend-profile v2 \
  --output-dir "$LMF_RUN_DIR" --run-id "$LMF_RUN_ID" \
  --timeout-seconds 300

python3 "$LMF_REPO/scripts/verify_whisper_semantic_device.py" \
  --semantic-spec "$LMF_SPEC" --base-spec "$LMF_BASE_SPEC" \
  --local-verify "$LMF_RUN_DIR/local_apk_verify.json" \
  --device-report "$LMF_RUN_DIR/device_report_${LMF_RUN_ID}.json" \
  --evidence "$LMF_RUN_DIR/device_evidence_${LMF_RUN_ID}.json" \
  --run-id "$LMF_RUN_ID" \
  --output "$LMF_RUN_DIR/semantic_verify_${LMF_RUN_ID}.json"

"$LMF_ADB" -s "$LMF_DEVICE_SERIAL" shell am force-stop ai.liquid.lfmdemo.whisperprobe
```

## Repetition acceptance policy

For each semantic language, both collector device-verifier reports and both
semantic-verifier reports must have `status=pass`. The two raw device reports
must then have identical:

- installed whole-APK SHA-256;
- `semantic_decode.language_code`;
- `semantic_decode.candidate_transcript`;
- `semantic_decode.generated_token_ids`;
- EOT/termination result.

Run these exact host-only comparisons after each pair succeeds:

```sh
python3 -c 'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); sa=a["semantic_decode"]; sb=b["semantic_decode"]; keys=("language_code","candidate_transcript","generated_token_ids","eot_reached","termination_reason"); assert all(sa[k]==sb[k] for k in keys), "ZH semantic repetition mismatch"; print("ZH semantic repetition: pass")' \
  "$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_zh_v2_run_20260730_b33bf5_r1/device_report_semantic-zh-v2-20260730-b33bf5-r1.json" \
  "$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_zh_v2_run_20260730_b33bf5_r2/device_report_semantic-zh-v2-20260730-b33bf5-r2.json"

python3 -c 'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); sa=a["semantic_decode"]; sb=b["semantic_decode"]; keys=("language_code","candidate_transcript","generated_token_ids","eot_reached","termination_reason"); assert all(sa[k]==sb[k] for k in keys), "KO semantic repetition mismatch"; print("KO semantic repetition: pass")' \
  "$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_ko_v2_run_20260730_81b75c_r1/device_report_semantic-ko-v2-20260730-81b75c-r1.json" \
  "$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_ko_v2_run_20260730_81b75c_r2/device_report_semantic-ko-v2-20260730-81b75c-r2.json"

python3 -c 'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); sa=a["semantic_decode"]; sb=b["semantic_decode"]; keys=("language_code","candidate_transcript","generated_token_ids","eot_reached","termination_reason"); assert all(sa[k]==sb[k] for k in keys), "RU semantic repetition mismatch"; print("RU semantic repetition: pass")' \
  "$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_ru_v2_run_20260730_e05848_r1/device_report_semantic-ru-v2-20260730-e05848-r1.json" \
  "$LMF_REPO/reports/android_phone/streaming_response/whisper_v81/device_semantic_ru_v2_run_20260730_e05848_r2/device_report_semantic-ru-v2-20260730-e05848-r2.json"
```

Any failed-closed report, transcript/token mismatch, frontend bound violation,
CPU fallback, missing QNN/HTP/FastRPC/DMA evidence, latency/accuracy gate
failure, protected-package drift, or incomplete evidence stops that language
lane. Do not promote or integrate a failed candidate, and do not describe it as
an accepted NPU multilingual route.
