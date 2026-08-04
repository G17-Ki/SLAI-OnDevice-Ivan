# Isolated physical TTS acceptance-probe plan

- Date: 2026-07-30 (Asia/Shanghai)
- Status: design complete; no APK was built or installed and no ADB command was
  run while preparing this plan.

## Decision

Validate the current TTS candidate source in a new, coexistable Android package:

`ai.liquid.lfmdemo.ttsprobe`

Do not replace, reinstall, force-stop, share a UID with, or reuse private data
from either accepted package:

- `ai.liquid.lfmdemo`:
  `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`;
- `ai.liquid.lfmdemo.runtime`:
  `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`.

The older `ai.liquid.ttsbench` app is safely isolated from the accepted demo,
but it is not the right implementation-under-test. Its `CandidateRegistry`,
`TtsRunner`, and `StreamPlayer` predate the current graph-reuse, per-request
candidate snapshot, ordered queue, cancellation, playback-drain, ranked
fallback, and offline-locale logic. Updating it would also mutate an existing
benchmark package and its 1.5-GiB data set. A fresh package gives a smaller,
auditable blast radius and requires only the shared 138.6-MiB Supertonic graph.

## Evidence boundary

The current accepted APKs were built before the latest routing-correctness
changes. A passing probe would validate the exact source hashes listed below;
it would not retroactively prove that those changes are present in either
accepted APK. Promotion requires a later, separately authorized candidate build
and a complete rerun of the accepted-package gates.

The CPU/Sherpa probe also cannot make an NPU claim. The production
`SherpaTtsEngine` sets `provider = "cpu"`. NPU TTS is a separate package/runtime
lane described near the end of this document.

## Exact production code to compile

Compiling only the four files under `lfmdemo/tts` is insufficient for the full
acceptance matrix: automatic secondary-candidate fallback and authoritative
eSpeak voice/locale checks live in `NativeAudioBridge.java`. The isolated probe
should therefore compile these six production files byte-for-byte:

| Source | SHA-256 at plan time |
|---|---|
| `android-app/app/src/main/java/ai/liquid/lfmdemo/tts/CandidateRegistry.kt` | `718c67bc867ff7fcab27d0b542f134aa1669694ebabdcacec1f8e922682e38c6` |
| `android-app/app/src/main/java/ai/liquid/lfmdemo/tts/SherpaTtsEngine.kt` | `491f1418a5039699cb7fb4055ada84706bb31daf38eb4fcf5ccf897aaeebf072` |
| `android-app/app/src/main/java/ai/liquid/lfmdemo/tts/PcmGate.kt` | `6ebc110541643d8609657a3e5f8c6bc55856bb45385891ae568d67537c891435` |
| `android-app/app/src/main/java/ai/liquid/lfmdemo/tts/WavIo.kt` | `5aa91ed6338a862da957093d96aa6b8d4471846b08f9ddbd8f6b0212fe9cb38f` |
| `android-app/app/src/main/java/ai/liquid/lfmdemo/NativeAudioBridge.java` | `bb9c6fa88a120ada4c1a611eff5e500d7d5e202c9d631f181d3b1c53618d59e3` |
| `android-app/app/src/main/java/ai/liquid/lfmdemo/AdaptiveEnergyVad.java` | `f43f07819d1bf8983809421bc6e0b436046fd7e7a96d2271a9209caf3db4da00` |

`AdaptiveEnergyVad.java` is included only because the exact bridge references
it. The probe manifest deliberately withholds microphone permission, so its
capture path cannot run.

The external Sherpa AAR must also be pinned:

- file: `sherpa-onnx-1.13.4.aar`;
- bytes: 48,847,529; both size and hash must be checked before every build;
- SHA-256:
  `03f9c4df965f21c71269365a7951a7f23b5696fddd093fa318c80d65550ab780`.

The module must not keep editable copies of these six files. A Gradle task
should verify the original hashes, copy only those exact files into a generated
source directory under `build/`, and compile that directory. `preBuild` must
depend on both verification and staging. The produced report must repeat the
original and staged hashes. This keeps the probe synchronized with the current
production candidate while preventing the entire main-app source tree from
being pulled into the probe accidentally.

## Proposed module and Android boundary

Add a new `:tts-probe` module under `android-app/tts-probe` with these rules:

- application ID: `ai.liquid.lfmdemo.ttsprobe`;
- arm64-v8a only, min SDK 28, target/compile SDK 35, Java/Kotlin 17;
- no `sharedUserId` and no shared signing or signature permission requirement;
- no `INTERNET`, `RECORD_AUDIO`, storage, foreground-service, notification, or
  accepted-runtime permission;
- no service, provider, or receiver;
- one launcher Activity is the only exported component;
- `allowBackup=false` and no cleartext-network opt-in;
- `<queries>` may name `com.reecedunn.espeak` and the Android TTS service intent;
  package visibility is not a permission;
- use `FLAG_KEEP_SCREEN_ON` for bounded runs. Add `WAKE_LOCK` only in a separate
  unattended-soak flavor if the bounded screen-on path proves insufficient;
- use a platform `Activity` and a local `file:///android_asset/` WebView so the
  probe does not need AndroidX or network access.

The probe Activity should live in Java package `ai.liquid.lfmdemo` so it can
construct the package-private production bridge. Its Android application ID
and Linux UID remain the distinct probe identity.

## Storage boundary and minimal payload

The probe owns only these paths:

- models and candidate metadata:
  `/sdcard/Android/data/ai.liquid.lfmdemo.ttsprobe/files/tts-eval/`;
- atomic reports:
  `/data/user/0/ai.liquid.lfmdemo.ttsprobe/files/tts_probe/reports/`;
- optional probe-generated listening artifacts:
  `/sdcard/Android/data/ai.liquid.lfmdemo.ttsprobe/files/tts_probe/audio/`.

It must not read from either accepted app's data directory or from
`/sdcard/Android/data/ai.liquid.ttsbench/`. Copying the minimal graph into the
new package costs 145,295,768 bytes rather than copying the full 1.5-GiB bench
bundle.

RU, JA, and KO use the same seven authenticated Supertonic files:

| File | Bytes | SHA-256 |
|---|---:|---|
| `duration_predictor.int8.onnx` | 3,700,147 | `c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db` |
| `text_encoder.int8.onnx` | 36,416,150 | `c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff` |
| `vector_estimator.int8.onnx` | 78,400,833 | `20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd` |
| `vocoder.int8.onnx` | 25,991,073 | `e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152` |
| `tts.json` | 8,253 | `42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09` |
| `unicode_indexer.bin` | 262,144 | `8402ca48e5189a8950138580b0fff64db6f072f24ac07cd54ba8b2fbb9883b30` |
| `voice.bin` | 517,168 | `67d5209b0ee8ce6c74105ffbe12fe6a7628aea3b4ba2fcb308a4a67938a93ce8` |

The authoritative source for these records is
`evidence/device_bundle_manifest.json`. The installer must hash every file,
write candidate metadata last, use stable partial files plus atomic rename,
and refuse unknown extra files inside the graph directory.

## Probe runner architecture

The local WebView should expose the exact production bridge as `NativeAudio`.
A small probe-only `ProbeControl` interface should do only the following:

1. run a fixed, versioned case list and capture every native TTS callback;
2. atomically write the report to app-private storage;
3. sample `Debug.MemoryInfo`, `/proc/self/statm`, and process/thread state;
4. inspect the exact bridge/engine through reflection without changing it;
5. expose an explicit stop button and always release the bridge/engine in
   `Activity.onDestroy()`.

Reflection is useful here because it avoids adding instrumentation to the
production sources. At each ready/start/done boundary, record:

- `System.identityHashCode()` for the bridge's `sherpaTts` instance;
- identity of the `SherpaTtsEngine` private `OfflineTts` object;
- the engine's current candidate ID;
- the active `AudioTrack` identity, playback state, playback-head position,
  buffer frames, and underrun count.

A probe-owned watcher should poll the playback head from before queueing until
it first becomes nonzero. That gives a real playback-head onset timestamp,
while the production `utterance_start` callback remains labeled as a software
proxy. The watcher must also confirm that the head stops advancing after
cancellation.

## Immutable run profiles

Use two authenticated `candidates.json` profiles. Their exact byte counts and
hashes must be generated and pinned during implementation.

### Profile A: graph reuse and queue safety

Contains only valid RU, JA, and KO five-step candidates. They must have the
same engine, thread count, and file map, while carrying distinct
`language`/`extra.lang` values.

Run a direct-core queue case against one `SherpaTtsEngine`:

1. load RU once;
2. queue a long RU chunk and a second RU chunk;
3. select the JA candidate before the second RU task begins;
4. queue JA, then select and queue KO;
5. require generation-language metrics in exact order `ru, ru, ja, ko`;
6. require one unchanged `OfflineTts` identity for the entire sequence;
7. require FIFO starts/completions and speaker drain before terminal success.

This is the physical proof that queued requests snapshot candidate settings and
that RU/JA/KO reuse one resident graph rather than reloading it.

Then run bridge-level single- and multi-chunk cases, immediate cancellation,
and cancellation 500 ms after playback-head movement.

### Profile B: routing and fallback

Contains a higher-priority primary and a lower-priority valid secondary for one
locale, plus the ordinary valid candidates. The primary initially points at a
probe-owned duplicate of an authenticated small model-support file.

Do not feed corrupt ONNX to native code. To create a deterministic, safe
`load_failure` after availability selection:

1. use reflection to occupy the bridge's single model executor with a latch;
2. initialize automatic TTS while every primary file exists, so the exact
   production selector chooses the primary;
3. post a probe action behind that UI selection which atomically renames the
   primary-only file;
4. release the model-executor latch;
5. require `SherpaTtsEngine.load()` to fail its Java/Kotlin missing-file check;
6. require a `route_fallback` event with reason `load_failure`, followed by the
   exact secondary candidate and successful playback;
7. restore and re-hash the renamed file during cleanup.

This exercises the production load-failure branch without malformed native
model data or a memory-pressure fault.

For eSpeak, first record the bridge's warmed
`systemTtsOfflineLanguages` set and the complete installed `Voice` metadata.
Choose the supported test locale dynamically from an actual installed,
non-network voice that has no neural candidate. Choose the unsupported locale
from a fixed sentinel list only after confirming its language is absent from
that set. Require:

- package-present/capability-unknown never reports locale support;
- the supported locale resolves to `system`, activates a matching offline
  voice, and completes;
- the unsupported locale reports no available engine and never emits ready or
  playback callbacks;
- a manually pinned JA neural voice rejects a KO locale before synthesis.

## Required physical gates

### 1. Local pre-install gate

- all six production-source hashes and the Sherpa AAR hash match;
- focused TTS source/metadata tests pass;
- the probe module compiles with JDK 17 and at most two Gradle workers;
- APK verifier proves the exact application ID, arm64 ABI, signer, alignment,
  dependency/source manifest, and absence of forbidden permissions/components;
- both candidate-profile manifests and every model asset authenticate;
- no accepted APK is assembled as a side effect.

### 2. Device isolation gate

Before and after every install/run cycle, record the accepted APK hashes. After
probe installation require:

- three distinct package names, UIDs, internal data directories, and APK paths;
- probe self-reported APK hash equals the host-verified installed APK hash;
- no shared UID, accepted signature permission, provider authority, service,
  socket, or data-directory access;
- accepted package versions and hashes remain unchanged.

### 3. Reuse/language gate

- exactly one valid graph identity across RU -> JA -> KO;
- exact per-request `generation_language` and candidate IDs;
- queued-task snapshot order `ru, ru, ja, ko`;
- no graph reload or transient second valid graph;
- first chunk no more than 1,200 ms, RTF no more than 0.8, and zero dropped PCM
  chunks for each ordinary five-step utterance.

### 4. Playback and cancellation gate

- three chunks queue, start, and complete in order;
- terminal completion occurs only after the final playback fence drains;
- playback-head onset is observed and timestamped;
- zero unexpected underruns after the initial stream-start artifact;
- immediate cancellation and post-onset cancellation produce exactly one
  cancelled terminal, no late start/done/metrics callback, and no continuing
  playback-head advance after the bounded stop window.

### 5. Routing/fallback gate

- controlled primary `load_failure` selects and completes on the ranked
  secondary candidate;
- actual supported offline eSpeak locale completes on a matching non-network
  voice;
- an absent locale remains unsupported despite package installation;
- a pinned neural candidate cannot cross its declared locale;
- every fallback event includes source candidate, destination, reason, turn,
  and generation.

### 6. PCM/resource gate

- every played chunk passes the exact production `PcmGate`;
- no non-finite, silent, clipped, or severe-DC chunk reaches `AudioTrack`;
- report first chunk, synthesis time, RTF, playback-head onset, drain time,
  zero-write retries, underruns, dropped chunks, peak PSS, and peak RSS;
- use a provisional hard ceiling of 900 MiB PSS and 1,024 MiB RSS for the
  standalone probe, and reject any evidence of simultaneous valid graph
  residency;
- no crash, LMK, freezer event, or severe thermal status.

### 7. Offline/restart gate

- airplane mode on, Wi-Fi off, and empty IP route before the run;
- repeat both profiles after force-stop/relaunch;
- repeat the final reuse and fallback verdicts after a cold reboot and first
  manual probe launch;
- retrieve the app-private report with `run-as` and verify its hash;
- repeat one final run with the USB cable physically removed. A host ADB-server
  shutdown is useful but is not a substitute for literal removal.

### 8. Post-run integrity gate

- force-stop only the probe;
- accepted APK hashes still match the two values at the top of this report;
- accepted app/runtime health and the established five-language stack remain
  intact after the probe is stopped;
- retain the probe until evidence is archived. Uninstalling it is optional and
  must remove only its own package/data.

## App-private report contract

`files/tts_probe/reports/latest.json` should be written atomically and contain:

- schema/run/profile IDs, wall and monotonic times, boot ID, device/build/ABI;
- self package, UID, data paths, APK hash, signer, requested/granted permissions;
- exact source, AAR, profile, and asset manifests;
- system-TTS package and complete offline voice inventory;
- every native callback with host-independent monotonic timestamps;
- graph/engine/AudioTrack identity observations;
- playback-head, underrun, PSS/RSS, and thermal samples;
- per-case assertions and a fail-closed overall verdict;
- cleanup status and any late callback observed during the grace window.

The report is authoritative only when joined with the external pre/post
accepted-APK hashes and installed-package/UID evidence.

## NPU TTS lane: deliberately separate

The accepted app-owned QNN runtime is QAIRT
`2.42.0.251225135753_193295`. Current Supertonic, MeloTTS-ZH, and Piper Voice AI
contexts were produced with QAIRT 2.45. They must not be loaded into the
accepted 2.42 process without a dedicated compatibility proof or recompilation.

The safe next NPU experiment is another isolated package/process, for example
`ai.liquid.lfmdemo.ttsprobe.qairt245`, carrying its own pinned QAIRT 2.45
runtime. It must prove app-UID HTP/CDSP execution, zero CPU fallback, output
quality, repeated scheduling, and device compatibility before any integration.
The standard local cache currently exposes the Piper-English 2.45 contexts
under `~/.cache/lmf-demo/piper-en-v0.58.0/extracted/`; the standard cache scan
did not locate the Supertonic or Melo context binaries, so those should be
materialized from the preserved AI Hub jobs or regenerated remotely. No heavy
conversion or compilation belongs on this MacBook.

Candidate order for that lane:

1. MeloTTS-ZH, because its four pinned components have strict all-NPU placement
   and a complete release manifest;
2. Supertonic for RU/JA/KO, after broader audio-quality review of its accumulated
   numerical drift;
3. Piper EN/IT as the strongest Qualcomm-native fallback where language and
   licensing fit.

Physical NPU acceptance additionally requires the probe UID to own loaded QNN
HTP/system/v81 libraries, an open CDSP FastRPC descriptor and QCOM DMA buffers,
plus exact context/runtime hashes and no fallback. AI Hub QRD profile rows alone
are not Nubia execution evidence.

## Effort and promotion rule

- Exact CPU/Sherpa coexistable probe: approximately 0.5-1 engineering day.
- QAIRT-2.42 recompilation or isolated QAIRT-2.45 TTS chain plus Nubia
  integration: several days, with Qualcomm AI Hub/Colab doing the heavy work.

No current source should be promoted into either accepted package until every
probe gate above passes and the complete unplugged five-language end-to-end
matrix is repeated on the candidate APK pair.
