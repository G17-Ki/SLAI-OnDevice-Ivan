# TTS routing correctness and graph-reuse fixes

- Date: 2026-07-29 23:40 CST
- Scope: source, metadata-contract tests, and compile-only Android verification.
  No ADB command, APK assembly, installation, or phone mutation was performed.

## Fixed behavior

- Russian, Japanese, and Korean Supertonic entries share one resident
  `OfflineTts` graph when engine type, thread count, and model-file map match.
  Language, step count, seed, speaker, speed, and reference inputs remain
  per-request settings. Each queued task snapshots its candidate, so changing
  the selected language cannot rewrite an already queued generation.
- Candidate metadata now rejects duplicate IDs, malformed language codes,
  missing Supertonic assets, non-positive Supertonic step counts, and every
  explicit `extra.lang` that differs from the candidate language. Supertonic
  requires `extra.lang`. Metrics report the validated per-request generation
  language.
- A manually pinned neural candidate is accepted only for its declared
  language. This prevents a Japanese turn from silently using a Korean or
  Russian generation configuration.
- Automatic neural routing is priority ordered and tries each later available
  candidate after a load failure. Once system capability discovery completes,
  it falls back to system TTS only for a cached usable locale. If discovery is
  still in flight, the route may await initialization, but it cannot report
  ready until the authoritative voice check passes.
- Installing `com.reecedunn.espeak` no longer claims support for every locale.
  The bridge warms the pinned engine, caches languages from its actual
  installed, non-network `Voice` records, and uses that cache for the
  synchronous locale gate. Final initialization still verifies
  `isLanguageAvailable`, selects a matching offline voice, and rejects a
  mismatched, not-installed, or network-required active voice.

## Regression coverage

`scripts/test_tts_candidate_registry_contract.py` covers:

- shared RU/JA/KO Supertonic graph identity and non-sharing when model threads
  differ;
- language/config mismatch rejection and per-request metric wiring;
- missing-primary, failed-primary, and failed-secondary automatic fallback;
- package-present/capability-unknown eSpeak rejection;
- unsupported-locale and network-only fallback rejection;
- pinned-candidate locale matching.

`scripts/test_android_native_tts_bridge.py` pins the matching source-level
bridge contracts.

Verification passed:

```text
python3 scripts/test_tts_candidate_registry_contract.py
python3 scripts/test_android_native_tts_bridge.py
python3 scripts/test_tts_device_bundle_metadata.py
python3 -m py_compile scripts/test_tts_candidate_registry_contract.py scripts/test_android_native_tts_bridge.py
ruff check scripts/test_tts_candidate_registry_contract.py scripts/test_android_native_tts_bridge.py
git diff --check -- <focused TTS files>
```

The compile-only Android check also passed under JDK 17 with two workers:

```text
./gradlew :app:compileDebugKotlin :app:compileDebugJavaWithJavac \
  --max-workers=2 --no-daemon
BUILD SUCCESSFUL
```

This proves source compatibility, not phone execution, audible quality, or
on-device latency. A future accepted-runtime build must re-run the physical
multilingual TTS gates before these changes are promoted.
