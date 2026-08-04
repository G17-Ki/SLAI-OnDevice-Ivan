# Offline Android TTS audit for Mandarin and Russian

Date: 2026-07-24
Scope: source/package audit only; no ADB commands, installs, or phone changes were made.

## Decision

The already installed F-Droid eSpeak engine is sufficient to begin the
Mandarin and Russian phone acceptance run. It contains both language data
sets, exposes both through Android `TextToSpeech`, and marks its voices as not
requiring a network connection. No additional TTS package is required before
testing.

There is one locale nuance: eSpeak 1.52.0 exposes language-only Android voices
`zh` and `ru`, not country-specific voices named `zh-CN` and `ru-RU`.
Requests for `zh-CN` and `ru-RU` therefore resolve as
`TextToSpeech.LANG_AVAILABLE`, rather than
`LANG_COUNTRY_AVAILABLE`. This is compatible with the current LFM Demo bridge:
it accepts language-level availability, chooses an offline voice with the same
language, and verifies the active voice language. Telemetry should consequently
show the resolved locale as `zh` or `ru` even when the requested tag was
`zh-CN` or `ru-RU`.

eSpeak is a compact formant synthesizer. It is a good offline functional
baseline and should be intelligible, but it is expected to sound robotic.
Mandarin intelligibility still needs a physical synthesize-to-file/back-ASR
check; package contents alone are not a quality result.

RHVoice is a worthwhile optional Russian A/B candidate, but not a Mandarin
solution and not an immediate replacement. The current app creates one
system-default `TextToSpeech` instance, so merely installing RHVoice would not
make the app use RHVoice for Russian while retaining eSpeak for Mandarin.
Explicit per-language engine routing would be required first.

## eSpeak evidence

| Item | Fact |
|---|---|
| Android package | `com.reecedunn.espeak` |
| Audited release | `1.52.0` (`versionCode 22`) |
| Distribution | F-Droid |
| License reported by F-Droid | `GPL-3.0-or-later` |
| APK SHA-256 | `0a7822fec54d7f7ae759ffedc9afeaa77f809587b171fb2101998787821ec790` |
| Mandarin data in exact APK | `espeak-ng-data/cmn_dict` and `lang/sit/cmn` |
| Russian data in exact APK | `espeak-ng-data/ru_dict` and `lang/zle/ru` |
| Network required for synthesis | No; Android voices are constructed with `networkConnectionRequired=false` |

The F-Droid APK hash above matches the package recorded as installed in the
project checkpoint. Inspection of the exact APK's embedded 9.2 MiB eSpeak data
archive found both compiled dictionaries and both language definitions.

At the pinned 1.52.0 source commit, the Mandarin definition declares `cmn`,
`zh-cmn`, and `zh`; Android explicitly maps `cmn` to `Locale("zh")`. The
Russian definition declares `ru`. The engine's locale matcher falls back to a
language-level match when the requested country differs or the voice has no
country, which is why `zh-CN` and `ru-RU` work but resolve to `zh` and `ru`.

This also matches the current app implementation:

- UI locale requests are `zh-CN` and `ru-RU`;
- `NativeAudioBridge` accepts `LANG_AVAILABLE`;
- preferred-voice selection requires an equal language code, not an exact
  country match;
- offline voices receive the strongest preference;
- the final guard requires the active voice language to match the request.

## RHVoice assessment

| Item | Fact |
|---|---|
| Android package | `com.github.olga_yakovleva.rhvoice.android` |
| Current F-Droid release audited | `1.18.4` (`versionCode 118040`) |
| Engine license reported by F-Droid | `GPL-3.0-or-later` |
| Mandarin support | None |
| Russian support | Yes; multiple `ru-RU` voices |
| Offline after voice download | Yes; installed voices are advertised with `networkConnectionRequired=false` |
| F-Droid flags | `NonFreeAdd`, `NonFreeNet` |

RHVoice uses small statistical models derived from natural recordings and is
likely to be more pleasant and natural for Russian than eSpeak. Its Android
catalog currently lists Russian language data plus Aleksandr, Aleksandr-hq,
Anna, Arina, Artemiy, Elena, Evgeniy-Rus, Irina, Mikhail, Pavel, Tatiana,
Victoria, Vitaliy, and Yuriy voices.

The 15 MiB engine APK does not by itself close the offline requirement. The
selected language and voice data must first be downloaded in RHVoice, after
which synthesis is local. Voice licensing varies. RHVoice's own license file
states that RHVoice Lab voices are `CC BY-NC-ND 4.0`; therefore voice data must
not be bundled into this project's public APK/repository without selecting a
specific voice, auditing that voice's license, and confirming redistribution
rights. The engine APK itself is GPL-3.0-or-later according to F-Droid.

Recommendation: keep eSpeak as the immediate EN/ZH/RU baseline. Add RHVoice
only as a controlled Russian A/B after eSpeak's Russian backcheck, and install
voice data separately rather than committing it to the repository.

If RHVoice is adopted, the smallest safe production change is:

1. map `zh` to explicit engine package `com.reecedunn.espeak`;
2. map `ru` to `com.github.olga_yakovleva.rhvoice.android` only when a local
   Russian voice is installed, otherwise fall back to eSpeak;
3. construct `TextToSpeech(context, listener, enginePackage)` for the selected
   engine and reinitialize only when the engine package changes;
4. include engine package, resolved voice name, resolved locale, and
   `isNetworkConnectionRequired()` in ready/error telemetry.

Without that change, the current single `TextToSpeech(context, listener)`
instance always follows Android's system-default engine.

## Mandarin quality upgrade path

No trusted F-Droid Android TTS engine found in this audit offers a clearly
better offline Mandarin path than eSpeak. The project's stronger neural
candidate remains MeloTTS-ZH, not another installable Android TTS engine.

Local evidence already shows that the public MeloTTS-ZH QAIRT 2.45 contexts
profile on exact SM8850 with 2,372/2,372 operations on HTP v81, zero fallback,
and a 144.839 ms four-component sum. That result does **not** yet prove a full
Mandarin text-to-wave chain, numerical equivalence, Nubia execution, or
intelligibility. MeloTTS source and the public `myshell-ai/MeloTTS-Chinese`
checkpoint report the MIT license. It is the right medium-term Mandarin
quality experiment, while eSpeak is the right immediate APK baseline.

## Proposed automated synthesize-to-file/back-ASR gate

Prefer an instrumentation/host test over adding a production WebView API.
Suggested reproducible artifacts:

- `android-app/app/src/androidTest/.../NativeTtsRoundTripTest.java`;
- `scripts/test_android_tts_roundtrip.py`;
- a checked-in JSON fixture list with short ZH and RU sentences;
- per-run WAV, ASR JSON, metrics JSON, and a manifest containing APK version,
  phone build, engine package/version, requested and resolved locale, voice
  name, audio hash, and timings.

The Android test should:

1. instantiate an explicit engine package rather than depend on the phone's
   default engine;
2. wait for `onInit`, check `isLanguageAvailable`, call `setLanguage`, select
   a local matching voice, and record the resolved locale/voice;
3. call `TextToSpeech.synthesizeToFile` into the app's external-files test
   directory and wait for `UtteranceProgressListener.onDone`;
4. fail on init, missing data, unsupported locale, network-required voice,
   zero-byte output, timeout, or TTS callback error;
5. use phrases comfortably below ten seconds so the fixed-shape SenseVoice
   Mandarin ASR input remains valid.

The host runner should pull each WAV and submit it directly, without speaker
or microphone loopback, to:

- SenseVoice QNN with explicit `zh` for Mandarin;
- Russian Zipformer with explicit `ru` for Russian.

Normalize Unicode and punctuation before scoring. Use character error rate for
Mandarin and both word and character error rate for Russian. Also check:

- duration, RMS, peak level, clipping fraction, and long internal silence;
- non-empty ASR output and expected-script ratio;
- repeated-character/token and repeated n-gram collapse;
- synthesis time and real-time factor.

Provisional gates, to be calibrated from the first physical batch:

- hard failure: empty/invalid WAV, empty transcript, wrong dominant script,
  repeated-output collapse, or any fixture above 40% Mandarin CER / 50%
  Russian WER;
- quality target: Mandarin micro-CER at most 20%; Russian micro-WER at most
  30% and micro-CER at most 20%;
- every language still requires a brief native-speaker listening A/B because
  back-ASR measures intelligibility, not naturalness or correct prosody.

Suggested fixtures:

```text
ZH: 你好，请告诉我今天北京的天气。
ZH: 我们下午三点开会，房间号是二零五。
ZH: 这副眼镜支持连续对话。

RU: Привет, расскажи, какая сегодня погода.
RU: Встреча начнётся в три часа, кабинет двести пять.
RU: Эти очки поддерживают непрерывный диалог.
```

## Public sources

- eSpeak F-Droid package: <https://f-droid.org/packages/com.reecedunn.espeak/>
- eSpeak F-Droid build metadata: <https://gitlab.com/fdroid/fdroiddata/-/blob/master/metadata/com.reecedunn.espeak.yml>
- eSpeak pinned Android locale mapping: <https://github.com/espeak-ng/espeak-ng/blob/4870adfa25b1a32b4361592f1be8a40337c58d6c/android/src/com/reecedunn/espeak/SpeechSynthesis.java>
- eSpeak pinned Android voice exposure: <https://github.com/espeak-ng/espeak-ng/blob/4870adfa25b1a32b4361592f1be8a40337c58d6c/android/src/com/reecedunn/espeak/TtsService.java>
- eSpeak pinned Mandarin definition: <https://github.com/espeak-ng/espeak-ng/blob/4870adfa25b1a32b4361592f1be8a40337c58d6c/espeak-ng-data/lang/sit/cmn>
- eSpeak pinned Russian definition: <https://github.com/espeak-ng/espeak-ng/blob/4870adfa25b1a32b4361592f1be8a40337c58d6c/espeak-ng-data/lang/zle/ru>
- RHVoice F-Droid package: <https://f-droid.org/packages/com.github.olga_yakovleva.rhvoice.android/>
- RHVoice F-Droid build metadata: <https://gitlab.com/fdroid/fdroiddata/-/blob/master/metadata/com.github.olga_yakovleva.rhvoice.android.yml>
- RHVoice source and license: <https://github.com/RHVoice/RHVoice/blob/fa9dd196fd2dac3b0bf089a2d80fc8477c2380e3/doc/en/License.md>
- RHVoice Android language/voice catalog: <https://github.com/RHVoice/RHVoice/blob/fa9dd196fd2dac3b0bf089a2d80fc8477c2380e3/src/android/RHVoice-core/src/main/res/raw/packages.json>
- Android `TextToSpeech.synthesizeToFile`: <https://developer.android.com/reference/android/speech/tts/TextToSpeech#synthesizeToFile(java.lang.CharSequence,android.os.Bundle,java.io.File,java.lang.String)>
- MeloTTS license: <https://github.com/myshell-ai/MeloTTS/blob/main/LICENSE>
- Public Chinese checkpoint metadata: <https://huggingface.co/myshell-ai/MeloTTS-Chinese>
