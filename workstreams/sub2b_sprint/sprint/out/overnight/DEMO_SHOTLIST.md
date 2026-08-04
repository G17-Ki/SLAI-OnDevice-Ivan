# Overnight demo shot list

## Morning go/no-go

Do not improvise around missing evidence.

| Demo | Green-light condition | Current committed checkpoint |
|---|---|---|
| **Demo 2 — “IT'S GOOD”** | Final no-LFM APK installed; complete inventory below 3B; EN and ZH cold filmed runs below the request-to-in-app-audio-start proxy gate. | **FILM COMPLETE:** untrimmed `films/DEMO2_ITS_GOOD.mp4`, 356.360 s, SHA-256 `68740c0d2d521d01564bebd53003bfd32e4bcef0c46b7f4a1ce732c27369871d`; automated media checks passed. |
| **Demo 1 — “IT FITS”** | Final inventory below 2B and EN/ZH/ES/FR/RU routes complete. | **FILM COMPLETE:** untrimmed `films/DEMO1_IT_FITS.mp4`, 5,701.162 s, SHA-256 `cf7ba53b9cf1a387b4c52501161253906e1002284b7cc48bd4907a0e80367958`; automated media checks passed. Final RU answer “Москва” manually confirmed factually correct. |

Both demos are green against their engineering gates, and this list now doubles as a locator for the two untrimmed handoff films. Demo 1 deliberately retains an earlier failed Russian attempt before the successful retry. Demo 2 preserves the full cold EN recording followed by the full cold ZH recording; the ZH section includes a no-audio setup attempt before the first real request. No detailed film-quality or semantic review was performed beyond confirming the final visible Russian answer.

## Preflight frame — use before either demo

Keep the phone and the verification screen in the same shot.

- Show **Nubia NX809J / SM8850** identity.
- Show airplane mode on.
- Show an empty ADB-forward list.
- Show main package and runtime-companion package versions, APK hashes, and their distinct app UIDs.
- Show the complete resident model inventory and computed cap total for the demo being filmed.
- Show that the main process does not contain/load another QNN provider.

One-sentence claim: **“This is an offline, app-owned run on the Nubia, with the shipped inventory counted before inference.”**

Do not say “NPU stack.” If the recorded evidence matches the established route, the narrower line is: **“SenseVoice ASR is executing on QNN/HTP with a version-matched 2.40 context and runtime in the companion process.”**

## Demo 2 — “IT'S GOOD”

Film complete: `films/DEMO2_ITS_GOOD.mp4` is the untrimmed 356.360 s EN→ZH handoff. Automated full-decode, audio-stream, volume, size, and hash checks passed; the user will perform final viewing approval. A later editorial cut may use shots 1–4 as locators.

Keep the one-time offline Qwen warmup in frame. Input remains disabled during startup warmup, and selecting Mandarin may trigger a second language-specific warmup. The `<3,000 ms` gate is request-to-`UtteranceProgressListener.onStart` after input becomes available; it is not boot-to-audio. Do not edit the warmup out while retaining the latency claim.

### Shot 1 — title and inventory

- On-screen title: `DEMO 2 — IT'S GOOD / EN + ZH / OFFLINE / <3B TARGET`.
- Show `inventory_receipt.json`: 42 rows, 1,974,795,009 learned / 2,208,452,225 conservative, 791,547,775 conservative headroom.
- Show the selected routes: SenseVoice, Qwen2.5-1.5B, Supertonic EN, Matcha ZH.

One-sentence claim: **“This EN+ZH stack is designed for quality at usable latency; the displayed resident inventory is the basis of the cap claim.”**

### Shot 2 — cold English

- Start after the cold reboot preflight.
- Select **English** and **Offline voice**; memory off.
- Keep the startup warmup visible and wait until input is enabled before starting the request.
- Speak or play the exact fixture line:

> What is the capital of France? Answer in one short sentence.

- Keep the screen visible through ASR text, response text, first audio, and playback completion.
- Acceptance: transcript contains “capital of France”; response contains “Paris”; Latin script; finite non-silent audio; zero underflows/dropped chunks; first-audio software proxy `<3,000 ms`.

One-sentence claim: **“The cold offline English turn completed on the phone, with first audio starting inside the three-second software-proxy gate.”**

### Shot 3 — cold Mandarin

- Reboot again so Mandarin has its own cold acceptance. Repeat the airplane-mode, empty-forward, pre-launch PID, package/hash, and main-app-only launch preflight.
- Without enabling networking, select **中文** and **Offline voice**; memory off.
- Keep any selection-triggered Mandarin warmup visible and wait until input is enabled before starting the request.
- Speak or play the exact fixture line:

> 请用中文简单介绍一下北京。

- Keep Latin acronyms and digits out of any manual prompt or forced response.
- Acceptance: transcript and response include “北京”; response is Han script; finite non-silent audio; zero underflows/dropped chunks; first-audio software proxy `<3,000 ms`.

One-sentence claim: **“The cold offline Mandarin turn completed on the phone, with first audio starting inside the three-second software-proxy gate.”**

### Shot 4 — historical repeatability insert (older LFM-bearing APK; not the final build)

Show the committed warm distribution card with a prominent caption: `HISTORICAL OLDER LFM-BEARING APK — NOT FINAL BUILD`. The final cap-valid APK has one accepted cold run per language, not this distribution.

| Language | Completed n | Median | P95 | Max |
|---|---:|---:|---:|---:|
| EN | 21 | 1,333.866 ms | 1,395.874 ms | 1,490.414 ms |
| ZH | 20 | 1,031.277 ms | 1,221.034 ms | 1,603.032 ms |

Add a small caption: `IN-APP AUDIO-START CALLBACK; NOT ACOUSTIC ONSET`. Also disclose one excluded EN and two excluded ZH operator-overlap cancellations.

One-sentence claim: **“Across completed warm turns, both languages stayed below the three-second software-proxy gate, with no playback underflow or dropped audio chunks.”**

## Demo 1 — “IT FITS”

Film complete: `films/DEMO1_IT_FITS.mp4` is the untrimmed 5,701.162 s archive. Automated decode/audio/hash checks passed. The only requested manual semantic check confirmed the final Russian answer **“Москва”** is factually correct; the user will review the rest. The full film retains the earlier failed Russian attempt and the successful retry.

Engineering note, not filming acceptance: the CPU Parakeet verifier already reproduced the exact expected ES and FR transcripts under the runtime service UID. Its 5,828.660/5,526.684 ms figures are ASR-only host round trips through a temporary diagnostic forward, so do not put them on screen as app end-to-end latency.

The accepted English checkpoint is committed at `cb8b306`. Its result frame visibly shows “Paris is the capital of France.” and Supertonic metrics (first chunk 1,027 ms, RTF 0.441, 2.5 s audio). The Russian coverage checkpoint was accepted at `794871f` and its per-chunk metrics corrected at `471db9c`. The accepted Mandarin checkpoint is committed at `a001915`; its response is Han-only and visibly includes “北京”. The accepted Spanish checkpoint is committed at `1948535`; its result visibly says “La capital de España es Madrid.” Frame results tightly around the selected route and answer: this accepted APK still carries stale static copy—`English response & advanced` above multilingual results and `EN / 中文 offline demo` in the bottom footer. Neither string is scope evidence. Do not rebuild solely to change the copy, because a new APK would require fresh acceptance.

### Shot 5 — title and strict cap

- On-screen title: `DEMO 1 — IT FITS / 5 LANGUAGES / OFFLINE / STRICTLY <2B`.
- Show `evidence/demo1/final_inventory/inventory_receipt.json`: 61 rows, 1,810,531,015 learned / 1,820,982,302 conservative, 179,017,698 conservative headroom.
- Show the exact Qwen3.5 hash and the absence of Qwen2.5, LFM, and stale TTS/rollback trees.

One-sentence claim: **“The complete displayed five-language resident inventory is strictly below two billion parameters.”**

### Shot 6 — English

> What is the capital of France? Answer in one short sentence.

Acceptance: transcript contains “capital of France”; response contains “Paris”; Latin script; finite non-silent playback.

Accepted engineering reference: first-audio proxy 2,339.209 ms; playback terminal 4,840.146 ms; zero underflows/dropped chunks; no user-visible reasoning leakage. Do not narrate that Qwen emitted no reasoning markers—one 19-character raw block was filtered before display.

One-sentence claim: **“The same sub-2B offline inventory completes an English voice turn.”**

### Shot 7 — Mandarin

> 请用中文简单介绍一下北京。

Acceptance: transcript/response include “北京”; Han script; finite non-silent playback; no Latin acronyms or digits in the prompt.

Accepted engineering reference: first-audio proxy 1,549.215 ms; playback terminal 11,291.198 ms; two Matcha chunks; zero underflows/drops; visible response contains no Latin letters or digits. Crop the stale `EN / 中文 offline demo` footer.

One-sentence claim: **“The same sub-2B offline inventory completes a Mandarin voice turn.”**

### Shot 8 — Spanish

> Dime brevemente cuál es la capital de España.

Acceptance: ASR transcript includes “España”; response includes “Madrid”; Latin script; finite non-silent playback. Keep the full wait visible and show the measured full-turn time.

Accepted engineering reference: first-audio proxy 3,343.357 ms; playback terminal 6,104.679 ms; one Supertonic chunk; zero underflows/drops; CPU-only. Crop the stale English-only labels.

One-sentence claim: **“The same sub-2B offline inventory completes a Spanish CPU voice turn; the visible delay is part of the result.”**

### Shot 9 — French

> Dis-moi brièvement quelle est la capitale de la France.

Acceptance: ASR transcript includes “capitale” and “France”; response includes “Paris”; Latin script; finite non-silent playback. Keep the full wait visible and show the measured full-turn time.

Accepted engineering reference: first-audio proxy 3,223.415 ms; playback terminal 6,022.732 ms; one Supertonic chunk; zero underflows/drops; CPU-only. Crop the stale English-only labels.

One-sentence claim: **“The same sub-2B offline inventory completes a French CPU voice turn; the visible delay is part of the result.”**

### Shot 10 — Russian

> Какова столица России?

Final fixture: `evidence/demo1/fixtures/question_ru_simple.wav`, SHA-256 `e8d931545ebddf124d63acd5d42bbde10fab7ba2f3f65998ce2ab7237693bc45`. Acceptance: transcript asks for Russia's capital; response includes “Москва”; Cyrillic script; finite non-silent playback; no false superlative.

The 4,437.486 ms / 15,605.082 ms figures belong only to the older historical RU fixture and must not be reused for this final prompt. In the untrimmed film, the earlier failed RU attempt remains visible before the retry; the final visible answer is **“Москва”**, manually confirmed factually correct. No latency is claimed for the new fixture.

One-sentence claim: **“The same sub-2B offline inventory completes a Russian CPU voice turn.”**

## Failure shots — include even if both demos pass

These are host evaluation findings, not phone-demo failures. Label them `HOST DIAGNOSTIC`.

### Shot 11 — Matcha silently drops Latin spans

Display the exact requested text:

> API返回了一个包含24条记录的JSON文件。

Show the captured Matcha frontend evidence containing `Ignore OOV` for the Latin material, and play the retained diagnostic WAV only if the original Lane D listening bundle is present and verified.

One-sentence claim: **“Matcha can emit finite Mandarin audio while silently omitting unsupported Latin spans, so finite PCM and a CER score do not prove mixed-script coverage.”**

Do not claim the digits were necessarily omitted; the captured defect is the Latin/OOV spans including `API` and `JSON`.

### Shot 12 — Kokoro Mandarin non-finite PCM

Display the short input:

> 你好

Show the Lane D host record: Kokoro v1.1 ZH failed this prompt 3/3 with all samples non-finite. Do not fabricate or play a WAV; silence/no serialized clip is the caught failure.

One-sentence claim: **“The evaluation rejected this Kokoro Mandarin route because a simple prompt produced non-finite PCM three times out of three.”**

## Closing truth card

Use this exact ending:

**“Two offline phone configurations passed their stated gates: Demo 2 for EN+ZH inventory, execution, and request-to-in-app-audio-start proxy, and Demo 1 for five-language coverage under a strict 2B inventory.”**

Permanent footer for every ending: `NO NATIVE-SPEAKER ACCEPTANCE · NO ACOUSTIC-ONSET RIG · NO GLASSES EVIDENCE`.
