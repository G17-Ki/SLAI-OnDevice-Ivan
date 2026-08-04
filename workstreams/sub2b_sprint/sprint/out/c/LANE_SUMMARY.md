# Lane C — ASR summary

All 15/15 rows are HOST and same-fixture comparable. Admissible clean leaders: ZH=D-fork / `FunAudioLLM/SenseVoiceSmall` (CER 6.55%); ES=D-fork / `nvidia/parakeet-tdt-0.6b-v3` (WER 3.55%); FR=D-fork / `nvidia/parakeet-tdt-0.6b-v3` (WER 5.34%). EN and RU clean winners are withheld because normaliser sensitivity exceeds 2 pp. OmniASR passed all five smoke gates but exceeded 15% clean error in 4/5 languages (EN, ZH, FR, RU).

Confidence: **HIGH for execution provenance and fixture comparability; MEDIUM for the n=150 point estimates and bootstrap intervals; NO formal ranking confidence for EN and RU because their normaliser deltas exceed 2 pp.**

## Important noisy-bucket provenance caveat

> **MUSAN subset-mirror caveat:** the 10 dB bucket uses `alexwengg/musan_mini@54ce99dd44add93cec39c4d37bdcc59c7870a849`, a Hugging Face subset mirror rather than the canonical OpenSLR-17 distribution. The receipt preserves the upstream README, licences, annotations, and exact noise-file inventory; results must retain this provenance qualifier.

Complete clean+MUSAN HOST rows: 15/15; UNKNOWN rows: 0. Clean HOST cells: 15/15; MUSAN HOST cells: 15/15.

Same-corpus comparability across all three banks: **True**.

Lane B dependency: **MET**. `out/c/normalisers.py` is byte-identical to `sprint/b-backbone:workstreams/sub2b_sprint/sprint/out/b/normalisers.py` at SHA-256 `82f38a8cbf3208e360acf0b6789e8940d345ace76e0bd11e3834a34edda828dc`.

No training, fine-tuning, quantization, export, or local heavyweight inference was performed.
