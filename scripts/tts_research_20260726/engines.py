#!/usr/bin/env python3
"""Candidate TTS engine adapters for the 2026-07-26 natural-voice research run.

Every candidate is exposed through one interface so latency, waveform gates,
intelligibility, and footprint are measured identically. Model files live
outside the repository; pass ``--model-root`` or set
``TTS_RESEARCH_MODEL_ROOT``.

Naming note: the sherpa-onnx Python binding returns ``GenerationConfig.extra``
by value, so mutating it in place is silently discarded. Every adapter assigns
a complete dict.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def model_root() -> Path:
    raw = os.environ.get("TTS_RESEARCH_MODEL_ROOT")
    if not raw:
        raise SystemExit(
            "set TTS_RESEARCH_MODEL_ROOT (or pass --model-root) to the external "
            "directory holding the pinned model archives"
        )
    return Path(raw).expanduser().resolve()


@dataclass
class SynthResult:
    samples: np.ndarray
    sample_rate: int
    first_chunk_s: float | None
    chunk_times_s: list
    chunk_sizes: list
    total_s: float
    first_chunk_samples: np.ndarray | None


@dataclass
class Candidate:
    """Static description of one candidate; the ranking tables read this."""

    id: str
    family: str
    languages: Sequence[str]
    # Files that must ship on the phone for this voice, relative to model root.
    shipped_files: Sequence[str]
    license_weights: str
    license_risk: str  # "clear" | "restricted" | "unclear"
    notes: str = ""
    # Files shared with another candidate (counted separately in footprint).
    shared_files: Sequence[str] = field(default_factory=tuple)

    def footprint_bytes(self, root: Path) -> dict:
        def total(paths):
            acc = 0
            for rel in paths:
                p = root / rel
                if p.is_dir():
                    acc += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                elif p.is_file():
                    acc += p.stat().st_size
            return acc

        exclusive = total(self.shipped_files)
        shared = total(self.shared_files)
        return {
            "exclusive_bytes": exclusive,
            "shared_bytes": shared,
            "total_bytes": exclusive + shared,
            "exclusive_mib": round(exclusive / 1048576, 2),
            "total_mib": round((exclusive + shared) / 1048576, 2),
        }


class Engine:
    """Base adapter. ``load`` is timed separately from ``synth``."""

    candidate: Candidate

    def __init__(self, root: Path, threads: int, **kwargs):
        self.root = root
        self.threads = threads
        self.options = kwargs

    def load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def synth(self, text: str, language: str) -> SynthResult:  # pragma: no cover
        raise NotImplementedError

    def config_record(self) -> dict:
        return {"threads": self.threads, **{k: v for k, v in self.options.items()}}


# --------------------------------------------------------------------------
# sherpa-onnx backed engines
# --------------------------------------------------------------------------


class _SherpaEngine(Engine):
    """Shared timing/callback plumbing for every sherpa-onnx candidate."""

    def _model_config(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _generation_extra(self, language: str) -> dict:
        return {}

    def load(self) -> None:
        import sherpa_onnx

        config = sherpa_onnx.OfflineTtsConfig(model=self._model_config())
        if not config.validate():
            raise ValueError(f"invalid sherpa-onnx configuration for {self.candidate.id}")
        self.tts = sherpa_onnx.OfflineTts(config)

    def synth(self, text: str, language: str) -> SynthResult:
        import sherpa_onnx

        gen = sherpa_onnx.GenerationConfig()
        gen.sid = int(self.options.get("sid", 0))
        gen.speed = float(self.options.get("speed", 1.0))
        if self.options.get("num_steps") is not None:
            gen.num_steps = int(self.options["num_steps"])
        extra = self._generation_extra(language)
        if extra:
            gen.extra = extra

        times: list = []
        sizes: list = []
        first: list = []
        start = time.perf_counter()

        def callback(samples, _progress) -> int:
            times.append(time.perf_counter() - start)
            sizes.append(int(np.asarray(samples).size))
            if not first:
                first.append(np.asarray(samples, dtype=np.float32).copy())
            return 1

        audio = self.tts.generate(text, gen, callback)
        total = time.perf_counter() - start
        return SynthResult(
            samples=np.asarray(audio.samples, dtype=np.float32),
            sample_rate=int(audio.sample_rate),
            first_chunk_s=times[0] if times else None,
            chunk_times_s=times,
            chunk_sizes=sizes,
            total_s=total,
            first_chunk_samples=first[0] if first else None,
        )


class MeloZhEngine(_SherpaEngine):
    candidate = Candidate(
        id="melotts_zh",
        family="MeloTTS (VITS)",
        languages=("zh", "en"),
        shipped_files=(
            "vits-melo-tts-zh_en/model.onnx",
            "vits-melo-tts-zh_en/lexicon.txt",
            "vits-melo-tts-zh_en/tokens.txt",
            "vits-melo-tts-zh_en/dict",
            "vits-melo-tts-zh_en/date.fst",
            "vits-melo-tts-zh_en/number.fst",
            "vits-melo-tts-zh_en/phone.fst",
            "vits-melo-tts-zh_en/new_heteronym.fst",
        ),
        license_weights="MIT (MyShell.ai MeloTTS)",
        license_risk="clear",
        notes="Priority-1 Mandarin candidate; has an account-owned exact-SM8850/HTP-v81 placement pass.",
    )

    def _model_config(self):
        import sherpa_onnx

        d = self.root / "vits-melo-tts-zh_en"
        return sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(d / "model.onnx"),
                lexicon=str(d / "lexicon.txt"),
                tokens=str(d / "tokens.txt"),
                dict_dir=str(d / "dict"),
            ),
            num_threads=self.threads,
            provider="cpu",
            debug=False,
        )


class MeloZhInt8Engine(MeloZhEngine):
    """MeloTTS-ZH with the INT8 weights shipped alongside the float model.

    Worth a run on its own terms: MeloTTS is the only MIT, eSpeak-free Mandarin
    candidate, and float32 misses the latency budget on CPU. If INT8 closes that
    gap the licence-clean route becomes CPU-viable; if it does not (ONNX dynamic
    quantization often costs time on ARM), the NPU route is the only option.
    """

    candidate = Candidate(
        id="melotts_zh_int8",
        family="MeloTTS (VITS, INT8)",
        languages=("zh", "en"),
        shipped_files=(
            "vits-melo-tts-zh_en/model.int8.onnx",
            "vits-melo-tts-zh_en/lexicon.txt",
            "vits-melo-tts-zh_en/tokens.txt",
            "vits-melo-tts-zh_en/dict",
            "vits-melo-tts-zh_en/date.fst",
            "vits-melo-tts-zh_en/number.fst",
            "vits-melo-tts-zh_en/phone.fst",
            "vits-melo-tts-zh_en/new_heteronym.fst",
        ),
        license_weights="MIT (MyShell.ai MeloTTS)",
        license_risk="clear",
        notes="INT8 counterpart of melotts_zh; same licence and same NPU story, different CPU cost.",
    )

    def _model_config(self):
        import sherpa_onnx

        d = self.root / "vits-melo-tts-zh_en"
        return sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(d / "model.int8.onnx"),
                lexicon=str(d / "lexicon.txt"),
                tokens=str(d / "tokens.txt"),
                dict_dir=str(d / "dict"),
            ),
            num_threads=self.threads,
            provider="cpu",
            debug=False,
        )


class KokoroZhEngine(_SherpaEngine):
    candidate = Candidate(
        id="kokoro_zh",
        family="Kokoro v1.1 multi-lang (INT8)",
        languages=("zh", "en"),
        shipped_files=(
            "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1/model.int8.onnx",
            "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1/voices.bin",
            "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1/tokens.txt",
            "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1/lexicon-zh.txt",
            "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1/lexicon-us-en.txt",
            "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1/dict",
            "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1/espeak-ng-data",
            "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1/date-zh.fst",
            "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1/number-zh.fst",
            "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1/phone-zh.fst",
        ),
        license_weights="Apache-2.0",
        license_risk="clear",
        notes="Experimental only: short Mandarin inputs previously produced non-finite PCM.",
    )

    def _model_config(self):
        import sherpa_onnx

        d = self.root / "kokoro-int8-v1_1-zh" / "kokoro-int8-multi-lang-v1_1"
        return sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(d / "model.int8.onnx"),
                voices=str(d / "voices.bin"),
                tokens=str(d / "tokens.txt"),
                data_dir=str(d / "espeak-ng-data"),
                dict_dir=str(d / "dict"),
                lexicon=f"{d / 'lexicon-us-en.txt'},{d / 'lexicon-zh.txt'}",
            ),
            num_threads=self.threads,
            provider="cpu",
            debug=False,
        )


class MatchaZhEngine(_SherpaEngine):
    candidate = Candidate(
        id="matcha_zh_baker",
        family="Matcha-TTS (flow matching) + Vocos",
        languages=("zh",),
        shipped_files=(
            "matcha-icefall-zh-baker/model-steps-3.onnx",
            "matcha-icefall-zh-baker/lexicon.txt",
            "matcha-icefall-zh-baker/tokens.txt",
            "matcha-icefall-zh-baker/dict",
            "matcha-icefall-zh-baker/date.fst",
            "matcha-icefall-zh-baker/number.fst",
            "matcha-icefall-zh-baker/phone.fst",
        ),
        shared_files=("vocos-22khz-univ.onnx",),
        license_weights="Baker (data-baker) corpus: non-commercial use only",
        license_risk="restricted",
        notes="New Mandarin candidate chosen for flow-matching naturalness at a small acoustic-model footprint.",
    )

    def _model_config(self):
        import sherpa_onnx

        d = self.root / "matcha-icefall-zh-baker"
        return sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(d / "model-steps-3.onnx"),
                vocoder=str(self.root / "vocos-22khz-univ.onnx"),
                lexicon=str(d / "lexicon.txt"),
                tokens=str(d / "tokens.txt"),
                dict_dir=str(d / "dict"),
            ),
            num_threads=self.threads,
            provider="cpu",
            debug=False,
        )


class KokoroZhFloatEngine(KokoroZhEngine):
    """Kokoro v1.1 multi-lang in float32.

    Run because dynamic INT8 quantization on ONNX Runtime frequently *costs*
    time on ARM: the per-tensor quantize/dequantize around each matmul can
    outweigh the cheaper arithmetic. Kokoro-INT8's 1.385 RTF is slow enough for
    an 82M-parameter model that this is worth measuring rather than assuming.
    """

    candidate = Candidate(
        id="kokoro_zh_fp32",
        family="Kokoro v1.1 multi-lang (float32)",
        languages=("zh", "en"),
        shipped_files=(
            "kokoro-multi-lang-v1_1/model.onnx",
            "kokoro-multi-lang-v1_1/voices.bin",
            "kokoro-multi-lang-v1_1/tokens.txt",
            "kokoro-multi-lang-v1_1/lexicon-zh.txt",
            "kokoro-multi-lang-v1_1/lexicon-us-en.txt",
            "kokoro-multi-lang-v1_1/dict",
            "kokoro-multi-lang-v1_1/espeak-ng-data",
            "kokoro-multi-lang-v1_1/date-zh.fst",
            "kokoro-multi-lang-v1_1/number-zh.fst",
            "kokoro-multi-lang-v1_1/phone-zh.fst",
        ),
        license_weights="Apache-2.0",
        license_risk="clear",
        notes="Float32 counterpart of kokoro_zh, run to separate model cost from INT8 overhead.",
    )

    def _model_config(self):
        import sherpa_onnx

        d = self.root / "kokoro-multi-lang-v1_1"
        return sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(d / "model.onnx"),
                voices=str(d / "voices.bin"),
                tokens=str(d / "tokens.txt"),
                data_dir=str(d / "espeak-ng-data"),
                dict_dir=str(d / "dict"),
                lexicon=f"{d / 'lexicon-us-en.txt'},{d / 'lexicon-zh.txt'}",
            ),
            num_threads=self.threads,
            provider="cpu",
            debug=False,
        )


class ZipVoiceEngine(_SherpaEngine):
    """ZipVoice-distill: zero-shot flow-matching TTS conditioned on a prompt clip.

    Unlike every other candidate here, this one has no fixed voice: it clones
    the speaker from a reference waveform plus that reference's transcript. That
    is a genuine deployment difference — the prompt clip and its text must ship
    with the app and the output identity depends on them — so it is recorded in
    the footprint and in the notes rather than glossed over.
    """

    candidate = Candidate(
        id="zipvoice_zh",
        family="ZipVoice-distill (flow matching, zero-shot)",
        languages=("zh", "en"),
        shipped_files=(
            "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx",
            "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx",
            "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt",
            "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt",
            "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data",
        ),
        shared_files=("vocos_24khz.onnx",),
        license_weights="Emilia dataset training; sherpa-onnx release (verify before shipping)",
        license_risk="unclear",
        notes="Zero-shot: needs a reference clip plus its transcript at run time. Prompt identity is part of the deliverable.",
    )

    def _model_config(self):
        import sherpa_onnx

        d = self.root / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"
        return sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(d / "tokens.txt"),
                encoder=str(d / "encoder.int8.onnx"),
                decoder=str(d / "decoder.int8.onnx"),
                vocoder=str(self.root / "vocos_24khz.onnx"),
                data_dir=str(d / "espeak-ng-data"),
                lexicon=str(d / "lexicon.txt"),
            ),
            num_threads=self.threads,
            provider="cpu",
            debug=False,
        )

    def load(self) -> None:
        import soundfile as sf

        super().load()
        prompt = self.options.get("prompt_wav")
        if not prompt:
            prompt = str(
                self.root
                / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"
                / "test_wavs"
                / "news-female.wav"
            )
        self.prompt_samples, self.prompt_rate = sf.read(prompt, dtype="float32", always_2d=False)
        self.prompt_text = self.options.get(
            "prompt_text", "各位村民, 大家新年好! 近期, 湖北省武汉市等多个地区"
        )
        self.prompt_name = Path(prompt).name

    def synth(self, text: str, language: str) -> SynthResult:
        import sherpa_onnx
        import time as _time

        gen = sherpa_onnx.GenerationConfig()
        gen.speed = float(self.options.get("speed", 1.0))
        if self.options.get("num_steps") is not None:
            gen.num_steps = int(self.options["num_steps"])
        gen.reference_audio = self.prompt_samples
        gen.reference_sample_rate = int(self.prompt_rate)
        gen.reference_text = self.prompt_text

        times: list = []
        sizes: list = []
        first: list = []
        start = _time.perf_counter()

        def callback(samples, _progress) -> int:
            times.append(_time.perf_counter() - start)
            sizes.append(int(np.asarray(samples).size))
            if not first:
                first.append(np.asarray(samples, dtype=np.float32).copy())
            return 1

        audio = self.tts.generate(text, gen, callback)
        total = _time.perf_counter() - start
        return SynthResult(
            samples=np.asarray(audio.samples, dtype=np.float32),
            sample_rate=int(audio.sample_rate),
            first_chunk_s=times[0] if times else None,
            chunk_times_s=times,
            chunk_sizes=sizes,
            total_s=total,
            first_chunk_samples=first[0] if first else None,
        )

    def config_record(self) -> dict:
        return {
            **super().config_record(),
            "prompt_wav": getattr(self, "prompt_name", None),
            "prompt_text": getattr(self, "prompt_text", None),
        }


class ZipVoiceFloatEngine(ZipVoiceEngine):
    """Float32 ZipVoice-distill, run only as a quality ceiling.

    At 502 MiB it is over the 500 MiB conditional footprint ceiling and the INT8
    build is already far outside any first-audio budget, so this cannot be a
    phone candidate. It exists to answer one question the INT8 run cannot: how
    much quality, if any, the dynamic quantization cost.
    """

    candidate = Candidate(
        id="zipvoice_zh_fp32",
        family="ZipVoice-distill float32 (quality ceiling, not a phone candidate)",
        languages=("zh", "en"),
        shipped_files=(
            "sherpa-onnx-zipvoice-distill-fp32-zh-en-emilia/encoder.onnx",
            "sherpa-onnx-zipvoice-distill-fp32-zh-en-emilia/decoder.onnx",
            "sherpa-onnx-zipvoice-distill-fp32-zh-en-emilia/tokens.txt",
            "sherpa-onnx-zipvoice-distill-fp32-zh-en-emilia/lexicon.txt",
            "sherpa-onnx-zipvoice-distill-fp32-zh-en-emilia/espeak-ng-data",
        ),
        shared_files=("vocos_24khz.onnx",),
        license_weights="Emilia dataset training; sherpa-onnx release (verify before shipping)",
        license_risk="unclear",
        notes="Reference ceiling only. Over the footprint ceiling and far outside any latency budget.",
    )

    def _model_config(self):
        import sherpa_onnx

        d = self.root / "sherpa-onnx-zipvoice-distill-fp32-zh-en-emilia"
        return sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(d / "tokens.txt"),
                encoder=str(d / "encoder.onnx"),
                decoder=str(d / "decoder.onnx"),
                vocoder=str(self.root / "vocos_24khz.onnx"),
                data_dir=str(d / "espeak-ng-data"),
                lexicon=str(d / "lexicon.txt"),
            ),
            num_threads=self.threads,
            provider="cpu",
            debug=False,
        )

    def load(self) -> None:
        # The prompt clip lives in the INT8 package; reuse it so both precisions
        # are conditioned on exactly the same reference.
        self.options.setdefault(
            "prompt_wav",
            str(self.root / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"
                / "test_wavs" / "news-female.wav"),
        )
        super().load()


class PiperEngine(_SherpaEngine):
    """One Piper VITS voice. ``voice_dir`` selects the language/speaker."""

    def __init__(self, root: Path, threads: int, *, candidate: Candidate, voice_dir: str, voice_stem: str, **kwargs):
        super().__init__(root, threads, **kwargs)
        self.candidate = candidate
        self.voice_dir = voice_dir
        self.voice_stem = voice_stem

    def _model_config(self):
        import sherpa_onnx

        d = self.root / self.voice_dir
        return sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(d / f"{self.voice_stem}.onnx"),
                tokens=str(d / "tokens.txt"),
                data_dir=str(d / "espeak-ng-data"),
            ),
            num_threads=self.threads,
            provider="cpu",
            debug=False,
        )


class SupertonicEngine(_SherpaEngine):
    candidate = Candidate(
        id="supertonic3",
        family="Supertonic 3 (INT8, diffusion-style denoiser)",
        languages=("ru", "en", "fr", "it", "es", "de", "ja", "ko"),
        shipped_files=(
            "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx",
            "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx",
            "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx",
            "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx",
            "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json",
            "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin",
            "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin",
        ),
        license_weights="OpenRAIL-M (use restrictions apply; redistribution permitted)",
        license_risk="restricted",
        notes="Russian incumbent; 31 languages but no Mandarin. Step count is the main quality/latency knob.",
    )

    def _model_config(self):
        import sherpa_onnx

        d = self.root / "supertonic3-int8" / "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
        return sherpa_onnx.OfflineTtsModelConfig(
            supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor=str(d / "duration_predictor.int8.onnx"),
                text_encoder=str(d / "text_encoder.int8.onnx"),
                vector_estimator=str(d / "vector_estimator.int8.onnx"),
                vocoder=str(d / "vocoder.int8.onnx"),
                tts_json=str(d / "tts.json"),
                unicode_indexer=str(d / "unicode_indexer.bin"),
                voice_style=str(d / "voice.bin"),
            ),
            num_threads=self.threads,
            provider="cpu",
            debug=False,
        )

    def _generation_extra(self, language: str) -> dict:
        return {"lang": language, "seed": "0"}


# --------------------------------------------------------------------------
# eSpeak NG control
# --------------------------------------------------------------------------


class EspeakEngine(Engine):
    """The robotic control this research is trying to replace."""

    candidate = Candidate(
        id="espeak_ng",
        family="eSpeak NG (formant synthesis)",
        languages=("zh", "ru", "en", "fr", "it", "es"),
        shipped_files=(),
        license_weights="GPL-3.0-or-later (engine + data)",
        license_risk="restricted",
        notes="Control only. Offline and tiny, but explicitly not an acceptable demo voice.",
    )

    VOICES = {"zh": "cmn", "ru": "ru", "en": "en-us", "fr": "fr", "it": "it", "es": "es"}

    def load(self) -> None:
        self.binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.binary:
            raise SystemExit("espeak-ng is not installed")

    def synth(self, text: str, language: str) -> SynthResult:
        import soundfile as sf

        voice = self.VOICES.get(language, language)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "espeak.wav"
            start = time.perf_counter()
            subprocess.run(
                [self.binary, "-v", voice, "-w", str(out), "--", text],
                check=True,
                capture_output=True,
            )
            total = time.perf_counter() - start
            samples, rate = sf.read(out, dtype="float32", always_2d=False)
        samples = np.asarray(samples, dtype=np.float32)
        # eSpeak has no streaming callback; the whole utterance is one chunk.
        return SynthResult(
            samples=samples,
            sample_rate=int(rate),
            first_chunk_s=total,
            chunk_times_s=[total],
            chunk_sizes=[int(samples.size)],
            total_s=total,
            first_chunk_samples=samples,
        )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_PIPER_ZH_HUAYAN = Candidate(
    id="piper_zh_huayan",
    family="Piper (VITS medium)",
    languages=("zh",),
    shipped_files=(
        "vits-piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx",
        "vits-piper-zh_CN-huayan-medium/tokens.txt",
        "vits-piper-zh_CN-huayan-medium/espeak-ng-data",
    ),
    license_weights="HuaYan_TTS dataset: license Unknown (per Piper MODEL_CARD)",
    license_risk="unclear",
    notes="New Mandarin candidate chosen because Piper already has an account-owned exact-SM8850 NPU chain.",
)

_PIPER_RU_DMITRI = Candidate(
    id="piper_ru_dmitri",
    family="Piper (VITS medium)",
    languages=("ru",),
    shipped_files=(
        "vits-piper-ru_RU-dmitri-medium/ru_RU-dmitri-medium.onnx",
        "vits-piper-ru_RU-dmitri-medium/tokens.txt",
        "vits-piper-ru_RU-dmitri-medium/espeak-ng-data",
    ),
    license_weights="Nabu Casa voice-datasets: CC0",
    license_risk="clear",
    notes="New Russian candidate; the only compact Russian voice inspected with a public-domain training corpus.",
)

_PIPER_RU_RUSLAN = Candidate(
    id="piper_ru_ruslan",
    family="Piper (VITS medium)",
    languages=("ru",),
    shipped_files=(
        "vits-piper-ru_RU-ruslan-medium/ru_RU-ruslan-medium.onnx",
        "vits-piper-ru_RU-ruslan-medium/tokens.txt",
        "vits-piper-ru_RU-ruslan-medium/espeak-ng-data",
    ),
    license_weights="RUSLAN corpus: CC BY-NC-SA 4.0",
    license_risk="restricted",
    notes="Russian reference voice; non-commercial corpus.",
)

_PIPER_RU_IRINA = Candidate(
    id="piper_ru_irina",
    family="Piper (VITS medium)",
    languages=("ru",),
    shipped_files=(
        "vits-piper-ru_RU-irina-medium/ru_RU-irina-medium.onnx",
        "vits-piper-ru_RU-irina-medium/tokens.txt",
        "vits-piper-ru_RU-irina-medium/espeak-ng-data",
    ),
    license_weights="RHVoice corpus: license Unknown (per Piper MODEL_CARD)",
    license_risk="unclear",
    notes="Russian reference voice only; unclear corpus license blocks a recommendation.",
)


def build(engine_id: str, root: Path, threads: int, **kwargs) -> Engine:
    if engine_id == "melotts_zh":
        return MeloZhEngine(root, threads, **kwargs)
    if engine_id == "melotts_zh_int8":
        return MeloZhInt8Engine(root, threads, **kwargs)
    if engine_id == "kokoro_zh":
        return KokoroZhEngine(root, threads, **kwargs)
    if engine_id == "kokoro_zh_fp32":
        return KokoroZhFloatEngine(root, threads, **kwargs)
    if engine_id == "zipvoice_zh":
        return ZipVoiceEngine(root, threads, **kwargs)
    if engine_id == "zipvoice_zh_fp32":
        return ZipVoiceFloatEngine(root, threads, **kwargs)
    if engine_id == "matcha_zh_baker":
        return MatchaZhEngine(root, threads, **kwargs)
    if engine_id == "supertonic3":
        return SupertonicEngine(root, threads, **kwargs)
    if engine_id == "espeak_ng":
        return EspeakEngine(root, threads, **kwargs)
    piper = {
        "piper_zh_huayan": (_PIPER_ZH_HUAYAN, "vits-piper-zh_CN-huayan-medium", "zh_CN-huayan-medium"),
        "piper_ru_dmitri": (_PIPER_RU_DMITRI, "vits-piper-ru_RU-dmitri-medium", "ru_RU-dmitri-medium"),
        "piper_ru_ruslan": (_PIPER_RU_RUSLAN, "vits-piper-ru_RU-ruslan-medium", "ru_RU-ruslan-medium"),
        "piper_ru_irina": (_PIPER_RU_IRINA, "vits-piper-ru_RU-irina-medium", "ru_RU-irina-medium"),
    }
    if engine_id in piper:
        cand, vdir, stem = piper[engine_id]
        return PiperEngine(root, threads, candidate=cand, voice_dir=vdir, voice_stem=stem, **kwargs)
    raise SystemExit(f"unknown engine id: {engine_id}")


ALL_ENGINE_IDS = (
    "melotts_zh",
    "melotts_zh_int8",
    "kokoro_zh",
    "kokoro_zh_fp32",
    "zipvoice_zh",
    "zipvoice_zh_fp32",
    "matcha_zh_baker",
    "piper_zh_huayan",
    "supertonic3",
    "piper_ru_dmitri",
    "piper_ru_ruslan",
    "piper_ru_irina",
    "espeak_ng",
)


def candidate_for(engine_id: str) -> Candidate:
    """Static metadata without loading any weights."""
    if engine_id == "melotts_zh":
        return MeloZhEngine.candidate
    if engine_id == "melotts_zh_int8":
        return MeloZhInt8Engine.candidate
    if engine_id == "kokoro_zh":
        return KokoroZhEngine.candidate
    if engine_id == "kokoro_zh_fp32":
        return KokoroZhFloatEngine.candidate
    if engine_id == "zipvoice_zh":
        return ZipVoiceEngine.candidate
    if engine_id == "zipvoice_zh_fp32":
        return ZipVoiceFloatEngine.candidate
    if engine_id == "matcha_zh_baker":
        return MatchaZhEngine.candidate
    if engine_id == "supertonic3":
        return SupertonicEngine.candidate
    if engine_id == "espeak_ng":
        return EspeakEngine.candidate
    return {
        "piper_zh_huayan": _PIPER_ZH_HUAYAN,
        "piper_ru_dmitri": _PIPER_RU_DMITRI,
        "piper_ru_ruslan": _PIPER_RU_RUSLAN,
        "piper_ru_irina": _PIPER_RU_IRINA,
    }[engine_id]
