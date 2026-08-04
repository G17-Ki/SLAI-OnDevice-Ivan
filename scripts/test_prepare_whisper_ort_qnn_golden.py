#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import prepare_whisper_ort_qnn_golden as golden


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class WhisperGoldenPureTest(unittest.TestCase):
    def test_mel_filter_hash(self) -> None:
        filters = golden.canonical_mel_filters()
        self.assertEqual(filters.shape, (201, 80))
        self.assertEqual(filters.dtype, np.float32)
        self.assertEqual(
            digest(filters.astype("<f4", copy=False).tobytes(order="C")),
            golden.EXPECTED_MEL_FILTER_SHA256,
        )

    def test_mask_quantization_formula(self) -> None:
        logical = np.asarray([-100.0, 0.0], dtype=np.float32)
        quantized = golden.quantize_uint16(
            logical,
            scale=golden.MASK_QUANTIZATION["scale"],
            zero_point=golden.MASK_QUANTIZATION["zero_point"],
        )
        self.assertEqual(quantized.dtype, np.dtype("<u2"))
        self.assertEqual(quantized.tolist(), [0, 65_535])

    def test_feature_quantization_uses_qai_float64_intermediate(self) -> None:
        # This Japanese-fixture value looks like a half in float32. QAI Hub
        # Models casts to float64 before division, so the correct result is
        # below the half and rounds down.
        logical = np.asarray([-0.8696660995483398], dtype=np.float32)
        quantized = golden.quantize_uint16(
            logical,
            scale=golden.FEATURE_QUANTIZATION["scale"],
            zero_point=golden.FEATURE_QUANTIZATION["zero_point"],
        )
        float32_intermediate = np.rint(
            logical / np.float32(golden.FEATURE_QUANTIZATION["scale"])
        ) + golden.FEATURE_QUANTIZATION["zero_point"]
        self.assertEqual(quantized.tolist(), [13_477])
        self.assertEqual(float32_intermediate.astype(np.uint16).tolist(), [13_478])

    def test_decoder_step0_reference_bytes(self) -> None:
        mask = np.zeros(200, dtype="<u2")
        mask[-1] = 65_535
        self.assertEqual(
            digest(mask.tobytes()),
            "116af0fc853dad9b457ed09d126242b536b4f9258a0b500d22d69cb42a1de71f",
        )
        self.assertEqual(
            digest(np.asarray([[50_258]], dtype="<i4").tobytes()),
            "eea21505fadd32ed72c78271dfbdf37c5bcb00f41b51e44e0e8a8d8eb9dbdfe7",
        )
        self.assertEqual(
            digest(np.asarray([0], dtype="<i4").tobytes()),
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
        )
        self_kv = np.full((12, 1, 64, 199), 128, dtype=np.uint8)
        self.assertEqual(
            digest(self_kv.tobytes()),
            "df75a967b954ebddbc0176f41a5564065e193b165586c071481e6cc1bf71aedb",
        )

    def test_detokenization_skips_special_ids_and_decodes_once(self) -> None:
        entries = [b"\xe2", b"\x82", b"\xac"]
        self.assertEqual(
            golden.decode_mergeable_tokens(entries, [0, 99, 1, 2]),
            "€",
        )

    def test_write_reproducible_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.bin"
            golden.write_reproducible(path, b"first", overwrite=False)
            golden.write_reproducible(path, b"first", overwrite=False)
            with self.assertRaises(golden.GoldenInputError):
                golden.write_reproducible(path, b"second", overwrite=False)
            golden.write_reproducible(path, b"second", overwrite=True)
            self.assertEqual(path.read_bytes(), b"second")


class WhisperGoldenPinnedIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.required_inputs = [
            golden.DEFAULT_WAV,
            golden.DEFAULT_MODEL_ARCHIVE,
            golden.DEFAULT_VOICE_ARCHIVE,
            golden.DEFAULT_QAI_WHEEL,
            golden.DEFAULT_TRANSFORMERS_WHEEL,
            golden.DEFAULT_MLX_WHISPER_WHEEL,
            golden.DEFAULT_MANIFEST,
        ]
        missing = [str(path) for path in cls.required_inputs if not path.is_file()]
        if missing:
            raise unittest.SkipTest("pinned local inputs unavailable: " + ", ".join(missing))
        cls.temp = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls.temp.name)
        cls.output_dir = cls.temp_root / "artifacts"
        cls.manifest_path = cls.temp_root / "manifest.json"
        cls.generated = golden.generate_bundle(
            output_dir=cls.output_dir,
            manifest_path=cls.manifest_path,
            wav_path=golden.DEFAULT_WAV,
            model_archive=golden.DEFAULT_MODEL_ARCHIVE,
            voice_archive=golden.DEFAULT_VOICE_ARCHIVE,
            qai_wheel=golden.DEFAULT_QAI_WHEEL,
            transformers_wheel=golden.DEFAULT_TRANSFORMERS_WHEEL,
            mlx_whisper_wheel=golden.DEFAULT_MLX_WHISPER_WHEEL,
        )
        cls.tracked = json.loads(golden.DEFAULT_MANIFEST.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_regeneration_matches_tracked_manifest(self) -> None:
        self.assertEqual(self.generated, self.tracked)
        self.assertEqual(
            self.manifest_path.read_bytes(),
            golden.DEFAULT_MANIFEST.read_bytes(),
        )

    def test_artifacts_match_manifest(self) -> None:
        expected_hashes = {
            "attention_mask_step0.u16le.bin": (
                "116af0fc853dad9b457ed09d126242b536b4f9258a0b500d22d69cb42a1de71f"
            ),
            "input_features.u16le.bin": golden.EXPECTED_QUANTIZED_FEATURE_SHA256,
            "input_ids_step0.i32le.bin": (
                "eea21505fadd32ed72c78271dfbdf37c5bcb00f41b51e44e0e8a8d8eb9dbdfe7"
            ),
            "mel_filters.f32le.bin": golden.EXPECTED_MEL_FILTER_SHA256,
            "position_ids_step0.i32le.bin": (
                "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
            ),
            "self_kv_zero_template.u8.bin": (
                "df75a967b954ebddbc0176f41a5564065e193b165586c071481e6cc1bf71aedb"
            ),
        }
        self.assertEqual(set(self.generated["artifacts"]), set(expected_hashes))
        for name, expected_sha256 in expected_hashes.items():
            value = (self.output_dir / name).read_bytes()
            self.assertEqual(digest(value), expected_sha256)
            self.assertEqual(
                len(value),
                self.generated["artifacts"][name]["bytes"],
            )

    def test_canonical_wav_and_frontend_contract(self) -> None:
        audio = self.generated["canonical_audio"]
        self.assertEqual(audio["sha256"], golden.WAV["sha256"])
        self.assertEqual(audio["pcm_sha256"], golden.WAV["pcm_sha256"])
        self.assertEqual(audio["frames"], 52_819)
        frontend = self.generated["frontend"]
        self.assertEqual(
            frontend["float_output"]["sha256"],
            golden.EXPECTED_FLOAT_FEATURE_SHA256,
        )
        self.assertEqual(frontend["float_output"]["shape"], [1, 80, 3000])
        self.assertEqual(frontend["input_features_quantization"]["minimum"], 18_112)
        self.assertEqual(frontend["input_features_quantization"]["maximum"], 60_874)

    def test_decoder_step0_artifact_semantics(self) -> None:
        mask = np.frombuffer(
            (self.output_dir / "attention_mask_step0.u16le.bin").read_bytes(),
            dtype="<u2",
        )
        self.assertEqual(mask.shape, (200,))
        self.assertTrue(np.all(mask[:199] == 0))
        self.assertEqual(int(mask[199]), 65_535)
        input_ids = np.frombuffer(
            (self.output_dir / "input_ids_step0.i32le.bin").read_bytes(),
            dtype="<i4",
        )
        position_ids = np.frombuffer(
            (self.output_dir / "position_ids_step0.i32le.bin").read_bytes(),
            dtype="<i4",
        )
        self.assertEqual(input_ids.tolist(), [50_258])
        self.assertEqual(position_ids.tolist(), [0])
        self_kv = np.frombuffer(
            (self.output_dir / "self_kv_zero_template.u8.bin").read_bytes(),
            dtype=np.uint8,
        )
        self.assertEqual(self_kv.size, 12 * 1 * 64 * 199)
        self.assertTrue(np.all(self_kv == 128))

    def test_qai_source_and_release_pins(self) -> None:
        sources = self.generated["sources"]
        self.assertEqual(
            sources["qai_hub_models_wheel"]["sha256"],
            golden.QAI_WHEEL["sha256"],
        )
        self.assertEqual(
            sources["qai_hub_models_wheel"]["source_members"],
            golden.QAI_SOURCE_MEMBERS,
        )
        self.assertEqual(
            sources["model_release"]["sha256"],
            golden.MODEL_ARCHIVE["sha256"],
        )
        self.assertEqual(
            self.generated["model"]["metadata_sha256"],
            golden.MODEL_METADATA["sha256"],
        )
        self.assertEqual(
            sources["transformers_reference_wheel"]["sha256"],
            golden.TRANSFORMERS_WHEEL["sha256"],
        )

    def test_tiktoken_and_voice_vocab_equivalence(self) -> None:
        tiktoken_bytes = golden.read_verified_member(
            golden.DEFAULT_MLX_WHISPER_WHEEL,
            golden.TIKTOKEN_MEMBER,
            golden.TIKTOKEN,
            "test tokenizer wheel",
        )
        tokenizer_entries = golden.parse_tiktoken(tiktoken_bytes)
        voice_vocab = golden.read_verified_member(
            golden.DEFAULT_VOICE_ARCHIVE,
            golden.VOICE_VOCAB_MEMBER,
            golden.VOICE_VOCAB,
            "test VoiceAI archive",
        )
        self.assertEqual(golden.build_qai_voice_vocab(tokenizer_entries), voice_vocab)
        split_entries = golden.split_voice_vocab(voice_vocab)
        self.assertEqual(len(split_entries), 50_257)
        self.assertEqual(tokenizer_entries[188], b"\0")
        self.assertEqual(split_entries[188], b"\0")
        self.assertEqual(tokenizer_entries[50_256], b"")
        self.assertEqual(split_entries[50_256], b"")
        self.assertEqual(
            golden.decode_mergeable_tokens(
                tokenizer_entries,
                [*golden.REFERENCE_TOKEN_IDS, golden.SPECIAL_TOKEN_IDS["eot"]],
            ),
            golden.WAV["text"],
        )

    def test_scope_stays_fail_closed(self) -> None:
        blocker_codes = {item["code"] for item in self.generated["blockers"]}
        self.assertEqual(
            blocker_codes,
            {
                "full_asr_output_tokens_unavailable_offline",
                "runner_and_app_uid_evidence_pending",
            },
        )
        self.assertFalse(self.generated["policy"]["adb_invoked"])
        self.assertFalse(self.generated["policy"]["phone_mutated"])
        self.assertIn(
            "ASR transcript correctness",
            self.generated["scope"]["does_not_prove"],
        )


if __name__ == "__main__":
    unittest.main()
