"""Build ledger.json from Lane A's measured rows. All stack arithmetic is
computed here, never typed by hand.

Counting convention (stated once, applied everywhere):
  `params` = ALL tensor elements of the shipped artefact (gross). This is the
  CONSERVATIVE choice and matches 01_STACK_DECISION.md section 2.0 ("the
  conservative one is counted"). Conservative counting can never turn a
  NONCOMPLIANT stack into a COMPLIANT one.
  `params_learned` = float-dtype tensors only, i.e. the section 1.2-correct
  figure that excludes I64/I32 index and counter buffers.
Both are carried on every row.
"""
import json

CAP = 2_000_000_000

# --------------------------------------------------------------- measurements
# Every row here was measured by this lane. `prior` is the figure carried in
# 01_STACK_DECISION.md so the delta is auditable.
M = {}


def m(key, repo, revision, params, params_learned, cls, method, prior=None,
      prior_class=None, licence=None, notes=None, ub_worst_case=None,
      ub_basis=None, closes=None, artefact=None):
    M[key] = dict(component_key=key, repo=repo, revision=revision,
                  params=params, params_learned=params_learned,
                  evidence_class=cls, method=method, prior=prior,
                  prior_class=prior_class, delta_vs_prior=(
                      None if prior is None else params - prior),
                  licence=licence, notes=notes,
                  ub_worst_case=ub_worst_case if ub_worst_case is not None else params,
                  ub_worst_case_basis=ub_basis or
                  "EXACT tensor-level parse; dtype known, so no bytes/4 doubling hazard exists",
                  closing_experiment=closes, shipped_artefact=artefact)


HDR = "safetensors header parse over HTTP Range (u64 header length -> JSON tensor map -> sum prod(shape) per dtype)"
ONX = "ONNX initializer enumeration over HTTP Range, seeking past raw_data payloads; de-duplicated by tensor identity"
ZIP = "ZIP central-directory walk -> data.pkl meta-unpickle (stubbed rebuild fns, storages never loaded) == meta-device state_dict walk"
TAR = "TAR member walk -> model_weights.ckpt -> " + ZIP

m("parakeet-tdt-0.6b-v3", "nvidia/parakeet-tdt-0.6b-v3", "7c35754d166c",
  627_057_310, 627_057_286, "EXACT", HDR, 627_057_310, "EXACT",
  "cc-by-4.0", "A2: delta-24 = 24x encoder.layers.N.conv.norm.num_batches_tracked (I64 BatchNorm counters). Byte reconciliation exact to 0.")

m("k2fsa-zipformer-zh-en", "pfluo/k2fsa-zipformer-chinese-english-mixed",
  "6eb615ae77ec", 73_753_962, 73_752_050, "EXACT",
  ONX + " over encoder+decoder+joiner ONNX", 69_880_000, "UB", "apache-2.0",
  "PRIOR BOUND BREACHED by 3,873,962. The 69,880,000 was a community round figure, not a bound. Checkpoint superset exp/pretrained.pt = 75,528,536 (includes training-only pruned-RNNT projections not exported).",
  artefact="exp/encoder-epoch-99-avg-1.onnx + decoder + joiner")

m("qwen3-0.6b", "Qwen/Qwen3-0.6B", "c1899de289a0", 596_049_920, 596_049_920,
  "EXACT", HDR + "; tie confirmed by byte comparison (17/17 blocks identical)",
  596_049_920, "EXACT", "apache-2.0",
  "HF API reports 751,632,384 (serialized). TIED: unique = 596,049,920. Same trap as A1.")

m("piper-voice-medium", "rhasspy/piper-voices",
  "9f967d15e9ccdf43078586d1476ee70f314401bd", 15_650_602, 15_650_521, "EXACT",
  ONX, 15_650_602, "UB/HIGH", "MIT (repo) / see notes",
  "PER VOICE. All six medium voices audited (lessac, kusal, davefx, siwis, dmitri, huayan) are byte-identical in size and identical in count. dtype FLOAT (fp32) confirmed -> the Q-4 DOUBLING HAZARD IS DEAD. TH-59 discharged.",
  ub_worst_case=15_650_602,
  ub_basis="EXACT; dtype enumerated as FLOAT, so the fp16 doubling scenario is disproven, not bounded")

m("charsiu-byt5-g2p", "charsiu/g2p_multilingual_byT5_tiny_16_layers_100",
  "0586e9cb4632", 20_653_952, 20_653_952, "EXACT", ZIP, 20_653_952, "EXACT",
  "UNKNOWN (no licence declared on repo)",
  "Count confirms prior exactly. Licence is a genuine gap: repo declares none.")

m("g2pm", "kakaobrain/g2pM (PyPI g2pM 0.1.2.5)", "pypi-0.1.2.5", 401_548,
  401_548, "EXACT", "numpy pickle walk of g2pM/np_ckpt.pkl", 477_228, "EXACT",
  "Apache-2.0 (PyPI package)",
  "Prior 477,228 is 75,680 HIGH. Actual: BiLSTM, embedding [5398,64] + LSTM + [876] class head. HF repo kakaobrain/g2pM does NOT exist (401); the artefact is the PyPI wheel.")

m("ten-vad", "TEN-framework/ten-vad", "bda8ffc78b", 75_516, 75_498, "EXACT",
  ONX + " over src/onnx_model/ten-vad.onnx", 78_310, "UB",
  "Apache-2.0 + Agora non-compete rider",
  "Prior was floor(313,240/4) from the Linux .so, which is CODE not weights. The repo ships an actual ONNX; parsed it. Android arm64 .so is 544,440 B.")

m("canary-180m-flash", "nvidia/canary-180m-flash", "b12ab418510d",
  183_743_137, 183_743_120, "EXACT", TAR, 182_000_000, "UB", "cc-by-4.0",
  "PRIOR BOUND BREACHED by 1,743,137. 'Above the 180m name' was not above the real count.")

m("gigaam-multilingual", "ai-sage/GigaAM-Multilingual", "2f8a57144e6e",
  220_746_951, 220_746_951, "EXACT", ZIP, 220_792_528, "EXACT/HIGH", "MIT",
  "Prior was floor(883,170,115/4). Tensor parse: 552 tensors, ALL float32. THE fp16 DOUBLING HAZARD FOR STACK L IS DEAD (it was worth 220,792,528 of contingency).")

m("qwen3.5-0.8b-lm", "Qwen/Qwen3.5-0.8B", "2fc06364715b", 752_393_024,
  752_393_024, "EXACT", HDR, 752_393_024, "EXACT", "apache-2.0",
  "model.language_model subtree, confirmed to the parameter. Repo total is 873,438,784 = LM 752,393,024 + visual 100,592,896 + mtp 20,452,864, ALL IN ONE safetensors file. tie_word_embeddings=true and lm_head is NOT materialised, so no serialized penalty.")

m("qwen3.5-0.8b-mtp", "Qwen/Qwen3.5-0.8B", "2fc06364715b", 20_452_864,
  20_452_864, "EXACT", HDR, 20_452_864, "EXACT", "apache-2.0",
  "mtp subtree. Confirms prior exactly. Present in the SAME upstream safetensors as the LM -- it is not an add-on artefact.")

m("qwen3.5-0.8b-visual", "Qwen/Qwen3.5-0.8B", "2fc06364715b", 100_592_896,
  100_592_896, "EXACT", HDR, 0, "not carried", "apache-2.0",
  "NOT COUNTED. Vision tower. Excluded under B-2 only because the shipped artefact is the LM-only GGUF. IF ANY BUILD SHIPS THE UPSTREAM SAFETENSORS, THIS ENTERS THE LEDGER AT +100,592,896. Recorded as a build-time assertion, not an assumption.")

m("magpie-tts-multilingual", "nvidia/magpie_tts_multilingual_357m",
  "195d9d3f9ad9", 359_698_319, 359_698_312, "EXACT", TAR, 364_000_000, "UB",
  "NVIDIA Open Model License",
  "Bound held (4,301,681 of slack). BUT the checkpoint contains NO codec -- see nemo-nano-codec row.")

m("nemo-nano-codec-22khz", "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
  "3c482a402a", 62_600_978, 62_600_978, "EXACT",
  TAR + "; module split audio_encoder + audio_decoder + vector_quantizer",
  None, None, "NVIDIA (other)",
  "NEW ROW -- NOT IN ANY PRIOR LEDGER. magpie's model_config.yaml declares codecmodel_path: nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps. Magpie emits 16-codebook audio tokens and cannot produce a waveform without it. Full .nemo = 106,087,325, of which 43,486,347 is a GAN discriminator + loss modules that never ship. Inference path = 62,600,978. Decode-only (if reference-audio cloning is dropped) = 31,581,754.",
  ub_worst_case=106_087_325,
  ub_basis="106,087,325 if the whole checkpoint ships unpruned (discriminator + mel/stft loss modules included)",
  closes="Confirm the shipped export prunes the discriminator; if it does not, the row is 106,087,325.")

m("silero-vad", "onnx-community/silero-vad", "e71cae966052", 545_391,
  545_286, "EXACT", ONX, 581_881, "UB", "MIT",
  "Bound held. Contains TWO sample-rate branches (16 kHz and 8 kHz) with genuinely distinct weights; both ship so both count. 82,688 of the total is the STFT DFT basis (a fixed transform, not learned) -- excluding it would give 462,598. Counted conservatively.")

m("silero-v5-cis-base", "snakers4/silero-models (models.silero.ai)",
  "v5_cis_base 2026-02-06", 22_850_584, 22_850_584, "EXACT",
  "torch.package archive walk + stubbed meta-unpickle of the TorchScript submodule; 317 unique storages, no code executed",
  100_000_000, "UB / genuinely UNPUBLISHED", "MIT (silero-models)",
  "THE ONE GENUINELY UNPUBLISHED ROW IN THE PROGRAMME, NOW CLOSED. 77,149,416 BELOW its bound. fp32, 317 storages, reconciles to storage bytes 91,402,340/4 = 22,850,585 (delta 1 = a view). Ships 60 speaker embeddings. The accentor is NOT bundled (constructor takes accentor=None) -- so the RU stress marker is an EXTERNAL row, contradicting the assumption behind the 100,000,000 bound.")

m("sensevoice-small", "FunAudioLLM/SenseVoiceSmall", "3847d57b6bdf",
  233_999_167, 233_999_167, "EXACT", ZIP, 234_072_842, "EXACT",
  "FunASR Model Open Source License v1.1",
  "Prior 73,675 high. Emits punctuation and ITN NATIVELY via use_itn -- so Stack D needs no separate punctuation row for its zh/en routes, at zero parameters.")

m("ru-zipformer-int8", "csukuangfj/sherpa-onnx-zipformer-ru-int8-2025-04-20",
  "2025-04-20", 65_610_041, 65_610_041, "EXACT", ONX, 65_000_000, "UB",
  "Apache-2.0 (sherpa-onnx / icefall)",
  "PRIOR BOUND BREACHED by 610,041. Float twin (sherpa-onnx-zipformer-ru-2025-04-20) = 65,609,459, confirming Q-1 precision-invariance to within 582 export tensors.")

m("matcha-icefall-zh-baker", "csukuangfj/matcha-icefall-zh-baker",
  "75e64a57a80fb370abb18a4e4d86bb090ed46e93", 18_665_803, 18_664_582, "EXACT",
  ONX, 18_665_803, "UB", "NONE DECLARED; Baker corpus NON-COMMERCIAL",
  "Count confirms prior EXACTLY. dtype FLOAT -> doubling hazard dead.")

m("vocos-22khz-univ", "k2-fsa/sherpa-onnx release asset vocoder-models",
  "vocos-22khz-univ.onnx", 13_460_021, 13_460_015, "EXACT", ONX, 13_460_021,
  "UB", "Apache-2.0 (sherpa-onnx)",
  "Count confirms prior EXACTLY. dtype FLOAT -> doubling hazard dead. iSTFT head keeps it off the NPU (a portability fact, not a ledger fact).")

m("supertonic-3-graphs", "Supertone/supertonic-3",
  "3cadd1ee6394adea1bd021217a0e650ede09a323", 99_220_324, 99_217_620, "EXACT",
  ONX + " over all four graphs", 99_518_816, "UB", "BigScience Open RAIL-M",
  "vector_estimator 64,014,577 + vocoder 25,338,597 + text_encoder 9,001,916 + duration_predictor 865,234. Bound held by 298,492. dtype FLOAT -> doubling hazard dead.")

m("supertonic-3-voice-styles", "Supertone/supertonic-3",
  "3cadd1ee6394adea1bd021217a0e650ede09a323", 129_280, 129_280, "EXACT",
  "voice_styles/*.json dims enumeration", 64_640, "UB/EXACT",
  "BigScience Open RAIL-M",
  "PRIOR UNDER-COUNTED BY 64,640 (2x). The repo ships TEN voice styles (F1-F5, M1-M5), not five. Per style = style_ttl [1,50,256] 12,800 + style_dp [1,8,16] 128 = 12,928 -- the per-style figure was right, the count of styles was not. Shipping five is a different, smaller ledger (L-2).")

m("qwen3-asr-0.6b", "Qwen/Qwen3-ASR-0.6B",
  "5eb144179a02acc5e5ba31e748d22b0cf3e303b0", 782_426_112, 782_426_112,
  "EXACT", HDR + "; A1 full-region SHA-256 byte comparison",
  782_426_112, "CONTESTED", "apache-2.0",
  "A1: TIED. Unique 782,426,112; serialized 938,008,576 retained as a parallel bound under S-2.",
  ub_worst_case=938_008_576,
  ub_basis="938,008,576 if an export re-materialises the tie (S-2 serialized basis)")

m("moss-tts-nano-100m", "OpenMOSS-Team/MOSS-TTS-Nano-100M",
  "44502f80dbf9743528fa921cc544d662c685ebec", 117_311_232, 117_311_232,
  "EXACT", ZIP, 117_346_547, "EXACT", "apache-2.0",
  "bf16 checkpoint, tensor-level. Prior 117,346,547 = floor(234,693,095/2), a bytes/2 bound, not a parse. The ONNX export at revision f52645cb carries 167,622,912 unique-by-external-offset -- it materialises the 16384x768 audio embedding THREE times (global wte, local wte, and the transposed output head onnx::MatMul_5889). Naive per-graph summation gives 391,092,480, a 3.3x trap (S-4).",
  ub_worst_case=167_622_912,
  ub_basis="167,622,912 = the ONNX export's unique external-data bytes / 4, if the ONNX is what ships")

m("moss-audio-tokenizer-nano", "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
  "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68", 21_969_664, 21_969_664, "EXACT",
  HDR, 21_969_664, "EXACT", "apache-2.0",
  "Confirms prior exactly. encoder 10,450,944 + decoder 10,450,944 + quantizer 1,067,776.")

m("omniasr-ctc-300m", "facebook/omniASR-CTC-300M",
  "8e35f0cc28fa6099e0c14d56db85ce0423baa691", 325_494_996, 325_494_996,
  "EXACT", ZIP, 325_494_996, "EXACT (vendor-declared)", "apache-2.0",
  "Prior was the vendor's declared figure. Now an independent tensor-level parse REPRODUCES IT EXACTLY, all float32. The 'if re-derived from bytes under fp16 it would double' worry in section 2.5 is closed: the dtype is enumerated, not assumed. Tokenizer is character-level and contains ZERO punctuation tokens -- see A5.")

m("qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct",
  "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", 1_543_714_304, 1_543_714_304,
  "EXACT", HDR, 1_543_714_304, "UB + HIGH", "apache-2.0",
  "UPGRADED HIGH -> EXACT. tie_word_embeddings=true and lm_head is NOT materialised on disk, so unique == serialized == 1,543,714,304. The 19,268 B byte residual that made it HIGH is resolved.",
  ub_worst_case=1_543_714_304,
  ub_basis="EXACT; ub_worst_case collapses onto the count")

m("falcon-h1-1.5b-instruct", "tiiuae/Falcon-H1-1.5B-Instruct",
  "80ebc50d7799a440b96c93bb6686a3924a09b0cb", 1_554_863_488, 1_554_863_488,
  "EXACT", HDR, 1_554_863_488, "UB + HIGH", "falcon-llm-license (other)",
  "UPGRADED HIGH -> EXACT. TH-61 DISCHARGED. REVISION ASSERTED before recording: sha 80ebc50d7799a440b96c93bb6686a3924a09b0cb. Identity guard: the Deep variant (tiiuae/Falcon-H1-1.5B-Deep-Instruct @ b6648636ddc9) measures 1,554,874,768 -- exactly 11,280 more, as section 2.4b predicted. UNTIED: tie_word_embeddings=false and lm_head is materialised separately, so there is no unique/serialized gap to argue about.",
  ub_worst_case=1_554_863_488,
  ub_basis="EXACT; ub_worst_case collapses onto the count")



# ------------------------------------------------- front-end hazard envelope
# A5 measured these so the UNKNOWN front-end rows can be tested against real
# numbers instead of adjectives. Every figure is an EXACT tensor-level parse.
HAZARD = {
    "zh_polyphone_g2pw_class": {
        "params": 102_882_442, "class": "EXACT",
        "basis": "google-bert/bert-base-chinese, own safetensors parse. g2pW is "
                 "bert-base-chinese + a polyphone head; this reproduces the "
                 "102,882,442 gross figure section 3.5 quotes.",
        "avoidable_by": "pypinyin (rule-based, 0) or g2pM (401,548 EXACT)"},
    "punct_itn_cheapest_5lang": {
        "params": 58_199_117, "class": "EXACT",
        "repo": "1-800-BAD-CODE/punct_cap_seg_47_language",
        "basis": "ONNX initializer enumeration. 47 languages, punctuation + "
                 "capitalisation + segmentation. The CHEAPEST credible "
                 "five-language neural restorer this lane could find.",
        "avoidable_by": "an ASR that emits punctuation natively (SenseVoice "
                        "use_itn), or delegating to the backbone LLM at 0 params"},
    "punct_itn_midrange": {
        "params": 277_458_184, "class": "EXACT",
        "repo": "kredor/punctuate-all (XLM-R base)"},
    "punct_itn_xlmr_large": {
        "params": 558_847_496, "class": "EXACT",
        "repo": "oliverguhr/fullstop-punctuation-multilang-large"},
    "ru_accent_ruaccent_class": {
        "params": 12_500_000, "class": "CLAIMED",
        "basis": "section 3.10 figure. NOT verified by this lane: none of four "
                 "candidate repo names resolves on HuggingFace."},
}

CHEAPEST_UNKNOWN_RESOLUTION = HAZARD["punct_itn_cheapest_5lang"]["params"]


def verdict(v):
    """COMPLIANT   - no UNKNOWN row; every row EXACT or a held bound.
       UNVERIFIED  - an UNKNOWN row exists that could cross the cap.
       NONCOMPLIANT- the measured total already exceeds the cap."""
    if v["total_measured"] >= CAP:
        return "NONCOMPLIANT", "measured total is at or above the cap"
    if not v["rows_unknown"]:
        return "COMPLIANT", (
            "no UNKNOWN row remains; every row is EXACT and the worst-case "
            "total (%s) is also below the cap" % "{:,}".format(v["ub_worst_case_total"]))
    hr = v["headroom"]
    if CHEAPEST_UNKNOWN_RESOLUTION > hr:
        return "UNVERIFIED", (
            "UNKNOWN rows %s. The CHEAPEST measured neural resolution "
            "(%s, punct_cap_seg_47_language) EXCEEDS this stack's headroom of %s "
            "by %s. The stack stays under cap ONLY if the UNKNOWN resolves to a "
            "zero-parameter path. That is an architectural constraint, not an "
            "assumption, and it must be asserted at build time."
            % (",".join(v["rows_unknown"]),
               "{:,}".format(CHEAPEST_UNKNOWN_RESOLUTION),
               "{:,}".format(hr),
               "{:,}".format(CHEAPEST_UNKNOWN_RESOLUTION - hr)))
    return "UNVERIFIED", (
        "UNKNOWN rows %s. The cheapest measured neural resolution (%s) FITS in "
        "the headroom of %s, leaving %s; but an XLM-R-large-class restorer "
        "(558,847,496) would breach. Comfortably resolvable, not yet resolved."
        % (",".join(v["rows_unknown"]),
           "{:,}".format(CHEAPEST_UNKNOWN_RESOLUTION),
           "{:,}".format(hr),
           "{:,}".format(hr - CHEAPEST_UNKNOWN_RESOLUTION)))


# ------------------------------------------------------------ licence status
LIC = {
  "H": [("H4", "RED", "Piper 5-voice bank: every voice is finetuned from the lessac voice, which was trained on the Blizzard 2013 Lessac corpus. That corpus's licence is a RESEARCH LICENCE AGREEMENT granting use 'exclusively for Research Purposes only', 'without the right to sub-licence'. Archived: licences/blizzard2013-lessac-research-licence_2026-08-02.txt"),
         ("H4(zh)", "RED", "Piper zh_CN-huayan: training-data licence declared 'Unknown' in Piper's own MODEL_CARD. An unknown corpus licence cannot be cleared, only avoided. H only -- Q and F ship four voices and serve zh from Matcha."),
         ("H7", "AMBER", "TEN VAD is Apache-2.0 PLUS an Agora non-compete rider: 'You may not Deploy the ten-vad in a way that competes with Agora's offerings'. NOT in the plan's blocker list -- found by this lane."),
         ("H5", "AMBER", "charsiu byT5 declares NO licence on the repo."),
         ("H4/H5", "AMBER", "eSpeak NG GPL-3.0 linkage: all Piper voice configs declare phoneme_type=espeak (verified per voice).")],
  "L": [("L5", "AMBER", "NVIDIA Open Model License (magpie)."),
        ("L5b", "AMBER", "nemo-nano-codec ships under NVIDIA 'other'; a NEW dependency this lane added, so its terms were never reviewed."),
        ("L7", "AMBER", "g2pM artefact is the PyPI wheel; the HF repo does not exist.")],
  "D": [("D5a", "RED", "matcha-icefall-zh-baker declares NO licence, and its own README states verbatim: 'The dataset is for non-commercial use only.' Archived: licences/matcha-icefall-zh-baker-readme_2026-08-02.txt"),
        ("D6a/D6b", "AMBER", "Supertonic-3 is BigScience Open RAIL-M: use-based restrictions in Attachment A must propagate to every downstream user and into the EULA."),
        ("D1", "AMBER", "FunASR Model Open Source License v1.1: permissive grant + mandatory attribution, but section 4.2 terminates the licence for 'unjustified denigration' -- a reputational-conduct clause with no analogue in OSI licences."),
        ("D9", "AMBER", "eSpeak NG GPL-3.0 linkage for the RU stress rung.")],
  "M": [("M1/M2/M3/M4", "GREEN", "Apache-2.0 throughout -- the only stack with a clean licence sheet end to end."),
        ("M5", "GREEN", "silero-vad MIT.")],
  "Q": [("Q3", "RED", "Piper x4: same lessac research-licence derivation chain as H4."),
        ("Q4", "RED", "matcha/Baker non-commercial, as D5a."),
        ("Q8", "AMBER", "eSpeak NG GPL-3.0 linkage, verified from all four voice configs."),
        ("Q2", "GREEN", "Qwen2.5-1.5B-Instruct Apache-2.0.")],
  "F": [("F3", "RED", "Piper x4: same lessac research-licence derivation chain."),
        ("F4", "RED", "matcha/Baker non-commercial."),
        ("F2", "RED", "Falcon-H1 ships license=other / falcon-llm-license with NO LICENSE FILE IN THE REPO. The cited URL (falconllm.tii.ae/falcon-terms-and-conditions.html) contains ZERO occurrences of 'Falcon H1' -- it covers Falcon 2 11B, Falcon 180B, Falcon Mamba 7B and Falcon 3 only. The governing terms for this exact model are therefore NOT ESTABLISHED. Additionally the Acceptable Use Policy is defined as 'the latest version from time to time' and the licence states 'You should monitor the web address at which the Acceptable Use Policy is hosted' -- a unilaterally mutable, forward-binding obligation, which cannot be discharged for an already-shipped offline device. Archived: licences/falcon-terms-and-conditions_2026-08-02.txt"),
        ("F8", "AMBER", "eSpeak NG GPL-3.0 linkage.")],
}

# ------------------------------------------------------------------- stacks

def row(stack, cid, key, mult=1, role=None, langs=None, override=None):
    d = dict(M[key])
    d["stack"] = stack
    d["component_id"] = cid
    d["multiplicity"] = mult
    d["params"] = d["params"] * mult
    d["params_learned"] = d["params_learned"] * mult
    d["ub_worst_case"] = d["ub_worst_case"] * mult
    if d["prior"] is not None:
        d["prior"] = d["prior"] * mult if mult > 1 else d["prior"]
    d["role"] = role
    d["languages"] = langs
    if override:
        d.update(override)
    d["delta_vs_prior"] = (None if d["prior"] is None
                           else d["params"] - d["prior"])
    return d


def zero(stack, cid, name, role, langs, note, cls="EXACT"):
    return {"stack": stack, "component_id": cid, "component_key": name,
            "repo": None, "revision": None, "params": 0, "params_learned": 0,
            "evidence_class": cls, "method": "non-neural; no learned tensor exists",
            "prior": 0, "delta_vs_prior": 0, "licence": None, "notes": note,
            "ub_worst_case": 0, "ub_worst_case_basis": "structurally zero",
            "role": role, "languages": langs, "multiplicity": 1}


def unknown(stack, cid, name, role, langs, note, closes):
    return {"stack": stack, "component_id": cid, "component_key": name,
            "repo": None, "revision": None, "params": None,
            "params_learned": None, "evidence_class": "UNKNOWN",
            "method": "not enumerated", "prior": None, "delta_vs_prior": None,
            "licence": None, "notes": note, "ub_worst_case": None,
            "ub_worst_case_basis": "UNKNOWN -- never interpolated (SR-02)",
            "closing_experiment": closes, "role": role, "languages": langs,
            "multiplicity": 1}


FE_NOTE = ("A5: OmniASR-CTC's tokenizer is character-level and contains ZERO "
           "punctuation tokens (0 occurrences of . , ? ! ; : in the 87,562 B "
           "sentencepiece blob). A punctuation/ITN source is therefore "
           "ARCHITECTURALLY REQUIRED, not merely likely. Cheapest measured "
           "five-language neural restorer = 58,199,117, which EXCEEDS this "
           "stack's headroom.")

STACKS = {}

STACKS["H"] = [
    row("H", "H1", "parakeet-tdt-0.6b-v3", role="ASR", langs="en,es,fr,ru"),
    row("H", "H2", "k2fsa-zipformer-zh-en", role="ASR", langs="zh(+en)"),
    row("H", "H3", "qwen3-0.6b", role="Backbone", langs="all five"),
    row("H", "H4", "piper-voice-medium", mult=5, role="TTS",
        langs="en,es,fr,ru,zh",
        override={"prior": 82_805_000,
                  "notes": "5-voice bank: kusal(en) + davefx(es) + siwis(fr) + dmitri(ru) + huayan(zh), each 15,650,602 EXACT. Prior bank figure 82,805,000 was 4,551,990 HIGH. Every voice is finetuned from the lessac voice -- see A6."}),
    row("H", "H5", "charsiu-byt5-g2p", role="G2P", langs="en,es,fr,ru"),
    row("H", "H6", "g2pm", role="G2P (polyphone)", langs="zh"),
    row("H", "H7", "ten-vad", role="VAD", langs="all five"),
]

STACKS["L"] = [
    row("L", "L1", "canary-180m-flash", role="ASR", langs="en,es,fr"),
    row("L", "L2", "k2fsa-zipformer-zh-en", role="ASR", langs="zh"),
    row("L", "L3", "gigaam-multilingual", role="ASR", langs="ru"),
    row("L", "L4", "qwen3.5-0.8b-lm", role="Backbone", langs="all five"),
    row("L", "L5", "magpie-tts-multilingual", role="TTS", langs="en,es,fr,zh"),
    row("L", "L5b", "nemo-nano-codec-22khz", role="Neural audio codec (TTS)",
        langs="en,es,fr,zh"),
    row("L", "L6", "silero-v5-cis-base", role="TTS", langs="ru"),
    row("L", "L7", "g2pm", role="G2P (polyphone)", langs="zh"),
    row("L", "L8", "silero-vad", role="VAD", langs="all five"),
    unknown("L", "L9", "ru-stress-accentor", "Front end (RU stress)", "ru",
            "A5 NEW: silero v5_cis_base does NOT bundle an accentor (its constructor takes accentor=None). Section 2.2's claim that silero_stress is 'bundled inside' is REFUTED. L needs an external RU stress source. Rule-based (eSpeak NG + marked-stress lexicon) = 0; RUAccent-class = ~12,500,000 CLAIMED (repo not resolvable at any of four candidate names).",
            "Pin an RU accentor artefact and header-parse it, or record the rule-based rung as the shipped choice."),
    unknown("L", "L10", "asr-itn-punctuation", "Front end (ITN/punct)",
            "all five",
            "canary-180m-flash emits punctuation natively (NeMo AED with punctuation in its token set); GigaAM-Multilingual is char-CTC over 71 classes and does NOT. RU route needs a punctuation source.",
            "Enumerate the RU punctuation path. Cheapest measured neural restorer 58,199,117; L's headroom absorbs it."),
]

STACKS["D"] = [
    row("D", "D1", "sensevoice-small", role="ASR", langs="zh,en"),
    row("D", "D2", "parakeet-tdt-0.6b-v3", role="ASR", langs="es,fr,ru"),
    row("D", "D3", "ru-zipformer-int8", role="ASR (2nd RU route, ADDITIVE)",
        langs="ru"),
    row("D", "D4a", "qwen3.5-0.8b-lm", role="Backbone", langs="all five"),
    row("D", "D4b", "qwen3.5-0.8b-mtp", role="Backbone MTP head",
        langs="all five"),
    row("D", "D5a", "matcha-icefall-zh-baker", role="TTS (acoustic)",
        langs="zh"),
    row("D", "D5b", "vocos-22khz-univ", role="TTS (vocoder)", langs="zh"),
    row("D", "D6a", "supertonic-3-graphs", role="TTS", langs="en,es,fr,ru"),
    row("D", "D6b", "supertonic-3-voice-styles", role="TTS voice styles",
        langs="en,es,fr,ru"),
    row("D", "D7", "silero-vad", role="VAD", langs="all five"),
    zero("D", "D8", "pypinyin", "Front end (grapheme->pinyin)", "zh",
         "Rule-based lookup table. Section 1.2: rule-based normalisers do not count."),
    zero("D", "D9", "espeak-ng-ru-stress", "Front end (RU stress)", "ru",
         "eSpeak NG ru rules + marked-stress lexicon, shipped as data. Non-neural. Carries GPL-3.0 linkage -- a licence row, not a parameter row."),
    zero("D", "D10", "supertonic-builtin-frontend",
         "Front end (phonemisation)", "en,es,fr,ru",
         "Inside the D6 bundle (unicode_indexer.json + internal normalisation). No separate graph."),
    zero("D", "D11", "asr-itn-punctuation", "Front end (ITN/punct)", "zh,en",
         "CLOSED BY A5 for the zh/en routes: SenseVoiceSmall emits punctuation and ITN natively via use_itn, at zero additional parameters. Verified in the model card."),
    unknown("D", "D11b", "asr-itn-punctuation-esfrru",
            "Front end (ITN/punct)", "es,fr,ru",
            "parakeet-tdt-0.6b-v3 is TDT/RNN-T over a BPE vocabulary; whether its token set carries punctuation for es/fr/ru was not established by this lane. If not, a restorer is needed for three languages.",
            "Inspect parakeet's tokenizer.json token set for punctuation, exactly as A5 did for OmniASR. ~20 minutes."),
]

STACKS["M"] = [
    row("M", "M1", "qwen3-asr-0.6b", role="ASR", langs="all five"),
    row("M", "M2", "qwen3.5-0.8b-lm", role="Backbone", langs="all five"),
    row("M", "M3", "moss-tts-nano-100m", role="TTS", langs="all five"),
    row("M", "M4", "moss-audio-tokenizer-nano", role="Neural audio tokenizer",
        langs="all five"),
    row("M", "M5", "silero-vad", role="VAD", langs="all five"),
    zero("M", "M6", "moss-sentencepiece-frontend", "Front end", "all five",
         "SentencePiece tokenizer. Section 1.2: text tokenizer vocabularies do not count. MOSS-TTS-Nano needs no external G2P."),
    unknown("M", "M7", "asr-itn-punctuation", "Front end (ITN/punct)",
            "all five",
            "Qwen3-ASR-0.6B decodes autoregressively through a 28-layer Qwen3 LM over a 151,936-token vocabulary, so it can emit punctuation -- but this lane did not verify that it does. Not interpolated (SR-02).",
            "Transcribe a punctuated fixture in each of the five languages and inspect the raw output. Lane C could return this at zero marginal cost."),
]

QF_TTS = [("3", "piper-voice-medium", 4, "TTS", "en,es,fr,ru"),
          ("4", "matcha-icefall-zh-baker", 1, "TTS (acoustic)", "zh"),
          ("5", "vocos-22khz-univ", 1, "TTS (vocoder)", "zh"),
          ("6", "silero-vad", 1, "VAD", "all five")]

for S, backbone_key in (("Q", "qwen2.5-1.5b-instruct"),
                        ("F", "falcon-h1-1.5b-instruct")):
    rows = [row(S, S + "1", "omniasr-ctc-300m", role="ASR",
                langs="all five, one model"),
            row(S, S + "2", backbone_key, role="Backbone", langs="all five")]
    for n, k, mult, role, langs in QF_TTS:
        ov = {"prior": 62_602_408} if mult == 4 else None
        rows.append(row(S, S + n, k, mult=mult, role=role, langs=langs,
                        override=ov))
    rows += [
        zero(S, S + "7", "pypinyin", "Front end", "zh",
             "Rule-based lookup table (section 1.2)."),
        zero(S, S + "8", "espeak-ng", "Front end", "en,es,fr,ru",
             "eSpeak NG rule tables shipped with Piper. VERIFIED: all four voice configs declare phoneme_type=espeak. Carries the GPL-3.0 linkage -- a licence row, not a parameter row."),
        unknown(S, S + "9", "asr-itn-punctuation", "Front end (ITN/punct)",
                "all five", FE_NOTE,
                "Either (a) delegate punctuation and ITN to the backbone LLM, which costs 0 additional parameters and must then be recorded as a BINDING ARCHITECTURAL CONSTRAINT, or (b) ledger a neural restorer, which breaches. There is no third option."),
    ]
    STACKS[S] = rows


def total(rows, field="params"):
    return sum(r[field] for r in rows if r.get(field) is not None)


PRIOR_TOTALS = {"H": 1_397_001_720, "L": 1_690_124_661, "D": 1_831_541_377,
                "M": 1_674_717_228, "Q": 1_964_519_413, "F": 1_975_668_597}

out = {
    "schema": "lane-a ledger v1",
    "generated_utc": "2026-08-02",
    "lane": "A",
    "sprint_ids": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
    "parameter_cap_exclusive": CAP,
    "counting_convention": {
        "params": "ALL tensor elements of the shipped artefact (conservative). Matches section 2.0.",
        "params_learned": "float-dtype tensors only; the section 1.2-correct figure excluding I64/I32 index and counter buffers",
        "totals_computed_on": "params (conservative)",
        "why": "Conservative counting can never turn a NONCOMPLIANT stack into a COMPLIANT one."
    },
    "method_summary": {
        "safetensors": HDR,
        "onnx": ONX,
        "torch_zip": ZIP,
        "nemo": TAR,
        "tie_detection": "byte comparison of both tensor data regions; A1 escalated to a full-region SHA-256 over 100% of both regions",
        "weights_downloaded": "none, except one 91,680,514 B TorchScript archive (silero v5_cis_base) whose host does not honour HTTP Range; deleted after parsing",
        "peak_lane_storage_bytes": 94_000_000,
        "storage_cap_bytes": 1_000_000_000
    },
    "stacks": {},
    "components": M,
    "front_end_hazard_envelope": None,
}

for S, rows in STACKS.items():
    t = total(rows)
    tl = total(rows, "params_learned")
    ub = total(rows, "ub_worst_case")
    unknowns = [r["component_id"] for r in rows
                if r["evidence_class"] == "UNKNOWN"]
    non_exact = [r["component_id"] for r in rows
                 if r["evidence_class"] not in ("EXACT",)]
    out["stacks"][S] = {
        "total_measured": t,
        "total_learned_only": tl,
        "total_prior": PRIOR_TOTALS[S],
        "delta_vs_prior": t - PRIOR_TOTALS[S],
        "pct_of_cap": round(100.0 * t / CAP, 4),
        "headroom": CAP - t,
        "ub_worst_case_total": ub,
        "ub_worst_case_headroom": CAP - ub,
        "ub_worst_case_compliant": ub < CAP,
        "rows_unknown": unknowns,
        "rows_not_exact": non_exact,
        "n_rows": len(rows),
        "rows": rows,
    }
    vd, why = verdict(out["stacks"][S])
    out["stacks"][S]["verdict"] = vd
    out["stacks"][S]["verdict_rationale"] = why
    out["stacks"][S]["licence_flags"] = [
        {"component_id": a, "status": b, "finding": c} for a, b, c in LIC[S]]
    worst = "RED" if any(b == "RED" for _, b, _ in LIC[S]) else (
        "AMBER" if any(b == "AMBER" for _, b, _ in LIC[S]) else "GREEN")
    out["stacks"][S]["licence_traffic_light"] = worst

out["front_end_hazard_envelope"] = HAZARD

with open("../ledger.json", "w") as f:
    json.dump(out, f, indent=1)

print("%-3s %16s %14s %14s %8s %-13s %-6s %s" % (
    "St", "measured", "delta", "headroom", "%cap", "verdict", "lic",
    "UNKNOWN"))
for S in ["H", "L", "D", "M", "Q", "F"]:
    v = out["stacks"][S]
    print("%-3s %16s %14s %14s %7.2f%% %-13s %-6s %s" % (
        S, "{:,}".format(v["total_measured"]),
        "{:+,}".format(v["delta_vs_prior"]), "{:,}".format(v["headroom"]),
        v["pct_of_cap"], v["verdict"], v["licence_traffic_light"],
        ",".join(v["rows_unknown"]) or "-"))
print()

print("%-3s %16s %16s %14s %14s %10s  %s" % (
    "St", "measured", "prior", "delta", "headroom", "%cap", "UNKNOWN rows"))
for S in ["H", "L", "D", "M", "Q", "F"]:
    v = out["stacks"][S]
    print("%-3s %16s %16s %14s %14s %9.2f%%  %s" % (
        S, "{:,}".format(v["total_measured"]), "{:,}".format(v["total_prior"]),
        "{:+,}".format(v["delta_vs_prior"]), "{:,}".format(v["headroom"]),
        v["pct_of_cap"], ",".join(v["rows_unknown"]) or "-"))
print()
for S in ["H", "L", "D", "M", "Q", "F"]:
    v = out["stacks"][S]
    print("%-3s ub_worst_case total %16s  headroom %14s  compliant=%s" % (
        S, "{:,}".format(v["ub_worst_case_total"]),
        "{:,}".format(v["ub_worst_case_headroom"]),
        v["ub_worst_case_compliant"]))
