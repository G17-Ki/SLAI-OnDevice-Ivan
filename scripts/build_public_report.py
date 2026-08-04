#!/usr/bin/env python3
"""Build the public Markdown and visually styled PDF feasibility report."""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#246BCE")
CYAN = colors.HexColor("#3DB7C5")
GREEN = colors.HexColor("#2E8B57")
AMBER = colors.HexColor("#D98E04")
RED = colors.HexColor("#C3423F")
INK = colors.HexColor("#20252B")
MUTED = colors.HexColor("#5D6874")
PALE = colors.HexColor("#F3F6F9")
GRID = colors.HexColor("#D8E0E8")


def load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def clean_component(name: str) -> str:
    return {
        "fastconformer": "FastConformer + adapter",
        "backbone-conv-prefill": "Conv layer prefill",
        "backbone-attention-prefill": "Attention layer prefill",
        "backbone-conv-cached-decode": "Conv cached decode",
        "backbone-attention-cached-decode": "Attention cached decode",
        "depth-decoder": "Depth/RQ decoder",
        "detokenizer-t4": "Detokenizer neural T=4",
        "detokenizer-t8": "Detokenizer neural T=8",
    }.get(name, name)


def clean_quant_config(name: str) -> tuple[str, str]:
    return {
        "f16_fa_on": ("FP16", "on"),
        "q8_fa_on": ("Q8_0", "on"),
        "q4_fa_on": ("Q4_0", "on"),
        "f16_fa_off": ("FP16", "off"),
    }.get(name, (name, "n/a"))


def median_p95(row: dict[str, Any], metric: str, digits: int = 1) -> str:
    med = row.get(f"{metric}_median")
    p95 = row.get(f"{metric}_p95")
    if med is None or p95 is None:
        return "n/a"
    return f"{float(med):.{digits}f} / {float(p95):.{digits}f}"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def architecture_drawing() -> Drawing:
    drawing = Drawing(500, 135)
    boxes = [
        (5, "VAD", GREEN),
        (80, "FastConformer\nNPU", BLUE),
        (180, "LFM backbone\nNPU shards", BLUE),
        (300, "Depth decoder\nNPU", BLUE),
        (405, "FP32 detokenizer\nproposed host", AMBER),
    ]
    widths = [55, 80, 100, 85, 90]
    for (x, label, color), width in zip(boxes, widths):
        drawing.add(Rect(x, 57, width, 42, rx=5, ry=5, fillColor=color, strokeColor=color))
        parts = label.split("\n")
        for index, part in enumerate(parts):
            drawing.add(
                String(
                    x + width / 2,
                    80 - index * 12,
                    part,
                    textAnchor="middle",
                    fillColor=colors.white,
                    fontName="Helvetica-Bold" if index == 0 else "Helvetica",
                    fontSize=8.2,
                )
            )
    for start, end in ((60, 80), (160, 180), (280, 300), (385, 405)):
        drawing.add(Line(start, 78, end, 78, strokeColor=NAVY, strokeWidth=1.5))
        drawing.add(Line(end - 5, 82, end, 78, strokeColor=NAVY, strokeWidth=1.5))
        drawing.add(Line(end - 5, 74, end, 78, strokeColor=NAVY, strokeWidth=1.5))
    drawing.add(Rect(178, 8, 205, 28, rx=4, ry=4, fillColor=PALE, strokeColor=GRID))
    drawing.add(String(280, 25, "Session manager: bounded context, reset/summary policy", textAnchor="middle", fontSize=8, fillColor=INK))
    drawing.add(Line(280, 36, 280, 57, strokeColor=MUTED, strokeDashArray=[3, 2]))
    drawing.add(String(250, 116, "Recommended LFM partition for the first device integration", textAnchor="middle", fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    return drawing


def latency_chart(rows: list[dict[str, Any]]) -> Drawing:
    labels = [clean_component(row["component"]) for row in rows]
    values = [float(row["latency_ms"]) for row in rows]
    drawing = Drawing(500, 245)
    chart = HorizontalBarChart()
    chart.x = 155
    chart.y = 28
    chart.height = 185
    chart.width = 300
    chart.data = [values]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.boxAnchor = "e"
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(values) * 1.15
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labelTextFormat = "%0.0f"
    chart.bars[0].fillColor = BLUE
    chart.bars[0].strokeColor = BLUE
    drawing.add(chart)
    drawing.add(String(310, 8, "Per-component latency on QCS8550 proxy (ms)", textAnchor="middle", fontSize=8, fillColor=MUTED))
    return drawing


def memory_chart(memory: dict[str, Any]) -> Drawing:
    totals = memory["totals"]
    values = [
        totals["bf16_downloaded_runtime_checkpoints_gib"],
        totals["q4_gguf_bundle_gib"],
        2.0,
    ]
    drawing = Drawing(500, 220)
    chart = VerticalBarChart()
    chart.x = 80
    chart.y = 45
    chart.height = 135
    chart.width = 340
    chart.data = [values]
    chart.categoryAxis.categoryNames = ["BF16 assets", "Q4 GGUF bundle", "Total device RAM"]
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 4.0
    chart.valueAxis.valueStep = 0.5
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = BLUE
    chart.bars[0].strokeColor = BLUE
    drawing.add(chart)
    drawing.add(String(250, 15, "Static storage reference in GiB; device bar is total RAM, not app budget", textAnchor="middle", fontSize=8, fillColor=MUTED))
    return drawing


def quality_chart(rows: list[dict[str, Any]]) -> Drawing | None:
    if not rows:
        return None
    conditions = ["clean", "gaussian_10db", "competing_speech_5db"]
    model_order = [
        "LiquidAI/LFM2.5-Audio-1.5B",
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "UsefulSensors/moonshine-base",
    ]
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        grouped.setdefault(row["model"], {})[row["condition"]] = float(row["wer_percent"])
    present = [model for model in model_order if model in grouped]
    if not present:
        return None
    data = [[grouped[model].get(condition, 0.0) for condition in conditions] for model in present]
    drawing = Drawing(500, 245)
    chart = VerticalBarChart()
    chart.x = 60
    chart.y = 55
    chart.height = 150
    chart.width = 390
    chart.data = data
    chart.categoryAxis.categoryNames = ["Clean", "White noise 10 dB", "Competing speech 5 dB"]
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(max(series) for series in data) * 1.15 + 1
    chart.valueAxis.labels.fontSize = 7
    palette = [NAVY, CYAN, GREEN, AMBER, RED]
    for index in range(len(present)):
        chart.bars[index].fillColor = palette[index]
        chart.bars[index].strokeColor = palette[index]
    drawing.add(chart)
    legend_y = 32
    short = {
        "LiquidAI/LFM2.5-Audio-1.5B": "LFM2.5",
        "openai/whisper-tiny": "Whisper Tiny",
        "openai/whisper-base": "Whisper Base",
        "openai/whisper-small": "Whisper Small",
        "UsefulSensors/moonshine-base": "Moonshine Base",
    }
    cursor = 28
    for index, model in enumerate(present):
        drawing.add(Rect(cursor, legend_y, 8, 8, fillColor=palette[index], strokeColor=palette[index]))
        drawing.add(String(cursor + 11, legend_y, short[model], fontSize=7, fillColor=INK))
        cursor += 82
    drawing.add(String(250, 10, "WER on the same 18-utterance diagnostic; lower is better", textAnchor="middle", fontSize=8, fillColor=MUTED))
    return drawing


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=26, leading=30, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontSize=12, leading=17, textColor=MUTED, spaceAfter=18),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=8, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=7, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13.3, textColor=INK, spaceAfter=6),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.6, leading=10.4, textColor=MUTED, spaceAfter=4),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10.5, leading=15, textColor=NAVY, backColor=colors.HexColor("#EAF2FB"), borderColor=BLUE, borderWidth=0.8, borderPadding=9, spaceBefore=6, spaceAfter=10),
        "covermeta": ParagraphStyle("CoverMeta", parent=base["Normal"], fontSize=9, leading=13, textColor=MUTED),
        "ref": ParagraphStyle("Ref", parent=base["BodyText"], fontSize=7.4, leading=10.2, textColor=INK, leftIndent=10, firstLineIndent=-10, spaceAfter=3),
    }


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs: Any) -> None:
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=self._header_footer))

    def _header_footer(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(GRID)
            canvas.line(self.leftMargin, A4[1] - 18 * mm, A4[0] - self.rightMargin, A4[1] - 18 * mm)
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(MUTED)
            canvas.drawString(self.leftMargin, A4[1] - 14 * mm, "LFM2.5-Audio on AI Glasses - Feasibility Report")
            canvas.drawRightString(A4[0] - self.rightMargin, 11 * mm, f"Page {doc.page}")
        canvas.restoreState()


def table(data: list[list[Any]], widths: list[float], font_size: float = 7.2, header: bool = True) -> Table:
    header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=font_size,
        leading=font_size + 2,
        textColor=colors.white,
    )
    body_style = ParagraphStyle(
        "TableBody",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 2,
        textColor=INK,
    )
    wrapped = []
    for row_index, row in enumerate(data):
        wrapped.append(
            [
                Paragraph(
                    html.escape(str(cell)),
                    header_style if header and row_index == 0 else body_style,
                )
                for cell in row
            ]
        )
    value = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    value.setStyle(TableStyle(commands))
    return value


def build_markdown(data: dict[str, Any]) -> str:
    strict = data["strict_rows"]
    qrows = data["quality_rows"]
    by_model_condition = {(r["model"], r["condition"]): r for r in qrows}
    quant_summary = data.get("quant", {}).get("summary", [])
    lines = [
        "# LFM2.5-Audio on AI Glasses: Feasibility, Quality, and Baseline Decision",
        "",
        f"**Public technical report - {data['date']}**",
        "",
        "## Executive decision",
        "",
        "Proceed with LFM2.5-Audio as the integrated research baseline, but do not yet call it glasses-ready. The tested fixed-shape neural partitions compile and execute entirely on the Hexagon NPU of a Qualcomm QCS8550 proxy. The full BF16 model is functionally healthy on an NVIDIA L4, including persistent two-turn speech interaction. The remaining blockers are target-hardware verification, target-QNN resident memory, a production cache interface, and a numerically safe output-audio path.",
        "",
        "Run a second, modular baseline in parallel as deployment insurance: Moonshine Base ASR -> Qwen3-0.6B in non-thinking mode -> Pocket TTS. If Moonshine or Pocket export stalls, use the Qualcomm-native Zipformer/Piper components as the conservative fallback. This challenger is not yet a quality winner; it is the best footprint and toolchain control.",
        "",
        "## Product mode",
        "",
        "The confirmed mode is VAD-gated activation followed by a continuous conversational session. VAD lowers idle compute, but it does not bound memory during an active session. The runtime therefore needs a deliberate context reset, summarization, or eviction policy.",
        "",
        "## Recommended LFM partition",
        "",
        "```text",
        "Microphone -> VAD -> FastConformer on NPU -> LFM backbone on NPU",
        "           -> depth/RQ decoder on NPU -> FP32 detokenizer outside FP16 HTP path -> speaker",
        "```",
        "",
        "The FP32 detokenizer placement is a proposed partition. CPU-host latency and memory have not yet been measured on the glasses.",
        "",
        "## What was measured",
        "",
        "- Eight official-weight fixed-shape PT2 graphs exported and reloaded locally.",
        "- Strict QNN HTP FP16 placement, component latency, tool-reported peak memory, and numerical comparison on QCS8550 (Proxy), Android 12, Hexagon v73.",
        "- Full official BF16 ASR and two-turn interleaved speech interaction on one NVIDIA L4.",
        "- A small matched ASR diagnostic under clean, 10 dB white noise, and 5 dB competing speech conditions.",
        "- Repeated official Q4_0 GGUF inference with FP16, Q8_0, and Q4_0 KV caches plus flash-attention on/off controls on Apple M2 Metal.",
        "- Static checkpoint and cache accounting.",
        "",
        "No exact AR1, AR1+, or unambiguous 5100 target was available in this AI Hub account. Proxy latency is not AR1 latency.",
        "",
        "## Qualcomm proxy component results",
        "",
        "| Component | NPU / CPU | Latency | Peak | Numerical result |",
        "|---|---:|---:|---:|---|",
    ]
    for row in strict:
        comparison = "passed"
        if row["status"] == "numerical_mismatch":
            comparison = "mismatch"
        lines.append(
            f"| {clean_component(row['component'])} | {row['npu_runtime_layers']} / {row['cpu_fallback_runtime_layers']} | {row['latency_ms']:.3f} ms | {row['peak_memory_mb']:.1f} MiB | {comparison} |"
        )
    lines.extend(
        [
            "",
            "All tested strict graphs placed with zero CPU fallback on the proxy. This is not a claim that the complete model is fully NPU placed. The full 16-layer backbone, orchestration, complete waveform reconstruction, VAD, and audio I/O were not submitted as one graph.",
            "",
            "## Quality findings",
            "",
            "The FastConformer proxy output has cosine similarity 0.998424 and NRMSE 0.06976 against the local golden. All ten frames preserve their nearest temporal identity. A frozen downstream LFM check preserves 16 of 17 top-1 choices; the only change is the semantically similar choice `seems` -> `looks`. The golden choice remains in the remote top five at every step. This is encouraging but proves that feature cosine alone is not an acceptance test.",
            "",
            "The strict-NPU depth decoder returns exact audio code tokens. By contrast, the FP16 NPU detokenizer is numerically unusable despite full placement. A repeat with the first eight real generated audio frames from turn 1 gives waveform cosine 0.0052, NRMSE 1.010, and SI-SDR -45.74 dB. The synthetic-probe failure was therefore not an input artifact. The report keeps the detokenizer outside the first FP16 HTP partition.",
            "",
        ]
    )
    if qrows:
        lines.extend(
            [
                "### Small matched ASR diagnostic",
                "",
                "WER is normalized, lower is better. This 18-utterance subset is a smoke diagnostic, not a publication-scale benchmark.",
                "",
                "| Model | Clean | White noise 10 dB | Competing speech 5 dB |",
                "|---|---:|---:|---:|",
            ]
        )
        for model, label in (
            ("LiquidAI/LFM2.5-Audio-1.5B", "LFM2.5-Audio"),
            ("openai/whisper-tiny", "Whisper Tiny"),
            ("openai/whisper-base", "Whisper Base"),
            ("openai/whisper-small", "Whisper Small"),
            ("UsefulSensors/moonshine-base", "Moonshine Base"),
        ):
            vals = [by_model_condition.get((model, condition)) for condition in ("clean", "gaussian_10db", "competing_speech_5db")]
            if any(vals):
                lines.append("| " + label + " | " + " | ".join(f"{float(v['wer_percent']):.2f}%" if v else "n/a" for v in vals) + " |")
        lines.extend(
            [
                "",
                "On this small test, LFM is strongest on clean speech and 10 dB white noise, while Whisper Small is clearly strongest under the synthetic competing-speaker condition. Moonshine is a footprint challenger, not a demonstrated multi-speaker quality winner.",
                "",
            ]
        )
    intelligibility = data.get("lfm_output_intelligibility", [])
    if intelligibility:
        lines.extend(
            [
                "### Generated-audio intelligibility proxy",
                "",
                f"A frozen Whisper Small retranscription gives {intelligibility[0]['wer_percent']:.2f}% WER for turn 1 and {intelligibility[1]['wer_percent']:.2f}% WER for turn 2 against LFM's own text stream. This measures audio-text consistency, not naturalness or MOS.",
                "",
            ]
        )
    memory = data["memory"]
    lines.extend(
        [
            "## Memory and context",
            "",
            f"The pinned BF16 runtime checkpoint set occupies {memory['totals']['bf16_downloaded_runtime_checkpoints_gib']:.3f} GiB in static files. The complete local Q4 GGUF bundle occupies {memory['totals']['q4_gguf_bundle_gib']:.3f} GiB, a 3.39x smaller package-to-package comparison. The formats and backends differ; neither number is total resident application memory.",
            "",
            "The LFM backbone has six attention layers and ten convolution layers. The idealized FP16 KV payload is 48 MiB at 4,096 positions, 96 MiB at 8,192, 192 MiB at 16,384, and 384 MiB at 32,768; idealized INT8 payload halves those values. Actual runtime buffers can differ because of scales and alignment. The convolution cache is small and fixed; the attention cache is the active-session growth risk.",
            "",
        ]
    )
    if quant_summary:
        total_runs = sum(int(row.get("runs", 0)) for row in quant_summary)
        exact_runs = sum(int(row.get("exact_match_runs", 0)) for row in quant_summary)
        lines.extend(
            [
                "## Local Q4 inference-technique matrix",
                "",
                "The official Q4_0 GGUF bundle was repeated on two official audio samples using Apple M2 Metal. This is a two-sample functional sanity check, not Qualcomm QNN/AR1 evidence or a general ASR benchmark. CLI model time excludes model load; wall time includes process startup and file-cache effects. Five-run p95 values are descriptive.",
                "",
                f"Across {total_runs} runs, {exact_runs}/{total_runs} normalized transcripts match their references exactly.",
                "",
                "| KV cache | Flash | Sample | Exact | WER med/p95 | KV MiB | Encode ms med/p95 | Gen tok/s med/p95 | Model ms med/p95 | Wall ms med/p95 |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in quant_summary:
            cache, flash = clean_quant_config(row["config"])
            lines.append(
                f"| {cache} | {flash} | {row['audio']} | {row['exact_match_runs']}/{row['runs']} | {median_p95(row, 'wer_percent', 2)}% | {fmt(row.get('kv_buffer_mib_median'), 2)} | {median_p95(row, 'audio_encode_ms')} | {median_p95(row, 'generation_tokens_per_second')} | {median_p95(row, 'total_model_ms')} | {median_p95(row, 'wall_ms')} |"
            )
        lines.extend(
            [
                "",
                "The runner reports Metal fallback hotspots in the audio encoder (`CONV_2D_DW`, `ROLL`, `UNARY`). These are Apple Metal implementation gaps, not QNN CPU-fallback evidence. No post-training-quantized LFM QNN graph has yet been profiled on QCS8550 or the physical glasses.",
                "",
            ]
        )
    lines.extend(
        [
            "## Additional Qualcomm speech profiles",
            "",
            "Qualcomm's `qai-hub-models==0.57.3` scorecards provide a useful deployment control on the same QCS8550 proxy. Every component below reports full NPU placement.",
            "",
            "| Model | Encoder / main stage | Recurrent stage | Interpretation |",
            "|---|---:|---:|---|",
            "| Zipformer | 8.895 ms per 0.71 s chunk | decoder 0.075 ms; joiner 0.187 ms | Strong streaming fallback |",
            "| Whisper Tiny | 25.307 ms | 2.459 ms per decoder call | Smallest Whisper control |",
            "| Whisper Base | 47.978 ms | 4.202 ms per decoder call | Middle footprint/quality control |",
            "| Whisper Small FP16 | 130.318 ms | 12.074 ms per decoder call | Best overlap robustness in the small quality test |",
            "| Whisper Small W8A16 | 376.895 ms | 7.856 ms per decoder call | Lower decoder cost, but 2.9x slower encoder |",
            "| Piper TTS | encoder 30.344 ms; flow 15.189 ms | decoder 3.018 ms per chunk | Conservative output-audio fallback |",
            "",
            "These are per-component fixed-shape scorecard values, not end-to-end pipeline latency. Whisper Small W8A16 makes the encoder 2.89x slower while making the decoder about 35% faster, so quantization must be profiled stage by stage.",
            "",
            "## Candidate decision",
            "",
            "| Candidate | Role | Weight view | Deployment evidence | Decision |",
            "|---|---|---:|---|---|",
            "| LFM2.5-Audio-1.5B | Integrated research baseline | Q4 bundle 1.001 GiB locally | Strongest measured partition evidence; detokenizer blocker | Continue |",
            "| Moonshine Base + Qwen3-0.6B + Pocket TTS | Best next footprint challenger | Estimated 0.55-0.66 GB weights | Qwen has official Qualcomm package; ASR/TTS export pending | Profile next |",
            "| Zipformer + Qwen3-0.6B + Piper TTS | Conservative toolchain fallback | About 0.84 GB from Qualcomm component size cards | Zipformer and Piper are fully NPU placed on QCS8550 cards; Qwen QCS8550 result absent | Demo insurance |",
            "| Mini-Omni | End-to-end scientific control | About 976M total estimated | Custom scheduler and SNAC export risk | Later control |",
            "| LLaMA-Omni2-0.5B | Misleading size label | Public checkpoint alone 3.857 GB BF16 | Multiple external components | Exclude |",
            "",
            "## Exact next steps",
            "",
            "1. Confirm the physical glasses SoC, available app RAM, QNN/Voice AI SDK version, and deployment interface.",
            "2. Run the modular challenger export in this order: Qwen3-0.6B, Moonshine Base encoder/decoder, then Pocket TTS host path. Keep Zipformer/Piper ready as swaps.",
            "3. Build four-layer LFM backbone shards and a production cached decoder with explicit position/mask handling and a bounded context policy.",
            "4. Convert and profile target-QNN quantized encoder/backbone/depth variants separately. Validate downstream tokens/transcripts after every precision change.",
            "5. Measure the official FP32 detokenizer on the intended host CPU/DSP path. Do not move it to FP16 HTP until real-code audio fidelity passes.",
            "6. Integrate VAD and measure microphone-end to first audible PCM, total process RSS, 30-minute temperature, and battery drop on the real glasses.",
            "",
            "## Acceptance gates for a physical-glasses demo",
            "",
            "- Provisional model-process peak at or below 1.3 GB, to be revised after confirming app-available RAM and device reserve.",
            "- No unsupported main-backbone operation and no undeclared CPU fallback in NPU shards.",
            "- Quantized ASR WER increase no more than 0.3 percentage points on each chosen evaluation split.",
            "- First audible PCM within 500 ms after VAD close on the actual device.",
            "- Two-turn context retention no more than five points below the BF16 golden test set.",
            "- No NaN/Inf, systematic truncation, or material audio-text consistency regression.",
            "",
            "## Limitations",
            "",
            "- No exact AR1/AR1+/5100 AI Hub target was available.",
            "- Component latency and memory cannot be summed into an end-to-end estimate.",
            "- Backbone placement covers representative official layers and fixed cache probes, not one complete compiled network.",
            "- The small quality suite uses synthetic noise and a dummy LibriSpeech subset, not glasses microphone recordings.",
            "- Host FP32 detokenizer viability remains proposed, not measured on target.",
            "- The Q4 matrix uses two official samples on Apple M2 Metal; it does not establish general quality, Qualcomm placement, or glasses memory.",
            "- LFM weights use the LFM Open License v1.0, including a commercial-use revenue threshold; downstream components carry their own licenses and attribution duties.",
            "",
            "## Reproducibility",
            "",
            "- Model: `LiquidAI/LFM2.5-Audio-1.5B` revision `c362a0625dfe45aa588dce5f0ada28a7e5707628`.",
            "- Q4 GGUF: `LiquidAI/LFM2.5-Audio-1.5B-GGUF` revision `7d525f883a077e20afb782f2ff618edcae0e39e4`; runner build 7641 (`68d8edf2`), Apple M2 Metal.",
            "- Liquid Audio source commit: `19e65845923a7f136442c95137884ec61eb386aa`.",
            "- QCS8550 proxy: Android 12, Hexagon v73, QNN HTP FP16.",
            "- Qualcomm comparator catalog: `qai-hub-models==0.57.3`, QAIRT 2.45.0 scorecards.",
            "- Full golden: PyTorch 2.8.0, Transformers 4.56.1, NVIDIA L4 BF16.",
            "",
            "## References",
            "",
            "1. Liquid AI, LFM2.5-Audio model card: https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B",
            "2. Liquid Audio source: https://github.com/Liquid4All/liquid-audio",
            "3. Qualcomm AI Hub model catalog: https://aihub.qualcomm.com/models/",
            "4. Qualcomm Zipformer: https://aihub.qualcomm.com/models/zipformer",
            "5. Qualcomm Whisper Small: https://aihub.qualcomm.com/models/whisper_small",
            "6. Qualcomm PiperTTS English: https://aihub.qualcomm.com/models/pipertts_en",
            "7. Moonshine Base: https://huggingface.co/UsefulSensors/moonshine-base",
            "8. Moonshine paper: https://arxiv.org/abs/2410.15608",
            "9. Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B",
            "10. Qualcomm Qwen3-0.6B: https://aihub.qualcomm.com/models/qwen3_0_6b",
            "11. Pocket TTS: https://github.com/kyutai-labs/pocket-tts",
            "12. Mini-Omni: https://arxiv.org/abs/2408.16725",
            "13. LLaMA-Omni2: https://arxiv.org/abs/2505.02625",
        ]
    )
    return "\n".join(lines) + "\n"


def build_pdf(output: Path, data: dict[str, Any]) -> None:
    styles = make_styles()
    doc = ReportDoc(
        str(output),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=23 * mm,
        bottomMargin=18 * mm,
        title="LFM2.5-Audio on AI Glasses: Feasibility, Quality, and Baseline Decision",
        author="Ivan A. Novosad",
        subject="On-device speech model deployment feasibility",
    )
    story: list[Any] = []

    story.extend(
        [
            Spacer(1, 18 * mm),
            Paragraph("LFM2.5-Audio on AI Glasses", styles["title"]),
            Paragraph("Feasibility, Quality, and Baseline Decision", styles["subtitle"]),
            Spacer(1, 8 * mm),
            architecture_drawing(),
            Spacer(1, 9 * mm),
            Paragraph("Public technical report", styles["covermeta"]),
            Paragraph(data["date"], styles["covermeta"]),
            Paragraph("Ivan A. Novosad", styles["covermeta"]),
            Spacer(1, 14 * mm),
            Paragraph(
                "Decision: continue LFM as the integrated research baseline, while profiling a sub-1B modular pipeline as deployment control and demo insurance.",
                styles["callout"],
            ),
            Paragraph(
                "Hardware boundary: Qualcomm AI Hub exposed QCS8550 (Proxy), Android 12, Hexagon v73. No AR1/AR1+/5100 target was available. All proxy figures are placement evidence, not AR1 performance.",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    story.append(Paragraph("1. Executive decision", styles["h1"]))
    story.append(Paragraph("Proceed with LFM2.5-Audio as the integrated research baseline, but do not call it glasses-ready yet. All tested fixed-shape neural partitions compiled to strict NPU contexts with zero CPU fallback on the QCS8550 proxy. The full official BF16 model also completed ASR and persistent two-turn speech interaction on NVIDIA L4.", styles["body"]))
    story.append(Paragraph("The main blockers are target-hardware verification, target-QNN resident memory, a production cache interface, and the output-audio path. The FP16 NPU detokenizer fails numerically despite full placement. The first device integration should keep the official FP32 detokenizer outside the FP16 HTP path until a host implementation is measured and accepted.", styles["body"]))
    story.append(Paragraph("In parallel, profile Moonshine Base -> Qwen3-0.6B -> Pocket TTS. It is the strongest weight-footprint challenger. Keep Qualcomm-native Zipformer and Piper TTS as conservative swaps if export or host integration becomes the schedule risk.", styles["callout"]))

    story.append(Paragraph("2. Confirmed usage mode", styles["h1"]))
    story.append(Paragraph("The product mode is VAD-gated activation followed by a continuous conversational session. This avoids constant full-model inference while idle, but it does not bound active-session context. The session manager needs an explicit reset, summarization, or eviction policy.", styles["body"]))
    story.append(architecture_drawing())
    story.append(Paragraph("The proposed host detokenizer is an architectural recommendation, not a measured CPU-host result on the glasses.", styles["small"]))

    story.append(Paragraph("3. Experiment scope", styles["h1"]))
    scope = [
        ["Evidence class", "What is included"],
        ["Local", "Eight official-weight fixed-shape PT2 exports; repeated Q4_0/KV-cache matrix on Apple M2 Metal; static memory ledger."],
        ["Qualcomm proxy", "Strict NPU placement, component-only latency/memory, and local-vs-remote numerical comparison."],
        ["Reference GPU", "Full BF16 ASR plus two-turn interleaved text/audio generation on one NVIDIA L4."],
        ["Quality", "Downstream encoder sensitivity, exact depth tokens, generated-wave integrity, and a small matched ASR diagnostic."],
        ["Not measured", "Full AR1 pipeline, host/NPU transfers, real glasses RSS, power, thermals, VAD, microphone-to-speaker latency."],
    ]
    story.append(table(scope, [34 * mm, 126 * mm], font_size=7.6))

    story.append(PageBreak())
    story.append(Paragraph("4. Qualcomm proxy component results", styles["h1"]))
    story.append(Paragraph("Every tested strict graph placed fully on the Hexagon NPU of QCS8550 (Proxy). Placement and numerical acceptance are separate: the detokenizer demonstrates that a graph can be fully placed and still be unusable.", styles["body"]))
    strict = data["strict_rows"]
    component_table = [["Component", "NPU/CPU", "Latency", "Peak", "Numerics"]]
    for row in strict:
        component_table.append(
            [
                clean_component(row["component"]),
                f"{row['npu_runtime_layers']}/{row['cpu_fallback_runtime_layers']}",
                f"{row['latency_ms']:.3f} ms",
                f"{row['peak_memory_mb']:.1f} MiB",
                "Pass" if row["status"] == "passed" else "Mismatch",
            ]
        )
    story.append(table(component_table, [54 * mm, 20 * mm, 24 * mm, 24 * mm, 28 * mm], font_size=7.0))
    story.append(Spacer(1, 5 * mm))
    story.append(latency_chart(strict))
    story.append(Paragraph("Do not add these latency or peak-memory values into a full-system estimate. The components have different shapes, repeated invocation counts, compiled-context overheads, and transfer boundaries.", styles["small"]))

    story.append(Paragraph("5. Functional full-model golden", styles["h1"]))
    bf16 = data["bf16"]
    full_table = [
        ["Metric", "Turn 1", "Turn 2"],
        ["First text", f"{bf16['interleaved']['turn1']['first_text_seconds']*1000:.2f} ms", f"{bf16['interleaved']['turn2']['first_text_seconds']*1000:.2f} ms"],
        ["First audio token", f"{bf16['interleaved']['turn1']['first_audio_seconds']*1000:.2f} ms", f"{bf16['interleaved']['turn2']['first_audio_seconds']*1000:.2f} ms"],
        ["Generated steps/s", f"{bf16['interleaved']['turn1']['generated_steps_per_second']:.2f}", f"{bf16['interleaved']['turn2']['generated_steps_per_second']:.2f}"],
        ["Decoded waveform", f"{bf16['interleaved']['decode']['turn1']['waveform_seconds']:.2f} s", f"{bf16['interleaved']['decode']['turn2']['waveform_seconds']:.2f} s"],
    ]
    story.append(table(full_table, [54 * mm, 45 * mm, 45 * mm], font_size=7.5))
    story.append(Paragraph(f"The full model used {bf16['load']['memory']['allocated_bytes']/1e9:.03f} GB steady CUDA allocation and {bf16['load']['memory']['peak_allocated_bytes']/1e9:.03f} GB peak allocation. These are reference-GPU allocator figures, not mobile RSS. The official output detokenizer remained FP32 on the L4 GPU.", styles["body"]))

    story.append(PageBreak())
    story.append(Paragraph("6. Quality findings", styles["h1"]))
    story.append(Paragraph("FastConformer proxy features: cosine 0.998424, NRMSE 0.06976, and 100% nearest-frame temporal identity over the ten captured frames. A frozen downstream LFM sensitivity test preserves 16/17 top-1 decisions. The only branch change is 'seems' to 'looks'; the golden choice stays in the remote top five for every step.", styles["body"]))
    story.append(Paragraph("Interpretation: the encoder path is promising enough for the baseline, but exact tensor closeness is not the final gate. Every quantization change must be checked on transcripts, tokens, and task-level behavior.", styles["callout"]))
    real_detok = data.get("real_detok", {})
    if real_detok:
        wave = real_detok["waveform"]
        story.append(Paragraph(f"The depth/RQ decoder returns exact audio code tokens. The FP16 NPU detokenizer remains unusable on real generated codes: waveform cosine {wave['cosine_similarity']:.4f}, NRMSE {wave['normalized_rmse']:.3f}, and SI-SDR {wave['si_sdr_db']:.2f} dB for the first eight valid turn-1 frames. The graph still places 371/371 runtime layers on NPU at 2.564 ms, proving that placement alone is insufficient.", styles["body"]))
    else:
        story.append(Paragraph("The depth/RQ decoder returns exact audio code tokens. The tested FP16 NPU detokenizer does not preserve the waveform and remains outside the recommended FP16 HTP path.", styles["body"]))

    qchart = quality_chart(data["quality_rows"])
    if qchart is not None:
        story.append(Paragraph("Small matched ASR diagnostic", styles["h2"]))
        story.append(Paragraph("The same 18 LibriSpeech dummy utterances were tested clean, with deterministic 10 dB white noise, and with 5 dB competing speech. WER is normalized and lower is better. This is a smoke diagnostic, not a product benchmark.", styles["body"]))
        story.append(qchart)
        qdata = [["Model", "Clean", "White 10 dB", "Competing 5 dB"]]
        mapping = {(r["model"], r["condition"]): r for r in data["quality_rows"]}
        for model, label in (
            ("LiquidAI/LFM2.5-Audio-1.5B", "LFM2.5"),
            ("openai/whisper-tiny", "Whisper Tiny"),
            ("openai/whisper-base", "Whisper Base"),
            ("openai/whisper-small", "Whisper Small"),
            ("UsefulSensors/moonshine-base", "Moonshine Base"),
        ):
            values = [mapping.get((model, condition)) for condition in ("clean", "gaussian_10db", "competing_speech_5db")]
            if any(values):
                qdata.append([label, *[f"{float(value['wer_percent']):.2f}%" if value else "n/a" for value in values]])
        story.append(table(qdata, [48 * mm, 31 * mm, 31 * mm, 31 * mm], font_size=7.3))
        story.append(Paragraph("On this small diagnostic, LFM is strongest on clean speech and 10 dB white noise, while Whisper Small is clearly strongest under competing speech. Moonshine remains the footprint challenger, not a demonstrated multi-speaker quality winner.", styles["body"]))
    intelligibility = data.get("lfm_output_intelligibility", [])
    if intelligibility:
        story.append(Paragraph("Generated-audio intelligibility proxy", styles["h2"]))
        story.append(
            Paragraph(
                f"Whisper Small retranscription gives {intelligibility[0]['wer_percent']:.2f}% WER for turn 1 and {intelligibility[1]['wer_percent']:.2f}% WER for turn 2 against LFM's own generated text. The WAVs are finite, 24 kHz mono, and unclipped. This is an audio-text consistency proxy, not a naturalness or MOS score.",
                styles["body"],
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("7. Memory and context budget", styles["h1"]))
    story.append(memory_chart(data["memory"]))
    mem = data["memory"]
    story.append(Paragraph(f"The pinned BF16 runtime checkpoint set occupies {mem['totals']['bf16_downloaded_runtime_checkpoints_gib']:.3f} GiB in static files. The complete local Q4 GGUF bundle occupies {mem['totals']['q4_gguf_bundle_gib']:.3f} GiB, a 3.39x smaller package-to-package comparison across different formats and backends. Static file size is not process RSS; the BF16 assets cannot all be resident inside a 2 GiB RAM budget.", styles["body"]))
    cache_table = [
        ["Context positions", "Ideal FP16 payload", "Ideal INT8 payload"],
        ["4,096", "48 MiB", "24 MiB"],
        ["8,192", "96 MiB", "48 MiB"],
        ["16,384", "192 MiB", "96 MiB"],
        ["32,768", "384 MiB", "192 MiB"],
    ]
    story.append(table(cache_table, [55 * mm, 45 * mm, 45 * mm], font_size=7.5))
    story.append(Paragraph("The fixed convolution cache is only about 0.12 MiB at BF16. Active-session growth is dominated by the six attention-layer KV caches. These are tensor-payload estimates; actual quantized runtime buffers can differ because of scales and alignment. VAD reduces idle work but does not remove this requirement.", styles["body"]))

    quant_summary = data.get("quant", {}).get("summary", [])
    if quant_summary:
        story.append(PageBreak())
        story.append(Paragraph("8. Local Q4 inference-technique matrix", styles["h1"]))
        total_runs = sum(int(row.get("runs", 0)) for row in quant_summary)
        exact_runs = sum(int(row.get("exact_match_runs", 0)) for row in quant_summary)
        story.append(Paragraph(f"The official Q4_0 GGUF bundle was run {total_runs} times across two official audio samples with FP16, Q8_0, and Q4_0 KV caches and a flash-attention control. {exact_runs}/{total_runs} normalized transcripts match exactly. This is an Apple M2 Metal sanity check, not Qualcomm QNN/AR1 evidence or a general quality benchmark.", styles["body"]))
        quant_table = [["KV", "Flash", "Sample", "Exact", "KV MiB", "Encode ms med/p95", "Gen tok/s med/p95", "Model ms med/p95"]]
        for row in quant_summary:
            cache, flash = clean_quant_config(row["config"])
            quant_table.append(
                [
                    cache,
                    flash,
                    row["audio"],
                    f"{row['exact_match_runs']}/{row['runs']}",
                    fmt(row.get("kv_buffer_mib_median"), 1),
                    median_p95(row, "audio_encode_ms"),
                    median_p95(row, "generation_tokens_per_second"),
                    median_p95(row, "total_model_ms"),
                ]
            )
        story.append(table(quant_table, [16 * mm, 14 * mm, 20 * mm, 15 * mm, 17 * mm, 28 * mm, 28 * mm, 28 * mm], font_size=6.1))
        story.append(Paragraph("Values are median/p95 over five runs per sample and configuration. CLI model time excludes model load; wall time in the public data includes process startup and file-cache effects. The runner reports Metal fallback hotspots for CONV_2D_DW, ROLL, and UNARY in the audio encoder. Those are not QNN CPU-fallback evidence. No quantized LFM QNN graph has yet been profiled on QCS8550 or the glasses.", styles["small"]))

    story.append(Paragraph("9. Additional Qualcomm speech profiles", styles["h1"]))
    comparator = [
        ["Model", "Encoder/main", "Recurrent", "Use"],
        ["Zipformer", "8.895 ms / 0.71 s chunk", "0.075 ms decoder; 0.187 ms joiner", "Streaming fallback"],
        ["Whisper Tiny", "25.307 ms", "2.459 ms decoder", "Small control"],
        ["Whisper Base", "47.978 ms", "4.202 ms decoder", "Middle control"],
        ["Whisper Small FP16", "130.318 ms", "12.074 ms decoder", "Overlap-robust control"],
        ["Whisper Small W8A16", "376.895 ms", "7.856 ms decoder", "Memory tradeoff"],
        ["Piper TTS", "30.344 ms encoder; 15.189 ms flow", "3.018 ms decoder", "Audio fallback"],
    ]
    story.append(table(comparator, [36 * mm, 47 * mm, 48 * mm, 29 * mm], font_size=6.5))
    story.append(Paragraph("All rows are full-NPU QCS8550 proxy scorecards from qai-hub-models 0.57.3. They are fixed component invocations, not end-to-end pipeline latency. Whisper Small W8A16 makes the encoder 2.89x slower while making the decoder about 35% faster, so quantization must be profiled stage by stage.", styles["body"]))

    story.append(PageBreak())
    candidates = [
        ["Candidate", "Role", "Footprint view", "Decision"],
        ["LFM2.5-Audio", "Integrated research baseline", "Q4 bundle 1.001 GiB locally", "Continue with gates"],
        ["Moonshine + Qwen3-0.6B + Pocket", "Best footprint challenger", "Estimated 0.55-0.66 GB weights", "Profile next"],
        ["Zipformer + Qwen3-0.6B + Piper", "Conservative toolchain fallback", "About 0.84 GB from component cards", "Demo insurance"],
        ["Mini-Omni", "End-to-end control", "About 976M total estimated", "Later control"],
        ["LLaMA-Omni2-0.5B", "Misleading size label", "Checkpoint alone 3.857 GB BF16", "Exclude"],
    ]
    story.append(
        KeepTogether(
            [
                Paragraph("10. Candidate decision", styles["h1"]),
                table(candidates, [47 * mm, 42 * mm, 43 * mm, 28 * mm], font_size=6.7),
                Paragraph("Moonshine/Qwen/Pocket is the recommended next challenger because it creates the most headroom while keeping the text model on a known Qualcomm path. It still needs export, quality, and host-TTS measurements before it can replace LFM.", styles["body"]),
            ]
        )
    )

    story.append(Paragraph("11. Exact next steps", styles["h1"]))
    steps = [
        "Confirm the physical glasses SoC, app-available RAM, QNN/Voice AI SDK version, and deployment interface.",
        "Profile Qwen3-0.6B first, then Moonshine Base encoder/cached decoder, then Pocket TTS on the official host path. Keep Zipformer/Piper ready as swaps.",
        "Export four-layer LFM backbone shards and a production cached decoder with explicit position/mask inputs and bounded context.",
        "Convert and profile target-QNN quantized encoder, backbone, depth decoder, and caches separately. Run downstream transcript/token checks after every precision change.",
        "Measure the official FP32 detokenizer on the intended CPU/DSP path. Do not move it to FP16 HTP until real-code audio fidelity passes.",
        "Integrate VAD and measure speech-end to first audible PCM, process RSS, 30-minute temperature, and battery drop on the real glasses.",
    ]
    for index, value in enumerate(steps, 1):
        story.append(Paragraph(f"<b>{index}.</b> {value}", styles["body"]))

    story.append(Paragraph("12. Demo acceptance gates", styles["h1"]))
    gates = [
        ["Gate", "Provisional threshold"],
        ["Memory", "Provisional model-process peak <= 1.3 GB; revise after app RAM is confirmed."],
        ["Placement", "No unsupported main-backbone op; no undeclared CPU fallback in NPU shards."],
        ["ASR quality", "Quantized WER increase <= 0.3 percentage points on each evaluation split."],
        ["Responsiveness", "First audible PCM <= 500 ms after VAD close on the actual device."],
        ["Context", "Two-turn retention no more than five points below the BF16 golden set."],
        ["Audio", "No NaN/Inf, truncation, or material audio-text consistency regression."],
    ]
    story.append(table(gates, [40 * mm, 120 * mm], font_size=7.3))

    story.append(Paragraph("13. Limitations and release notes", styles["h1"]))
    for item in (
        "No exact AR1/AR1+/5100 target was available; QCS8550 proxy latency is not AR1 latency.",
        "Component latency and memory cannot be summed into an end-to-end figure.",
        "Backbone evidence is representative rather than one full compiled network.",
        "The small ASR diagnostic uses synthetic noise and is not a glasses microphone benchmark.",
        "Host FP32 detokenizer viability remains proposed rather than measured on target.",
        "The Q4 matrix uses two official samples on Apple M2 Metal; it does not establish general quality, Qualcomm placement, or glasses memory.",
        "LFM Open License v1.0 contains a commercial-use revenue threshold. Candidate components have separate attribution and license terms.",
    ):
        story.append(Paragraph("- " + item, styles["body"]))

    story.append(Paragraph("14. Reproducibility", styles["h1"]))
    repro = [
        ["Item", "Pinned value"],
        ["LFM model revision", "c362a0625dfe45aa588dce5f0ada28a7e5707628"],
        ["Q4 GGUF / runner", "7d525f883a077e20afb782f2ff618edcae0e39e4; build 7641 (68d8edf2); Apple M2 Metal"],
        ["Liquid Audio source", "19e65845923a7f136442c95137884ec61eb386aa"],
        ["Qualcomm proxy", "QCS8550 (Proxy), Android 12, Hexagon v73, QNN HTP FP16"],
        ["Comparator catalog", "qai-hub-models 0.57.3; QAIRT 2.45.0 scorecards"],
        ["Full-model golden", "PyTorch 2.8.0, Transformers 4.56.1, NVIDIA L4 BF16"],
    ]
    story.append(table(repro, [42 * mm, 118 * mm], font_size=7.2))

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("References", styles["h1"]))
    references = [
        ("Liquid AI, LFM2.5-Audio model card", "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B"),
        ("Liquid Audio source", "https://github.com/Liquid4All/liquid-audio"),
        ("Qualcomm AI Hub model catalog", "https://aihub.qualcomm.com/models/"),
        ("Qualcomm Zipformer", "https://aihub.qualcomm.com/models/zipformer"),
        ("Qualcomm Whisper Small", "https://aihub.qualcomm.com/models/whisper_small"),
        ("Qualcomm PiperTTS English", "https://aihub.qualcomm.com/models/pipertts_en"),
        ("Moonshine Base model card", "https://huggingface.co/UsefulSensors/moonshine-base"),
        ("Moonshine paper", "https://arxiv.org/abs/2410.15608"),
        ("Qwen3-0.6B model card", "https://huggingface.co/Qwen/Qwen3-0.6B"),
        ("Qualcomm Qwen3-0.6B", "https://aihub.qualcomm.com/models/qwen3_0_6b"),
        ("Pocket TTS", "https://github.com/kyutai-labs/pocket-tts"),
        ("Mini-Omni", "https://arxiv.org/abs/2408.16725"),
        ("LLaMA-Omni2", "https://arxiv.org/abs/2505.02625"),
    ]
    for index, (label, url) in enumerate(references, 1):
        story.append(Paragraph(f"{index}. {label}: <link href=\"{url}\" color=\"#246BCE\">{url}</link>", styles["ref"]))

    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()

    component_payload = load_json(project / "reports/aihub_component_results.json")
    strict_rows = [
        row
        for row in component_payload["results"]
        if row["runtime"] == "strict-npu"
    ]
    preferred_order = {
        name: index
        for index, name in enumerate(
            (
                "fastconformer",
                "backbone-conv-prefill",
                "backbone-attention-prefill",
                "backbone-conv-cached-decode",
                "backbone-attention-cached-decode",
                "depth-decoder",
                "detokenizer-t4",
                "detokenizer-t8",
            )
        )
    }
    strict_rows.sort(key=lambda row: preferred_order.get(row["component"], 99))

    quality_payload = load_json(project / "reports/colab_candidate_quality/summary.json", {})
    quality_rows = list(quality_payload.get("summary", []))
    moonshine_payload = load_json(project / "reports/moonshine_quality/summary.json", {})
    quality_rows.extend(moonshine_payload.get("summary", []))
    data = {
        "date": datetime.now().astimezone().strftime("%B %d, %Y"),
        "strict_rows": strict_rows,
        "bf16": load_json(project / "reports/colab_bf16/summary.json"),
        "quality_rows": quality_rows,
        "lfm_output_intelligibility": quality_payload.get("lfm_output_intelligibility", []),
        "real_detok": load_json(project / "reports/detok_real_turn1_t8/report.json", {}),
        "memory": load_json(project / "reports/memory/memory_ledger.json"),
        "quant": load_json(project / "reports/local_q4_matrix/summary.json", {}),
    }
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(build_markdown(data), encoding="utf-8")
    build_pdf(args.pdf, data)
    print(json.dumps({"markdown": str(args.markdown), "pdf": str(args.pdf)}, indent=2))


if __name__ == "__main__":
    main()
