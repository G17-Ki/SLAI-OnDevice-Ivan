#!/usr/bin/env python3
"""Profile the native LFM Android APK through its app-owned local API."""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import http.client
import json
import math
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PACKAGE = "ai.liquid.lfmdemo.dev"
ACTIVITY = f"{PACKAGE}/.MainActivity"
API_URL = "http://127.0.0.1:18080/v1/chat/completions"
ROOT_URL = "http://127.0.0.1:18080/"
REFERENCE_TEXT = {
    "question.wav": "Can you help me come up with a slogan for my woodworking site business?",
    "asr.wav": (
        "The stale smell of old beer lingers. It takes heat to bring out the odor. "
        "A cold dip restores health and zest. A salt pickle tastes fine with ham. "
        "Tacos al pastor are my favorite. A zestful food is the hot cross bun."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-starts", type=int, default=3)
    parser.add_argument("--question-runs", type=int, default=10)
    parser.add_argument("--long-asr-runs", type=int, default=5)
    parser.add_argument("--chat-runs", type=int, default=5)
    parser.add_argument("--question-audio", type=Path)
    parser.add_argument("--long-asr-audio", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def default_asset_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "vendor" / "liquid-audio" / "assets"


def run(command: list[str], *, check: bool = True, timeout: float = 120) -> str:
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout, errors="replace")
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")
    return result.stdout


def adb_shell(command: str, *, check: bool = True, timeout: float = 120) -> str:
    return run(["adb", "shell", command], check=check, timeout=timeout).replace("\r", "")


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def word_error_rate(reference: str, prediction: str) -> float:
    ref = normalize(reference)
    hyp = normalize(prediction)
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return 100.0 * previous[-1] / max(1, len(ref))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def make_payload(audio_path: Path, mode: str) -> bytes:
    audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    payload = {
        "model": "LFM2.5-Audio-1.5B",
        "stream": True,
        "temperature": 0,
        "max_tokens": 320 if mode == "chat" else 192,
        "messages": [
            {
                "role": "system",
                "content": "Respond with interleaved text and audio." if mode == "chat" else "Perform ASR.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio, "format": "wav"},
                    }
                ],
            },
        ],
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def request_once(audio_path: Path, mode: str, label: str, run_index: int, measured: bool) -> dict[str, Any]:
    body = make_payload(audio_path, mode)
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_text: float | None = None
    first_audio: float | None = None
    text_parts: list[str] = []
    audio_bytes = 0
    audio_sample_rate: int | None = None
    events = 0
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            events += 1
            delta = event.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content")
            if text:
                if first_text is None:
                    first_text = time.perf_counter()
                text_parts.append(text)
            audio_chunk = delta.get("audio_chunk")
            if audio_chunk:
                if first_audio is None:
                    first_audio = time.perf_counter()
                raw_audio = base64.b64decode(audio_chunk["data"])
                audio_bytes += len(raw_audio)
                audio_sample_rate = int(audio_chunk.get("sample_rate", 24000))
    ended = time.perf_counter()
    transcript = "".join(text_parts).strip()
    reference = REFERENCE_TEXT.get(audio_path.name) if mode == "asr" else None
    return {
        "label": label,
        "mode": mode,
        "audio": audio_path.name,
        "run_index": run_index,
        "measured": measured,
        "first_text_ms": None if first_text is None else (first_text - started) * 1000,
        "first_audio_ms": None if first_audio is None else (first_audio - started) * 1000,
        "total_ms": (ended - started) * 1000,
        "transcript": transcript,
        "reference": reference,
        "wer_percent": None if reference is None else word_error_rate(reference, transcript),
        "exact_normalized": None if reference is None else normalize(reference) == normalize(transcript),
        "audio_output_bytes_f32": audio_bytes,
        "audio_output_sample_rate": audio_sample_rate,
        "audio_output_seconds": (
            None if not audio_bytes or not audio_sample_rate else audio_bytes / 4.0 / audio_sample_rate
        ),
        "sse_events": events,
    }


def wait_ready(timeout: float = 30) -> tuple[float, int, int]:
    started = time.perf_counter()
    app_pid = 0
    model_pid = 0
    while time.perf_counter() - started < timeout:
        pids = adb_shell(f"echo APP=$(pidof {PACKAGE}); echo MODEL=$(pidof liblfmserver.so)", check=False)
        app_match = re.search(r"APP=(\d+)", pids)
        model_match = re.search(r"MODEL=(\d+)", pids)
        app_pid = int(app_match.group(1)) if app_match else 0
        model_pid = int(model_match.group(1)) if model_match else 0
        if model_pid:
            try:
                with urllib.request.urlopen(ROOT_URL, timeout=0.5):
                    pass
            except urllib.error.HTTPError:
                return (time.perf_counter() - started) * 1000, app_pid, model_pid
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                OSError,
                http.client.RemoteDisconnected,
            ):
                pass
        time.sleep(0.1)
    raise TimeoutError("Native APK model server did not become ready")


def cold_start_once() -> dict[str, Any]:
    adb_shell(f"am force-stop {PACKAGE}")
    time.sleep(0.2)
    started = time.perf_counter()
    adb_shell(f"am start -n {ACTIVITY}")
    ready_ms, app_pid, model_pid = wait_ready()
    return {
        "tap_to_ready_ms": (time.perf_counter() - started) * 1000,
        "post_start_command_to_ready_ms": ready_ms,
        "app_pid": app_pid,
        "model_pid": model_pid,
    }


def parse_battery(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "AC powered",
        "USB powered",
        "Wireless powered",
        "level",
        "voltage",
        "temperature",
        "Charge counter",
        "status",
    ):
        match = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
        if not match:
            continue
        value = match.group(1).strip()
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
        else:
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
    if "temperature" in result:
        result["temperature_c"] = result["temperature"] / 10.0
    return result


def parse_thermal(text: str) -> dict[str, Any]:
    current = text.split("Current temperatures from HAL:", 1)[-1]
    current = current.split("Current cooling devices from HAL:", 1)[0]
    values: dict[str, list[float]] = {"cpu": [], "gpu": [], "nsp": []}
    named: dict[str, float] = {}
    for value, _kind, name, status in re.findall(
        r"Temperature\{mValue=([\d.+-]+), mType=(\d+), mName=([^,]+), mStatus=(\d+)\}", current
    ):
        number = float(value)
        named[name] = number
        if name.startswith("CPU"):
            values["cpu"].append(number)
        elif name.startswith("GPU"):
            values["gpu"].append(number)
        elif name.startswith("nsp"):
            values["nsp"].append(number)
    status_match = re.search(r"Thermal Status:\s*(\d+)", text)
    return {
        "status": int(status_match.group(1)) if status_match else None,
        "cpu_max_c": max(values["cpu"], default=math.nan),
        "gpu_max_c": max(values["gpu"], default=math.nan),
        "nsp_max_c": max(values["nsp"], default=math.nan),
        "skin_c": named.get("skin"),
        "battery_c": named.get("battery"),
        "all_current": named,
    }


def device_snapshot() -> dict[str, Any]:
    return {
        "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "battery": parse_battery(adb_shell("dumpsys battery")),
        "thermal": parse_thermal(adb_shell("dumpsys thermalservice", timeout=30)),
    }


def parse_proc_status(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in ("VmRSS", "VmHWM", "VmSize", "Threads"):
        match = re.search(rf"^{key}:\s*(\d+)", text, flags=re.MULTILINE)
        if match:
            result[key] = int(match.group(1))
    return result


class ResourceSampler:
    def __init__(self, app_pid: int, model_pid: int, interval: float = 0.25):
        self.app_pid = app_pid
        self.model_pid = model_pid
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="android-resource-sampler", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        started = time.perf_counter()
        while not self.stop_event.is_set():
            command = (
                f"echo APP; cat /proc/{self.app_pid}/status 2>/dev/null; "
                f"echo MODEL; cat /proc/{self.model_pid}/status 2>/dev/null"
            )
            text = adb_shell(command, check=False, timeout=10)
            app_text, _, model_text = text.partition("MODEL\n")
            self.samples.append(
                {
                    "elapsed_s": time.perf_counter() - started,
                    "app": parse_proc_status(app_text),
                    "model": parse_proc_status(model_text),
                }
            )
            self.stop_event.wait(self.interval)


def workload_summary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    selected = [row for row in rows if row["label"] == label and row["measured"]]
    first_text = [float(row["first_text_ms"]) for row in selected if row["first_text_ms"] is not None]
    first_audio = [float(row["first_audio_ms"]) for row in selected if row["first_audio_ms"] is not None]
    totals = [float(row["total_ms"]) for row in selected]
    return {
        "runs": len(selected),
        "first_text_ms": summary(first_text),
        "first_audio_ms": summary(first_audio) if first_audio else None,
        "total_ms": summary(totals),
        "exact_transcripts": sum(row["exact_normalized"] is True for row in selected),
        "mean_wer_percent": statistics.fmean(
            float(row["wer_percent"]) for row in selected if row["wer_percent"] is not None
        ) if any(row["wer_percent"] is not None for row in selected) else None,
        "unique_transcripts": sorted({row["transcript"] for row in selected}),
        "audio_output_seconds": summary(
            [float(row["audio_output_seconds"]) for row in selected if row["audio_output_seconds"] is not None]
        ) if any(row["audio_output_seconds"] is not None for row in selected) else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "label", "mode", "audio", "run_index", "measured", "first_text_ms", "first_audio_ms",
        "total_ms", "wer_percent", "exact_normalized", "audio_output_seconds", "audio_output_bytes_f32",
        "sse_events", "transcript",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_metric(stats: dict[str, float] | None) -> str:
    if not stats:
        return "—"
    return f"{stats['median']:.1f} / {stats['p95']:.1f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    workloads = report["workloads"]
    before = report["environment_before"]
    after = report["environment_after"]
    resource = report["resources"]
    lines = [
        "# Native Nubia on-device profile",
        "",
        f"Date: {report['created_utc']}",
        "",
        "This profile measures the native `LFM Audio` APK and its app-owned CPU Q4 runner. It is not a QNN/NPU measurement.",
        "",
        "## Latency and quality",
        "",
        "Median / P95 in milliseconds:",
        "",
        "| Workload | Runs | First text | First audio | Total | Exact ASR | Mean WER |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("question_asr", "long_asr", "question_chat"):
        item = workloads[label]
        exact = "—" if item["mean_wer_percent"] is None else f"{item['exact_transcripts']}/{item['runs']}"
        wer = "—" if item["mean_wer_percent"] is None else f"{item['mean_wer_percent']:.2f}%"
        lines.append(
            f"| {label} | {item['runs']} | {format_metric(item['first_text_ms'])} | "
            f"{format_metric(item['first_audio_ms'])} | {format_metric(item['total_ms'])} | {exact} | {wer} |"
        )
    cold = report["cold_start_ms"]
    lines += [
        "",
        "## Workload definitions",
        "",
        "- `question_asr`: the official 4.904-second mono `question.wav` sample (16 kHz). "
        "It asks, ‘Can you help me come up with a slogan for my woodworking site business?’ "
        "The system prompt is `Perform ASR.`, so the expected output is only a transcript. "
        "This is the short-speech latency and transcription-accuracy test.",
        "- `long_asr`: the official 18.356-second mono `asr.wav` sample (44.1 kHz), containing "
        "a six-sentence speech-recognition test passage. It uses the same `Perform ASR.` prompt. "
        "This checks how latency and accuracy scale with a substantially longer input.",
        "- `question_chat`: reuses `question.wav`, but changes the system prompt to "
        "`Respond with interleaved text and audio.` The model answers the woodworking question as "
        "an assistant, streaming both response text and generated 24 kHz speech. It is not scored "
        "with WER because many different answers can be valid.",
        "",
        "`First text` is time to the first streamed text fragment; `First audio` is time to the "
        "first generated audio chunk; and `Total` is time until the server's completion event. "
        "`Exact ASR` counts normalized transcript matches, while WER is word error rate. A dash "
        "means the metric does not apply.",
        "",
        "The complete audio file is submitted with each request before inference begins. These are "
        "batch/file inference timings after utterance completion, not end-to-end timings that include "
        "speaking, microphone capture, or VAD endpoint detection.",
        "",
        "## Process restart (warm filesystem cache)",
        "",
        f"- App launch to ready: median {cold['median']:.1f} ms, P95 {cold['p95']:.1f} ms across {len(report['cold_starts'])} launches.",
        "- These launches restart the process but retain model pages in the Linux page cache; this is not a post-reboot cold-load measurement.",
        "",
        "## Memory",
        "",
        f"- Model process peak RSS: {resource['model_peak_rss_kib'] / 1024:.1f} MiB.",
        f"- Android UI/service process peak RSS: {resource['app_peak_rss_kib'] / 1024:.1f} MiB.",
        f"- Combined sampled peak RSS: {resource['combined_peak_rss_kib'] / 1024:.1f} MiB.",
        "",
        "## Thermal and battery",
        "",
        f"- CPU max: {before['thermal']['cpu_max_c']:.1f} → {after['thermal']['cpu_max_c']:.1f} °C.",
        f"- GPU max: {before['thermal']['gpu_max_c']:.1f} → {after['thermal']['gpu_max_c']:.1f} °C.",
        f"- Skin: {before['thermal']['skin_c']:.1f} → {after['thermal']['skin_c']:.1f} °C.",
        f"- Battery: {before['battery']['temperature_c']:.1f} → {after['battery']['temperature_c']:.1f} °C; level {before['battery']['level']}% → {after['battery']['level']}%.",
        f"- Charging during profile: {before['battery'].get('AC powered') or before['battery'].get('USB powered')}. Battery drain is therefore not reported as a valid power measurement.",
        "",
        "## Notes",
        "",
        "- `first_text` and `first_audio` are client-observed streaming times from request submission.",
        "- Warm-up requests are excluded from reported distributions.",
        "- Exact transcript checks use normalized word equality against the two official reference samples.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    assets = default_asset_dir()
    question = args.question_audio or assets / "question.wav"
    long_asr = args.long_asr_audio or assets / "asr.wav"
    for path in (question, long_asr):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir or (
        Path(__file__).resolve().parents[1] / "reports" / "android_phone" / "native_profile" / utc_stamp()
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    if "device" not in run(["adb", "get-state"]):
        raise RuntimeError("No authorized Android device")
    run(["adb", "forward", "tcp:18080", "tcp:8080"])
    run(
        ["adb", "exec-out", "run-as", PACKAGE, "sh", "-c", ": > files/lfm-server.log"],
        check=False,
    )

    cold_starts: list[dict[str, Any]] = []
    for index in range(args.cold_starts):
        result = cold_start_once()
        result["run_index"] = index + 1
        cold_starts.append(result)

    app_pid = cold_starts[-1]["app_pid"]
    model_pid = cold_starts[-1]["model_pid"]
    environment_before = device_snapshot()
    sampler = ResourceSampler(app_pid, model_pid)
    sampler.start()

    rows: list[dict[str, Any]] = []
    workloads = [
        ("question_asr", question, "asr", args.question_runs),
        ("long_asr", long_asr, "asr", args.long_asr_runs),
        ("question_chat", question, "chat", args.chat_runs),
    ]
    try:
        for label, audio_path, mode, count in workloads:
            rows.append(request_once(audio_path, mode, label, 0, measured=False))
            for index in range(1, count + 1):
                rows.append(request_once(audio_path, mode, label, index, measured=True))
    finally:
        sampler.stop()

    environment_after = device_snapshot()
    model_rss = [sample["model"].get("VmRSS", 0) for sample in sampler.samples]
    app_rss = [sample["app"].get("VmRSS", 0) for sample in sampler.samples]
    combined = [a + m for a, m in zip(app_rss, model_rss)]
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": {
            "model": adb_shell("getprop ro.product.model").strip(),
            "soc": adb_shell("getprop ro.soc.model").strip(),
            "android": adb_shell("getprop ro.build.version.release").strip(),
            "sdk": adb_shell("getprop ro.build.version.sdk").strip(),
            "package": PACKAGE,
            "backend": "official Android ARM64 Q4 CPU runner",
        },
        "cold_starts": cold_starts,
        "cold_start_ms": summary([item["tap_to_ready_ms"] for item in cold_starts]),
        "workloads": {
            label: workload_summary(rows, label)
            for label in ("question_asr", "long_asr", "question_chat")
        },
        "resources": {
            "sample_count": len(sampler.samples),
            "interval_target_s": sampler.interval,
            "model_peak_rss_kib": max(model_rss, default=0),
            "app_peak_rss_kib": max(app_rss, default=0),
            "combined_peak_rss_kib": max(combined, default=0),
            "model_last": sampler.samples[-1]["model"] if sampler.samples else {},
            "app_last": sampler.samples[-1]["app"] if sampler.samples else {},
        },
        "environment_before": environment_before,
        "environment_after": environment_after,
        "runs": rows,
    }
    (output_dir / "profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "resource_samples.json").write_text(
        json.dumps(sampler.samples, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "runs.csv", rows)
    write_markdown(output_dir / "REPORT.md", report)
    server_log = run(
        ["adb", "exec-out", "run-as", PACKAGE, "cat", "files/lfm-server.log"],
        check=False,
        timeout=30,
    )
    (output_dir / "lfm-server.log").write_text(server_log, encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
