#!/usr/bin/env python3
"""Submit or poll resume-safe Supertonic QAIRT 2.42 compile probes.

The tool reuses authenticated, account-owned AI Hub source models from prior
successful QAIRT 2.45 compiles. It never uploads or runs a model locally, never
submits a 2.45 fallback, and never profiles or infers. Use the smallest duration
predictor first; submit the remaining graphs only after that compatibility gate
passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "supertonic_qairt242_compile_spec.json"
)
DEFAULT_STATE = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "supertonic_qairt242_compile_state.json"
)
EXPECTED_GATE_ID = "supertonic_sm8850_qairt242_context_recompile_v1"
EXPECTED_CLIENT_VERSION = "0.52.0"
EXPECTED_QAIRT_REQUEST = "2.42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def normalize_specs(value: dict[str, Any]) -> dict[str, list[Any]]:
    normalized: dict[str, list[Any]] = {}
    for name, spec in value.items():
        if not isinstance(spec, (list, tuple)) or len(spec) != 2:
            raise RuntimeError(f"malformed input spec for {name}: {spec!r}")
        shape, dtype = spec
        normalized[str(name)] = [[int(item) for item in shape], str(dtype)]
    return dict(sorted(normalized.items()))


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1 or spec.get("gate_id") != EXPECTED_GATE_ID:
        raise RuntimeError("unexpected Supertonic compile spec identity")
    if spec.get("qai_hub_client_version") != EXPECTED_CLIENT_VERSION:
        raise RuntimeError("unexpected pinned QAI Hub client version")
    if spec.get("requested_qairt_version") != EXPECTED_QAIRT_REQUEST:
        raise RuntimeError("only the QAIRT 2.42 compatibility request is allowed")
    fallback = spec.get("fallback_policy", {})
    if fallback.get("allow_qairt_245_retry_for_phone_artifact") is not False:
        raise RuntimeError("QAIRT 2.45 retry must remain disabled")
    if fallback.get("allow_qairt_245_context_in_qairt_242_phone_runtime") is not False:
        raise RuntimeError("2.45-context/2.42-runtime mixing must remain disabled")
    components = spec.get("components")
    if not isinstance(components, dict) or not components:
        raise RuntimeError("compile spec has no components")
    orders: list[int] = []
    for name, record in components.items():
        orders.append(int(record["pilot_order"]))
        if record["prior_compile_job"] == "":
            raise RuntimeError(f"missing prior compile job for {name}")
        if record["source_model"]["id"] == "":
            raise RuntimeError(f"missing source model ID for {name}")
        if "--qairt_version 2.42" not in record["compile_options"]:
            raise RuntimeError(f"component does not request QAIRT 2.42: {name}")
        if "--qairt_version 2.45" in record["compile_options"]:
            raise RuntimeError(f"component contains forbidden QAIRT 2.45 fallback: {name}")
        normalize_specs(record["input_specs"])
    if sorted(orders) != list(range(1, len(orders) + 1)):
        raise RuntimeError("component pilot order must be contiguous and unique")


def status_record(job: Any) -> dict[str, Any]:
    status = job.get_status()
    return {
        "job_id": str(job.job_id),
        "name": str(job.name),
        "status": str(status.code),
        "message": str(status.message),
        "finished": bool(status.finished),
        "success": bool(status.success),
    }


def framework_inventory(client: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": str(item.name),
            "api_version": str(item.api_version),
            "api_tags": [str(value) for value in item.api_tags],
            "full_version": str(item.full_version),
        }
        for item in client.get_frameworks()
    ]


def validate_prior_job(job: Any, record: dict[str, Any], target: dict[str, Any]) -> None:
    status = job.get_status()
    if not status.finished or not status.success:
        raise RuntimeError(f"prior compile job is not successful: {job.job_id}")
    if str(job.job_id) != record["prior_compile_job"]:
        raise RuntimeError("prior compile job ID mismatch")
    if str(job.model.model_id) != record["source_model"]["id"]:
        raise RuntimeError("prior source model ID mismatch")
    if str(job.model.name) != record["source_model"]["name"]:
        raise RuntimeError("prior source model name mismatch")
    if str(job.options) != record["prior_options"]:
        raise RuntimeError("prior compile options changed")
    if normalize_specs(job.shapes) != normalize_specs(record["input_specs"]):
        raise RuntimeError("prior compile input shapes changed")
    if str(job.device.name) != target["device"] or str(job.device.os) != target["os"]:
        raise RuntimeError("prior compile target changed")
    attributes = set(str(item) for item in job.device.attributes)
    missing = set(target["required_attributes"]) - attributes
    if missing:
        raise RuntimeError(f"prior compile target is missing attributes: {sorted(missing)}")


def load_or_initialize_state(
    path: Path,
    spec_path: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    spec_digest = sha256(spec_path)
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("gate_id") != EXPECTED_GATE_ID:
            raise RuntimeError("state belongs to another gate")
        if state.get("spec_sha256") != spec_digest:
            raise RuntimeError("state belongs to a different compile spec")
        return state
    return {
        "schema_version": 1,
        "gate_id": EXPECTED_GATE_ID,
        "spec_path": str(spec_path),
        "spec_sha256": spec_digest,
        "requested_qairt_version": spec["requested_qairt_version"],
        "phone_runtime_target_build": spec["phone_runtime_target_build"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "components": {},
        "claim_boundary": spec["claim_boundary"],
    }


def plan(spec: dict[str, Any], selected: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate_id": EXPECTED_GATE_ID,
        "status": "plan_only",
        "selected_components": selected,
        "jobs": [
            {
                "component": name,
                "pilot_order": spec["components"][name]["pilot_order"],
                "prior_compile_job": spec["components"][name]["prior_compile_job"],
                "source_model": spec["components"][name]["source_model"],
                "compile_options": spec["components"][name]["compile_options"],
                "input_specs": normalize_specs(spec["components"][name]["input_specs"]),
            }
            for name in selected
        ],
        "phone_or_host_model_execution": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--component", action="append")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true")
    action.add_argument("--poll", action="store_true")
    args = parser.parse_args()

    spec_path = args.spec.expanduser().resolve()
    state_path = args.state.expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_spec(spec)
    components = spec["components"]
    selected = args.component or [
        name
        for name, _ in sorted(
            components.items(), key=lambda item: int(item[1]["pilot_order"])
        )
    ]
    unknown = sorted(set(selected) - set(components))
    if unknown:
        parser.error(f"unknown components: {unknown}")
    if len(selected) != len(set(selected)):
        parser.error("component selection contains duplicates")

    if not args.submit and not args.poll:
        print(json.dumps(plan(spec, selected), indent=2, sort_keys=True))
        return 0

    import qai_hub as hub

    if str(getattr(hub, "__version__", "")) != EXPECTED_CLIENT_VERSION:
        raise RuntimeError("running QAI Hub client does not match the pinned version")
    client = hub.Client()
    state = load_or_initialize_state(state_path, spec_path, spec)
    exit_code = 0

    for name in selected:
        record = components[name]
        component_state = state["components"].setdefault(
            name,
            {
                "prior_compile_job": record["prior_compile_job"],
                "source_model": record["source_model"],
                "compile_options": record["compile_options"],
                "input_specs": normalize_specs(record["input_specs"]),
            },
        )
        existing_job_id = component_state.get("job", {}).get("job_id")
        if existing_job_id:
            job = client.get_job(existing_job_id)
        elif args.poll:
            raise RuntimeError(f"no submitted job exists for {name}")
        else:
            prior = client.get_job(record["prior_compile_job"])
            validate_prior_job(prior, record, spec["target"])
            try:
                job = client.submit_compile_job(
                    model=prior.model,
                    device=prior.device,
                    input_specs={
                        key: (tuple(value[0]), value[1])
                        for key, value in normalize_specs(record["input_specs"]).items()
                    },
                    options=record["compile_options"],
                    name=f"lfm-demo-v81-supertonic-{name}-qairt242-probe",
                    retry=False,
                )
            except Exception as error:
                component_state["status"] = "submission_rejected_before_job"
                component_state["submission_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "time_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                }
                component_state["advertised_frameworks"] = framework_inventory(client)
                component_state["job_created"] = False
                atomic_write_json(state_path, state)
                exit_code = 2
                continue
            component_state["submitted_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            component_state["job"] = {
                "job_id": str(job.job_id),
                "name": str(job.name),
                "status": "SUBMITTED",
                "finished": False,
                "success": False,
            }
            atomic_write_json(state_path, state)

        snapshot = status_record(job)
        component_state["job"] = snapshot
        component_state["last_checked_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        if snapshot["finished"] and snapshot["success"]:
            target_model = job.get_target_model()
            if target_model is None:
                raise RuntimeError(f"successful compile returned no target model: {name}")
            component_state["target_model"] = {
                "id": str(target_model.model_id),
                "name": str(target_model.name),
            }
            component_state["status"] = "compile_pass"
        elif snapshot["finished"]:
            component_state["status"] = "compile_rejected_or_failed"
        else:
            component_state["status"] = "compile_pending"
        atomic_write_json(state_path, state)

    print(json.dumps(state, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
