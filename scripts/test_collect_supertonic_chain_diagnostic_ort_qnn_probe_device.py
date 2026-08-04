#!/usr/bin/env python3
"""Static launch-hygiene gates for the diagnostic device collector."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = (
    ROOT / "scripts/collect_supertonic_chain_diagnostic_ort_qnn_probe_device.py"
)


def require_order(source: str, *needles: str) -> None:
    offsets = [source.index(needle) for needle in needles]
    assert offsets == sorted(offsets), (needles, offsets)


def main() -> None:
    source = COLLECTOR.read_text(encoding="utf-8")
    prepare = source.split("def prepare_prelaunch", 1)[1].split("def launch", 1)[0]
    require_order(
        prepare,
        '["shell", "am", "force-stop", PACKAGE]',
        '["rm", "-f", REPORT_RELATIVE, PARTIAL_RELATIVE]',
        "not app_private_path_exists(adb, REPORT_RELATIVE)",
        "not app_private_path_exists(adb, PARTIAL_RELATIVE)",
    )
    launch = source.split("def launch", 1)[1].split("def poll_report", 1)[0]
    require_order(
        launch,
        '"-n",\n            COMPONENT',
        '"--ez",\n            "run_probe",\n            "true"',
        '"--es",\n            "run_id",\n            run_id',
    )
    poll = source.split("def poll_report", 1)[1].split("def resolve_soc", 1)[0]
    assert 'report.get("run_id") != run_id' in poll
    assert 'report.get("report_complete") is not True' in poll
    assert 'report.get("completion_marker") != device_verifier.COMPLETION_MARKER' in poll
    main_source = source.split("def main()", 1)[1]
    require_order(
        main_source,
        "accepted_before = accepted_packages(adb, spec)",
        "prelaunch = prepare_prelaunch(adb)",
        "launch_evidence = launch(adb, arguments.run_id)",
        "raw_report, report = poll_report(",
        "atomic_write(report_path, raw_report)",
        "accepted_after = accepted_packages(adb, spec)",
        "verified = device_verifier.verify_documents(",
    )
    assert '["install"' not in source
    assert '["uninstall"' not in source
    assert '["pm", "clear"' not in source
    assert '"current_alias_published": False' in source
    assert '"verification_published": False' in source
    print("Supertonic chain diagnostic collector gates passed")


if __name__ == "__main__":
    main()
