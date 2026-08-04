#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re

import qai_hub as hub


DEFAULT_PATTERN = r"AR1|5100|wear|glasses|QCS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List AI Hub devices relevant to the glasses deployment target."
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Case-insensitive regex matched against name, OS, and attributes.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Return every available device instead of applying --pattern.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = re.compile(args.pattern, re.IGNORECASE)
    client = hub.Client()
    devices = client.get_devices()
    rows = []
    for device in devices:
        name = getattr(device, "name", str(device))
        os_version = getattr(device, "os", "")
        attributes = list(getattr(device, "attributes", []) or [])
        searchable = " ".join([name, os_version, *attributes])
        if args.all or pattern.search(searchable):
            rows.append(
                {
                    "name": name,
                    "os": os_version,
                    "attributes": attributes,
                    "supports_qnn": "framework:qnn" in attributes,
                    "selector": {
                        "device": name,
                        "device_os": os_version,
                    },
                }
            )
    rows.sort(key=lambda row: (row["name"].lower(), row["os"]))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
