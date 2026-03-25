from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from context_manager_config import get_config


CONFIG = get_config()
ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def main():
    owner = "test_owner"
    namespace = "test.mandatory_columns"
    value_limit = CONFIG.loader.value_cardinality_limit

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(
            root / "column_descriptions.json",
            [
                {
                    "table_name": "apps",
                    "column_name": "app_name",
                    "description": "Human-readable application name.",
                },
                {
                    "table_name": "apps",
                    "column_name": "app_id",
                    "description": "Stable application identifier.",
                },
                {
                    "table_name": "app_metadata",
                    "column_name": "launch_stage",
                    "description": "Current rollout stage.",
                },
            ],
        )
        write_json(
            root / "unique_values.json",
            [
                {
                    "table_name": "apps",
                    "column_name": "app_name",
                    "values": [
                        {
                            "raw_value": f"App {idx}",
                            "value_gloss": f"Application name {idx}.",
                        }
                        for idx in range(value_limit + 2)
                    ],
                },
                {
                    "table_name": "apps",
                    "column_name": "app_id",
                    "values": [
                        {
                            "raw_value": f"app_{idx}",
                            "value_gloss": f"Application identifier {idx}.",
                        }
                        for idx in range(value_limit + 2)
                    ],
                },
                {
                    "table_name": "app_metadata",
                    "column_name": "launch_stage",
                    "values": [
                        {"raw_value": "pilot", "value_gloss": "Pilot rollout."},
                        {"raw_value": "ga", "value_gloss": "General availability."},
                    ],
                },
            ],
        )
        write_json(root / "rules.json", [])

        subprocess.run(
            [
                sys.executable,
                "scripts/load_demo.py",
                "--owner",
                owner,
                "--namespace",
                namespace,
                "--data-dir",
                str(root),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/query.py",
                "--owner",
                owner,
                "--namespace",
                namespace,
                "show launch stage",
                "--json",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    stdout = result.stdout
    payload = json.loads(
        stdout if stdout.lstrip().startswith("{") else stdout[stdout.find("{") :]
    )
    prompt_seen = {
        (item["table_name"], item["column_name"])
        for item in payload["prompt_metadata"]["candidate_columns"]
    }
    required = {
        ("apps", "app_name"),
        ("apps", "app_id"),
    }
    missing = required - prompt_seen
    if missing:
        raise SystemExit(
            f"mandatory prompt injection missed high-cardinality columns: {sorted(missing)}"
        )

    debug_bundles = {
        (item["table_name"], item["column_name"]): item
        for item in payload["candidate_columns_debug"]
    }
    for required_identity in required:
        bundle = debug_bundles.get(required_identity)
        if not bundle:
            raise SystemExit(f"missing debug bundle for mandatory column: {required_identity}")
        if not bundle.get("mandatory_description_in_prompt"):
            raise SystemExit(
                f"mandatory flag missing for high-cardinality column: {required_identity}"
            )

    print("mandatory column injection passed")


if __name__ == "__main__":
    main()
