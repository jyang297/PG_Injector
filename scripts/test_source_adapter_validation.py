from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from source_adapters import get_source_adapter


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def main():
    adapter = get_source_adapter("demo_json")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(
            root / "column_descriptions.json",
            [
                {
                    "table_name": "app_metadata",
                    "column_name": "launch_stage",
                    "description": "Stage one.",
                },
                {
                    "table_name": "app_metadata",
                    "column_name": "launch_stage",
                    "description": "Stage duplicate.",
                },
                {
                    "table_name": "secondary_app",
                    "column_name": "launch_stage",
                    "description": "Stage in another table.",
                },
            ],
        )
        write_json(
            root / "unique_values.json",
            [
                {
                    "table_name": "app_metadata",
                    "column_name": "launch_stage",
                    "values": [
                        {"raw_value": "pilot", "value_gloss": "Pilot stage."},
                        {"raw_value": "pilot", "value_gloss": "Pilot duplicate."},
                    ],
                }
            ],
        )
        write_json(
            root / "rules.json",
            [
                {
                    "rule_id": "rule_one",
                    "description": "Only apply to known columns.",
                    "candidate_columns": [
                        "nonexistent_column",
                        "launch_stage",
                        "app_metadata.launch_stage",
                        "secondary_app.launch_stage",
                    ],
                },
                {
                    "rule_id": "rule_one",
                    "description": "Duplicate rule id.",
                    "candidate_columns": ["launch_stage"],
                },
            ],
        )

        catalog = adapter.load(root, "test.validation")
        issue_types = {issue.issue_type for issue in catalog.validation_issues}
        surviving_rule_ids = {rule.rule_id for rule in catalog.rules}

        expected_issue_types = {
            "duplicate_column_key",
            "duplicate_value_key",
            "duplicate_rule_id",
            "unknown_rule_candidate_column",
            "unqualified_rule_candidate_column",
        }
        missing = expected_issue_types - issue_types
        if missing:
            raise SystemExit(f"missing validation issue types: {sorted(missing)}")
        if surviving_rule_ids != {"rule_one"}:
            raise SystemExit(f"unexpected surviving rules: {sorted(surviving_rule_ids)}")
        if len(catalog.columns) != 2:
            raise SystemExit(
                f"table-qualified columns should survive as distinct identities; "
                f"got={len(catalog.columns)}"
            )
        surviving_rule = catalog.rules[0]
        if surviving_rule.candidate_columns != [
            "app_metadata::launch_stage",
            "secondary_app::launch_stage",
        ]:
            raise SystemExit(
                "rule candidate validation should keep only known table-qualified columns"
            )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(
            root / "column_descriptions.json",
            [
                {
                    "table_name": "accounts",
                    "column_name": "status",
                    "description": "Account status.",
                },
                {
                    "column_name": "owner",
                    "description": "Owner name without a table.",
                },
            ],
        )
        write_json(
            root / "unique_values.json",
            [
                {
                    "table_name": "accounts",
                    "column_name": "status",
                    "values": [{"raw_value": "active", "value_gloss": "Active."}],
                },
                {
                    "column_name": "owner",
                    "values": [{"raw_value": "amy", "value_gloss": "Owner Amy."}],
                },
            ],
        )
        write_json(root / "rules.json", [])

        catalog = adapter.load(root, "test.missing_table_name")
        issue_types = {issue.issue_type for issue in catalog.validation_issues}
        required_issue_types = {
            "missing_column_table_name",
            "missing_value_group_table_name",
        }
        if required_issue_types - issue_types:
            raise SystemExit(
                f"missing required table-name issues: "
                f"{sorted(required_issue_types - issue_types)}"
            )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/load_demo.py",
                "--namespace",
                "test.missing_table_name",
                "--data-dir",
                str(root),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            raise SystemExit(
                "loader should fail fast when source rows omit required table_name"
            )

    print("source adapter validation passed")
