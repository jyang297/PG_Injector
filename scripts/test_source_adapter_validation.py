from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import psycopg

from context_manager_config import get_config
from metadata_catalog import ColumnRef
from source_adapters import get_source_adapter


CONFIG = get_config()
DATABASE_URL = CONFIG.runtime.database_url


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def main():
    adapter = get_source_adapter("demo_json")
    owner = "test_owner"

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

        catalog = adapter.load(root, owner, "test.validation")
        issue_types = {issue.issue_type for issue in catalog.validation_issues}
        surviving_rule_ids = {rule.rule_id for rule in catalog.rules}

        expected_issue_types = {
            "duplicate_column_identity",
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
            ColumnRef("app_metadata", "launch_stage"),
            ColumnRef("secondary_app", "launch_stage"),
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

        catalog = adapter.load(root, owner, "test.missing_table_name")
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
                "--owner",
                owner,
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

    namespace = "test.failed_load_guard"
    with tempfile.TemporaryDirectory() as good_tmp, tempfile.TemporaryDirectory() as bad_tmp:
        good_root = Path(good_tmp)
        bad_root = Path(bad_tmp)
        write_json(
            good_root / "column_descriptions.json",
            [
                {
                    "table_name": "accounts",
                    "column_name": "status",
                    "description": "Account status.",
                }
            ],
        )
        write_json(
            good_root / "unique_values.json",
            [
                {
                    "table_name": "accounts",
                    "column_name": "status",
                    "values": [{"raw_value": "active", "value_gloss": "Active."}],
                }
            ],
        )
        write_json(good_root / "rules.json", [])

        subprocess.run(
            [
                sys.executable,
                "scripts/load_demo.py",
                "--owner",
                owner,
                "--namespace",
                namespace,
                "--data-dir",
                str(good_root),
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT count(*)
                    FROM metadata_chunks
                    WHERE resource_owner = %s
                      AND resource_namespace = %s
                    """,
                    (owner, namespace),
                )
                before_count = cur.fetchone()[0]

        write_json(
            bad_root / "column_descriptions.json",
            [
                {
                    "column_name": "owner",
                    "description": "Owner missing required table.",
                }
            ],
        )
        write_json(
            bad_root / "unique_values.json",
            [
                {
                    "column_name": "owner",
                    "values": [{"raw_value": "amy", "value_gloss": "Owner Amy."}],
                }
            ],
        )
        write_json(bad_root / "rules.json", [])

        result = subprocess.run(
            [
                sys.executable,
                "scripts/load_demo.py",
                "--owner",
                owner,
                "--namespace",
                namespace,
                "--data-dir",
                str(bad_root),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            raise SystemExit(
                "loader should fail fast before replacing an existing namespace"
            )

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT count(*)
                    FROM metadata_chunks
                    WHERE resource_owner = %s
                      AND resource_namespace = %s
                    """,
                    (owner, namespace),
                )
                after_count = cur.fetchone()[0]
        if before_count != after_count:
            raise SystemExit(
                f"failed load should preserve namespace rows; before={before_count} after={after_count}"
            )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(
            root / "column_descriptions.json",
            [
                {
                    "table_name": "bad::table",
                    "column_name": "status",
                    "description": "Invalid separator in source identifier.",
                }
            ],
        )
        write_json(root / "unique_values.json", [])
        write_json(root / "rules.json", [])

        catalog = adapter.load(root, owner, "test.invalid_identifier")
        issue_types = {issue.issue_type for issue in catalog.validation_issues}
        if "invalid_identifier_separator" not in issue_types:
            raise SystemExit("missing invalid_identifier_separator validation issue")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(
            root / "column_descriptions.json",
            [
                {
                    "table_name": "app_metadata",
                    "column_name": "launch_stage",
                    "description": "Launch stage.",
                }
            ],
        )
        write_json(
            root / "value_catalog.json",
            [
                {
                    "column_key": "launch_stage",
                    "values": [{"raw_value": "pilot", "value_gloss": "Pilot stage."}],
                }
            ],
        )
        write_json(root / "rules.json", [])

        catalog = adapter.load(root, owner, "test.legacy_value_catalog")
        issue_types = {issue.issue_type for issue in catalog.validation_issues}
        if "inferred_value_group_table_name" not in issue_types:
            raise SystemExit("expected inferred_value_group_table_name for legacy fallback")
        if len(catalog.values) != 1:
            raise SystemExit("legacy value_catalog fallback should still produce values")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(
            root / "column_descriptions.json",
            {
                "columns": [
                    {
                        "table_name": "apps",
                        "column_name": "owner",
                        "description": "Owner name.",
                        "aliases": "owner alias",
                    }
                ]
            },
        )
        write_json(root / "unique_values.json", {"value_groups": []})
        write_json(
            root / "rules.json",
            {
                "rules": [
                    {
                        "rule_id": "rule_with_type_drift",
                        "trigger_terms": "approval",
                        "priority": "high",
                        "candidate_columns": [
                            123,
                            {"table_name": "apps", "column_name": "owner"},
                        ],
                    }
                ]
            },
        )

        catalog = adapter.load(root, owner, "test.adapter_type_drift")
        issue_types = {issue.issue_type for issue in catalog.validation_issues}
        required_issue_types = {
            "wrapped_columns_payload",
            "wrapped_value_groups_payload",
            "wrapped_rules_payload",
            "coerced_scalar_list_field",
            "malformed_rule_candidate_column",
            "invalid_rule_priority",
        }
        if required_issue_types - issue_types:
            raise SystemExit(
                f"missing defensive validation issue types: "
                f"{sorted(required_issue_types - issue_types)}"
            )
        if catalog.columns[0].aliases != ["owner alias"]:
            raise SystemExit("scalar aliases should be coerced to a singleton list")
        if catalog.rules[0].trigger_terms != ["approval"]:
            raise SystemExit("scalar trigger_terms should be coerced to a singleton list")
        if catalog.rules[0].priority != 0:
            raise SystemExit("invalid rule priority should default to 0")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(root / "column_descriptions.json", {"bad": []})
        write_json(root / "unique_values.json", [])
        write_json(root / "rules.json", [])

        catalog = adapter.load(root, owner, "test.invalid_payload_shape")
        issue_types = {issue.issue_type for issue in catalog.validation_issues}
        if "invalid_columns_payload" not in issue_types:
            raise SystemExit("invalid top-level columns payload should become a validation issue")
        if catalog.columns:
            raise SystemExit("invalid top-level columns payload should not produce catalog columns")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(
            root / "column_descriptions.json",
            [
                {
                    "table_name": "apps",
                    "column_name": "owner",
                    "description": "Owner name.",
                }
            ],
        )
        write_json(root / "unique_values.json", [])
        write_json(root / "rules.json", {"bad": []})

        result = subprocess.run(
            [
                sys.executable,
                "scripts/load_demo.py",
                "--owner",
                owner,
                "--namespace",
                "test.invalid_rules_payload",
                "--data-dir",
                str(root),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            raise SystemExit("loader should fail fast on invalid top-level rules payload")

    print("source adapter validation passed")
