from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from metadata_catalog import (
    ColumnSpec,
    MetadataCatalog,
    RuleSpec,
    ValidationIssue,
    ValueSpec,
    build_column_key,
)
from normalization import dedupe_terms


class SourceAdapter(Protocol):
    # Adapters absorb raw source variability. Downstream loader/query code
    # should not care which JSON fields or nesting patterns produced the
    # catalog, only that they receive a normalized MetadataCatalog.
    def load(
        self,
        root: Path,
        namespace: str,
    ) -> MetadataCatalog: ...


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_json(path: Path, default=None):
    if not path.exists():
        return [] if default is None else default
    return _read_json(path)


def _select_value_source(root: Path) -> tuple[list[dict], str | None]:
    primary = root / "unique_values.json"
    if primary.exists():
        return _read_json(primary), primary.name

    fallback = root / "value_catalog.json"
    if fallback.exists():
        return _read_json(fallback), fallback.name

    return [], None


class DemoJsonAdapter:
    name = "demo_json"

    def load(
        self,
        root: Path,
        namespace: str,
    ) -> MetadataCatalog:
        # This adapter preserves the current demo file layout while translating
        # it into the canonical internal catalog contract used by the pipeline.
        column_path = root / "column_descriptions.json"
        if not column_path.exists():
            raise FileNotFoundError(
                f"Required metadata file is missing: {column_path}"
            )

        raw_columns = _read_json(column_path)
        raw_value_groups, value_source = _select_value_source(root)
        raw_rules = _read_optional_json(root / "rules.json", default=[])

        validation_issues: list[ValidationIssue] = []
        columns: list[ColumnSpec] = []
        known_column_keys: set[str] = set()
        seen_column_keys: set[str] = set()

        for item in raw_columns:
            table_name = (
                item.get("table_name")
                or item.get("table")
                or item.get("source_table")
            )
            raw_column_key = item.get("column_key")
            column_name = item.get("column_name")
            if not column_name and isinstance(raw_column_key, str):
                if "::" in raw_column_key:
                    embedded_table_name, embedded_column_name = raw_column_key.split(
                        "::", maxsplit=1
                    )
                    table_name = table_name or embedded_table_name
                    column_name = embedded_column_name
                else:
                    column_name = raw_column_key
            if not column_name:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="missing_column_name",
                        entity_key="column_definition",
                        action="skipped_column",
                    )
                )
                continue

            if not table_name:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="missing_column_table_name",
                        entity_key=column_name,
                        action="skipped_column",
                    )
                )
                continue
            column_key = build_column_key(table_name, column_name)

            if column_key in seen_column_keys:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="duplicate_column_key",
                        entity_key=column_key,
                        action="skipped_duplicate_column",
                    )
                )
                continue

            columns.append(
                ColumnSpec(
                    table_name=table_name,
                    column_name=column_name,
                    raw_column_name=item.get("raw_column_name"),
                    description=item.get("description", ""),
                    aliases=dedupe_terms(item.get("aliases", [])),
                    data_type=item.get("data_type"),
                )
            )
            seen_column_keys.add(column_key)
            known_column_keys.add(column_key)

        values: list[ValueSpec] = []
        seen_value_keys: set[tuple[str, str]] = set()
        for group in raw_value_groups:
            table_name = (
                group.get("table_name")
                or group.get("table")
                or group.get("source_table")
            )
            raw_column_key = group.get("column_key")
            column_name = group.get("column_name")
            if not column_name and isinstance(raw_column_key, str):
                if "::" in raw_column_key:
                    embedded_table_name, embedded_column_name = raw_column_key.split(
                        "::", maxsplit=1
                    )
                    table_name = table_name or embedded_table_name
                    column_name = embedded_column_name
                else:
                    column_name = raw_column_key
            if not column_name:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="missing_value_group_column_name",
                        entity_key="value_definition",
                        action="skipped_value_group",
                    )
                )
                continue

            if not table_name:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="missing_value_group_table_name",
                        entity_key=column_name,
                        action="skipped_value_group",
                    )
                )
                continue
            column_key = build_column_key(table_name, column_name)
            if column_key not in known_column_keys:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="missing_value_column",
                        entity_key=column_key,
                        action="skipped_value_group",
                        details={
                            "value_count": len(group.get("values", [])),
                        },
                    )
                )
                continue

            for raw_value_spec in group.get("values", []):
                raw_value = raw_value_spec.get("raw_value")
                if not raw_value:
                    validation_issues.append(
                        ValidationIssue(
                            issue_type="missing_raw_value",
                            entity_key=column_key,
                            action="skipped_value",
                        )
                    )
                    continue

                value_key = (column_key, raw_value)
                if value_key in seen_value_keys:
                    validation_issues.append(
                        ValidationIssue(
                            issue_type="duplicate_value_key",
                            entity_key=f"{column_key}::{raw_value}",
                            action="skipped_duplicate_value",
                        )
                    )
                    continue

                values.append(
                    ValueSpec(
                        table_name=table_name,
                        column_name=column_name,
                        raw_value=raw_value,
                        value_gloss=raw_value_spec.get("value_gloss")
                        or raw_value_spec.get("value_description", ""),
                        synonyms=dedupe_terms(raw_value_spec.get("synonyms", [])),
                        business_tags=dedupe_terms(
                            raw_value_spec.get("business_tags", [])
                        ),
                    )
                )
                seen_value_keys.add(value_key)

        rules: list[RuleSpec] = []
        seen_rule_ids: set[str] = set()
        for raw_rule in raw_rules:
            rule_id = raw_rule.get("rule_id")
            if not rule_id:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="missing_rule_id",
                        entity_key="rule_definition",
                        action="skipped_rule",
                    )
                )
                continue

            if rule_id in seen_rule_ids:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="duplicate_rule_id",
                        entity_key=rule_id,
                        action="skipped_duplicate_rule",
                    )
                )
                continue

            candidate_columns = dedupe_terms(raw_rule.get("candidate_columns", []))
            valid_candidate_columns = []
            for candidate_column in candidate_columns:
                if "::" in candidate_column:
                    candidate_key = candidate_column
                elif "." in candidate_column:
                    table_name, column_name = candidate_column.rsplit(".", maxsplit=1)
                    candidate_key = build_column_key(table_name, column_name)
                else:
                    validation_issues.append(
                        ValidationIssue(
                            issue_type="unqualified_rule_candidate_column",
                            entity_key=rule_id,
                            action="dropped_candidate_column",
                            details={"column_name": candidate_column},
                        )
                    )
                    continue

                if candidate_key in known_column_keys:
                    valid_candidate_columns.append(candidate_key)
                    continue
                validation_issues.append(
                    ValidationIssue(
                        issue_type="unknown_rule_candidate_column",
                        entity_key=rule_id,
                        action="dropped_candidate_column",
                        details={"column_name": candidate_column},
                    )
                )
            if candidate_columns and not valid_candidate_columns:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="rule_without_valid_candidate_columns",
                        entity_key=rule_id,
                        action="skipped_rule",
                    )
                )
                continue

            rules.append(
                RuleSpec(
                    rule_id=rule_id,
                    description=raw_rule.get("text_semantic")
                    or raw_rule.get("description")
                    or "",
                    candidate_columns=valid_candidate_columns,
                    trigger_terms=dedupe_terms(raw_rule.get("trigger_terms", [])),
                    intent=raw_rule.get("intent"),
                    priority=int(raw_rule.get("priority", 0)),
                    rule_text=raw_rule.get("rule_text"),
                )
            )
            seen_rule_ids.add(rule_id)

        return MetadataCatalog(
            namespace=namespace,
            columns=columns,
            values=values,
            rules=rules,
            source_files={
                "columns": column_path.name,
                "values": value_source,
                "rules": "rules.json" if (root / "rules.json").exists() else None,
            },
            source_counts={
                "columns": len(raw_columns),
                "value_groups": len(raw_value_groups),
                "values": sum(len(group.get("values", [])) for group in raw_value_groups),
                "rules": len(raw_rules),
            },
            validation_issues=validation_issues,
        )


def get_source_adapter(name: str) -> SourceAdapter:
    normalized_name = name.strip().lower()
    if normalized_name in {"demo_json", "demo"}:
        return DemoJsonAdapter()
    raise ValueError(f"Unsupported source adapter: {name}")
