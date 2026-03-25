from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from metadata_catalog import (
    ColumnRef,
    ColumnSpec,
    MetadataCatalog,
    RuleSpec,
    ValidationIssue,
    ValueSpec,
)
from normalization import dedupe_terms


class SourceAdapter(Protocol):
    # Adapters absorb raw source variability. Downstream loader/query code
    # should not care which JSON fields or nesting patterns produced the
    # catalog, only that they receive a normalized MetadataCatalog.
    def load(
        self,
        root: Path,
        resource_owner: str,
        resource_namespace: str,
    ) -> MetadataCatalog: ...


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_json(path: Path, default=None):
    if not path.exists():
        return [] if default is None else default
    return _read_json(path)


def _normalize_record_list(
    raw_payload,
    *,
    payload_name: str,
    wrapper_keys: tuple[str, ...],
    validation_issues: list[ValidationIssue],
) -> list[dict]:
    payload = raw_payload
    if isinstance(raw_payload, dict):
        for wrapper_key in wrapper_keys:
            wrapped_payload = raw_payload.get(wrapper_key)
            if isinstance(wrapped_payload, list):
                validation_issues.append(
                    ValidationIssue(
                        issue_type=f"wrapped_{payload_name}_payload",
                        entity_key=payload_name,
                        action="accepted_wrapped_payload",
                        details={"wrapper_key": wrapper_key},
                    )
                )
                payload = wrapped_payload
                break
        else:
            validation_issues.append(
                ValidationIssue(
                    issue_type=f"invalid_{payload_name}_payload",
                    entity_key=payload_name,
                    action="ignored_payload",
                    details={"actual_type": type(raw_payload).__name__},
                )
            )
            return []
    if not isinstance(payload, list):
        validation_issues.append(
            ValidationIssue(
                issue_type=f"invalid_{payload_name}_payload",
                entity_key=payload_name,
                action="ignored_payload",
                details={"actual_type": type(payload).__name__},
            )
        )
        return []

    normalized_records: list[dict] = []
    record_issue_type = f"invalid_{payload_name.rstrip('s')}_record"
    for index, record in enumerate(payload):
        if isinstance(record, dict):
            normalized_records.append(record)
            continue
        validation_issues.append(
            ValidationIssue(
                issue_type=record_issue_type,
                entity_key=f"{payload_name}[{index}]",
                action="skipped_record",
                details={"actual_type": type(record).__name__},
            )
        )
    return normalized_records


def _select_value_source(root: Path) -> tuple[list[dict], str | None]:
    primary = root / "unique_values.json"
    if primary.exists():
        return _read_json(primary), primary.name

    fallback = root / "value_catalog.json"
    if fallback.exists():
        return _read_json(fallback), fallback.name

    return [], None


def _contains_reserved_separator(value: str | None) -> bool:
    return bool(value) and "::" in value


def _normalize_string(
    value,
    *,
    field_name: str,
    entity_key: str,
    validation_issues: list[ValidationIssue],
) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    validation_issues.append(
        ValidationIssue(
            issue_type="invalid_field_type",
            entity_key=entity_key,
            action="ignored_field",
            details={
                "field_name": field_name,
                "actual_type": type(value).__name__,
            },
        )
    )
    return None


def _normalize_string_list(
    value,
    *,
    field_name: str,
    entity_key: str,
    validation_issues: list[ValidationIssue],
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        validation_issues.append(
            ValidationIssue(
                issue_type="coerced_scalar_list_field",
                entity_key=entity_key,
                action="coerced_scalar_to_singleton_list",
                details={"field_name": field_name},
            )
        )
        return [normalized] if normalized else []
    if not isinstance(value, list):
        validation_issues.append(
            ValidationIssue(
                issue_type="invalid_list_field_type",
                entity_key=entity_key,
                action="dropped_field",
                details={
                    "field_name": field_name,
                    "actual_type": type(value).__name__,
                },
            )
        )
        return []

    normalized_items: list[str] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            normalized = item.strip()
            if normalized:
                normalized_items.append(normalized)
            continue
        validation_issues.append(
            ValidationIssue(
                issue_type="invalid_list_field_item",
                entity_key=entity_key,
                action="dropped_list_item",
                details={
                    "field_name": field_name,
                    "index": index,
                    "actual_type": type(item).__name__,
                },
            )
        )
    return dedupe_terms(normalized_items)


def _resolve_column_identity(
    table_name: str | None,
    column_name: str | None,
    raw_column_key: str | None,
) -> tuple[str | None, str | None]:
    if column_name:
        return table_name, column_name
    if not isinstance(raw_column_key, str):
        return table_name, column_name
    if "::" in raw_column_key:
        embedded_table_name, embedded_column_name = raw_column_key.split("::", maxsplit=1)
        return table_name or embedded_table_name, embedded_column_name
    return table_name, raw_column_key


def _infer_table_name(
    column_name: str,
    known_columns: set[ColumnRef],
) -> str | None:
    matches = sorted({ref.table_name for ref in known_columns if ref.column_name == column_name})
    if len(matches) == 1:
        return matches[0]
    return None


def _normalize_rule_priority(
    raw_priority,
    *,
    rule_id: str,
    validation_issues: list[ValidationIssue],
) -> int:
    if raw_priority is None:
        return 0
    try:
        return int(raw_priority)
    except (TypeError, ValueError):
        validation_issues.append(
            ValidationIssue(
                issue_type="invalid_rule_priority",
                entity_key=rule_id,
                action="defaulted_priority",
                details={"actual_type": type(raw_priority).__name__},
            )
        )
        return 0


def _safe_list_length(value) -> int:
    return len(value) if isinstance(value, list) else 0


class DemoJsonAdapter:
    name = "demo_json"

    def load(
        self,
        root: Path,
        resource_owner: str,
        resource_namespace: str,
    ) -> MetadataCatalog:
        # This adapter preserves the current demo file layout while translating
        # it into the canonical internal catalog contract used by the pipeline.
        column_path = root / "column_descriptions.json"
        if not column_path.exists():
            raise FileNotFoundError(
                f"Required metadata file is missing: {column_path}"
            )

        validation_issues: list[ValidationIssue] = []
        raw_columns = _normalize_record_list(
            _read_json(column_path),
            payload_name="columns",
            wrapper_keys=("columns",),
            validation_issues=validation_issues,
        )
        raw_value_payload, value_source = _select_value_source(root)
        raw_value_groups = _normalize_record_list(
            raw_value_payload,
            payload_name="value_groups",
            wrapper_keys=("value_groups", "values"),
            validation_issues=validation_issues,
        )
        raw_rules = _normalize_record_list(
            _read_optional_json(root / "rules.json", default=[]),
            payload_name="rules",
            wrapper_keys=("rules",),
            validation_issues=validation_issues,
        )

        columns: list[ColumnSpec] = []
        known_columns: set[ColumnRef] = set()
        seen_columns: set[ColumnRef] = set()

        for item in raw_columns:
            raw_table_name = (
                item.get("table_name", item.get("table"))
                if isinstance(item, dict)
                else None
            )
            if raw_table_name is None:
                raw_table_name = item.get("source_table")
            raw_table_name = _normalize_string(
                raw_table_name,
                field_name="table_name",
                entity_key="column_definition",
                validation_issues=validation_issues,
            )
            raw_column_key = _normalize_string(
                item.get("column_key"),
                field_name="column_key",
                entity_key="column_definition",
                validation_issues=validation_issues,
            )
            column_name = _normalize_string(
                item.get("column_name"),
                field_name="column_name",
                entity_key="column_definition",
                validation_issues=validation_issues,
            )
            table_name, column_name = _resolve_column_identity(
                raw_table_name,
                column_name,
                raw_column_key,
            )
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
            if _contains_reserved_separator(table_name) or _contains_reserved_separator(
                column_name
            ):
                validation_issues.append(
                    ValidationIssue(
                        issue_type="invalid_identifier_separator",
                        entity_key=f"{table_name}::{column_name}",
                        action="skipped_column",
                    )
                )
                continue
            column_ref = ColumnRef(table_name=table_name, column_name=column_name)

            if column_ref in seen_columns:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="duplicate_column_identity",
                        entity_key=column_ref.log_label,
                        action="skipped_duplicate_column",
                    )
                )
                continue

            columns.append(
                ColumnSpec(
                    table_name=table_name,
                    column_name=column_name,
                    raw_column_name=_normalize_string(
                        item.get("raw_column_name"),
                        field_name="raw_column_name",
                        entity_key=column_ref.log_label,
                        validation_issues=validation_issues,
                    ),
                    description=_normalize_string(
                        item.get("description"),
                        field_name="description",
                        entity_key=column_ref.log_label,
                        validation_issues=validation_issues,
                    )
                    or "",
                    aliases=_normalize_string_list(
                        item.get("aliases"),
                        field_name="aliases",
                        entity_key=column_ref.log_label,
                        validation_issues=validation_issues,
                    ),
                    data_type=_normalize_string(
                        item.get("data_type"),
                        field_name="data_type",
                        entity_key=column_ref.log_label,
                        validation_issues=validation_issues,
                    ),
                )
            )
            seen_columns.add(column_ref)
            known_columns.add(column_ref)

        values: list[ValueSpec] = []
        seen_value_keys: set[tuple[ColumnRef, str]] = set()
        for group in raw_value_groups:
            raw_table_name = (
                group.get("table_name", group.get("table"))
                if isinstance(group, dict)
                else None
            )
            if raw_table_name is None:
                raw_table_name = group.get("source_table")
            raw_table_name = _normalize_string(
                raw_table_name,
                field_name="table_name",
                entity_key="value_definition",
                validation_issues=validation_issues,
            )
            raw_column_key = _normalize_string(
                group.get("column_key"),
                field_name="column_key",
                entity_key="value_definition",
                validation_issues=validation_issues,
            )
            column_name = _normalize_string(
                group.get("column_name"),
                field_name="column_name",
                entity_key="value_definition",
                validation_issues=validation_issues,
            )
            table_name, column_name = _resolve_column_identity(
                raw_table_name,
                column_name,
                raw_column_key,
            )
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
                inferred_table_name = _infer_table_name(column_name, known_columns)
                if inferred_table_name:
                    table_name = inferred_table_name
                    validation_issues.append(
                        ValidationIssue(
                            issue_type="inferred_value_group_table_name",
                            entity_key=column_name,
                            action="accepted_legacy_value_group",
                            details={"table_name": inferred_table_name},
                        )
                    )
                else:
                    validation_issues.append(
                        ValidationIssue(
                            issue_type="missing_value_group_table_name",
                            entity_key=column_name,
                            action="skipped_value_group",
                        )
                    )
                    continue
            if _contains_reserved_separator(table_name) or _contains_reserved_separator(
                column_name
            ):
                validation_issues.append(
                    ValidationIssue(
                        issue_type="invalid_identifier_separator",
                        entity_key=f"{table_name}::{column_name}",
                        action="skipped_value_group",
                    )
                )
                continue
            column_ref = ColumnRef(table_name=table_name, column_name=column_name)
            if column_ref not in known_columns:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="missing_value_column",
                        entity_key=column_ref.log_label,
                        action="skipped_value_group",
                        details={
                            "value_count": _safe_list_length(group.get("values", [])),
                        },
                    )
                )
                continue

            raw_value_specs = _normalize_record_list(
                group.get("values", []),
                payload_name="values",
                wrapper_keys=(),
                validation_issues=validation_issues,
            )
            for raw_value_spec in raw_value_specs:
                raw_value = raw_value_spec.get("raw_value")
                if raw_value is None or raw_value == "":
                    validation_issues.append(
                        ValidationIssue(
                            issue_type="missing_raw_value",
                            entity_key=column_ref.log_label,
                            action="skipped_value",
                        )
                    )
                    continue

                value_key = (column_ref, raw_value)
                if value_key in seen_value_keys:
                    validation_issues.append(
                        ValidationIssue(
                            issue_type="duplicate_value_key",
                            entity_key=f"{column_ref.log_label}::{raw_value}",
                            action="skipped_duplicate_value",
                        )
                    )
                    continue

                values.append(
                    ValueSpec(
                        table_name=column_ref.table_name,
                        column_name=column_ref.column_name,
                        raw_value=str(raw_value),
                        value_gloss=_normalize_string(
                            raw_value_spec.get("value_gloss")
                            or raw_value_spec.get("value_description"),
                            field_name="value_gloss",
                            entity_key=column_ref.log_label,
                            validation_issues=validation_issues,
                        )
                        or "",
                        synonyms=_normalize_string_list(
                            raw_value_spec.get("synonyms"),
                            field_name="synonyms",
                            entity_key=column_ref.log_label,
                            validation_issues=validation_issues,
                        ),
                        business_tags=_normalize_string_list(
                            raw_value_spec.get("business_tags"),
                            field_name="business_tags",
                            entity_key=column_ref.log_label,
                            validation_issues=validation_issues,
                        ),
                    )
                )
                seen_value_keys.add(value_key)

        rules: list[RuleSpec] = []
        seen_rule_ids: set[str] = set()
        for raw_rule in raw_rules:
            rule_id = _normalize_string(
                raw_rule.get("rule_id"),
                field_name="rule_id",
                entity_key="rule_definition",
                validation_issues=validation_issues,
            )
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

            raw_candidate_columns = raw_rule.get("candidate_columns", []) or []
            if isinstance(raw_candidate_columns, (str, dict)):
                validation_issues.append(
                    ValidationIssue(
                        issue_type="coerced_rule_candidate_columns",
                        entity_key=rule_id,
                        action="coerced_scalar_to_singleton_list",
                    )
                )
                raw_candidate_columns = [raw_candidate_columns]
            elif not isinstance(raw_candidate_columns, list):
                validation_issues.append(
                    ValidationIssue(
                        issue_type="invalid_rule_candidate_columns_payload",
                        entity_key=rule_id,
                        action="dropped_candidate_columns",
                        details={"actual_type": type(raw_candidate_columns).__name__},
                    )
                )
                raw_candidate_columns = []
            seen_candidate_markers: set[str] = set()
            candidate_columns = []
            for candidate_column in raw_candidate_columns:
                marker = (
                    json.dumps(candidate_column, sort_keys=True)
                    if isinstance(candidate_column, dict)
                    else str(candidate_column)
                )
                if marker in seen_candidate_markers:
                    continue
                candidate_columns.append(candidate_column)
                seen_candidate_markers.add(marker)
            valid_candidate_columns: list[ColumnRef] = []
            for candidate_column in candidate_columns:
                if isinstance(candidate_column, dict):
                    table_name = _normalize_string(
                        candidate_column.get("table_name"),
                        field_name="table_name",
                        entity_key=rule_id,
                        validation_issues=validation_issues,
                    )
                    column_name = _normalize_string(
                        candidate_column.get("column_name"),
                        field_name="column_name",
                        entity_key=rule_id,
                        validation_issues=validation_issues,
                    )
                    if table_name and column_name:
                        candidate_ref = ColumnRef(
                            table_name=table_name,
                            column_name=column_name,
                        )
                    else:
                        validation_issues.append(
                            ValidationIssue(
                                issue_type="malformed_rule_candidate_column",
                                entity_key=rule_id,
                                action="dropped_candidate_column",
                                details={"candidate_column": candidate_column},
                            )
                        )
                        continue
                elif isinstance(candidate_column, str) and "::" in candidate_column:
                    table_name, column_name = candidate_column.split("::", maxsplit=1)
                    candidate_ref = ColumnRef(
                        table_name=table_name,
                        column_name=column_name,
                    )
                elif isinstance(candidate_column, str) and "." in candidate_column:
                    table_name, column_name = candidate_column.rsplit(".", maxsplit=1)
                    candidate_ref = ColumnRef(
                        table_name=table_name,
                        column_name=column_name,
                    )
                elif isinstance(candidate_column, str):
                    validation_issues.append(
                        ValidationIssue(
                            issue_type="unqualified_rule_candidate_column",
                            entity_key=rule_id,
                            action="dropped_candidate_column",
                            details={"column_name": candidate_column},
                        )
                    )
                    continue
                else:
                    validation_issues.append(
                        ValidationIssue(
                            issue_type="malformed_rule_candidate_column",
                            entity_key=rule_id,
                            action="dropped_candidate_column",
                            details={
                                "candidate_column": candidate_column,
                                "actual_type": type(candidate_column).__name__,
                            },
                        )
                    )
                    continue

                if candidate_ref in known_columns:
                    valid_candidate_columns.append(candidate_ref)
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
                    description=_normalize_string(
                        raw_rule.get("text_semantic") or raw_rule.get("description"),
                        field_name="description",
                        entity_key=rule_id,
                        validation_issues=validation_issues,
                    )
                    or "",
                    candidate_columns=valid_candidate_columns,
                    trigger_terms=_normalize_string_list(
                        raw_rule.get("trigger_terms"),
                        field_name="trigger_terms",
                        entity_key=rule_id,
                        validation_issues=validation_issues,
                    ),
                    intent=_normalize_string(
                        raw_rule.get("intent"),
                        field_name="intent",
                        entity_key=rule_id,
                        validation_issues=validation_issues,
                    ),
                    priority=_normalize_rule_priority(
                        raw_rule.get("priority"),
                        rule_id=rule_id,
                        validation_issues=validation_issues,
                    ),
                    rule_text=_normalize_string(
                        raw_rule.get("rule_text"),
                        field_name="rule_text",
                        entity_key=rule_id,
                        validation_issues=validation_issues,
                    ),
                )
            )
            seen_rule_ids.add(rule_id)

        return MetadataCatalog(
            resource_owner=resource_owner,
            resource_namespace=resource_namespace,
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
                "values": sum(
                    _safe_list_length(group.get("values", []))
                    for group in raw_value_groups
                ),
                "rules": len(raw_rules),
            },
            validation_issues=validation_issues,
        )


def get_source_adapter(name: str) -> SourceAdapter:
    normalized_name = name.strip().lower()
    if normalized_name in {"demo_json", "demo"}:
        return DemoJsonAdapter()
    raise ValueError(f"Unsupported source adapter: {name}")
