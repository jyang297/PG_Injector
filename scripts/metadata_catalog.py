from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def log_column_identity(table_name: str, column_name: str) -> str:
    # Keep a compact `table::column` label for logs and human debugging only.
    # It is no longer part of the persisted source-of-truth contract.
    return f"{table_name}::{column_name}"


@dataclass(frozen=True, order=True)
class ColumnRef:
    table_name: str
    column_name: str

    @property
    def log_label(self) -> str:
        return log_column_identity(self.table_name, self.column_name)


@dataclass(frozen=True)
class ColumnSpec:
    # These dataclasses are the stable internal ingestion contract. Adapters may
    # change as raw source formats evolve, but the loader/chunk builder should
    # only depend on these normalized fields.
    table_name: str
    column_name: str
    description: str
    raw_column_name: str | None = None
    aliases: list[str] = field(default_factory=list)
    data_type: str | None = None

    @property
    def ref(self) -> ColumnRef:
        return ColumnRef(self.table_name, self.column_name)


@dataclass(frozen=True)
class ValueSpec:
    table_name: str
    column_name: str
    raw_value: str
    value_gloss: str
    synonyms: list[str] = field(default_factory=list)
    business_tags: list[str] = field(default_factory=list)

    @property
    def ref(self) -> ColumnRef:
        return ColumnRef(self.table_name, self.column_name)


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    description: str
    candidate_columns: list[ColumnRef] = field(default_factory=list)
    trigger_terms: list[str] = field(default_factory=list)
    intent: str | None = None
    priority: int = 0
    rule_text: str | None = None


@dataclass(frozen=True)
class ValidationIssue:
    issue_type: str
    entity_key: str
    action: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetadataCatalog:
    # The loader works inside one owner + namespace scope so one database can
    # host many tenant/project-specific metadata graphs without global name
    # collisions.
    resource_owner: str
    resource_namespace: str
    columns: list[ColumnSpec]
    values: list[ValueSpec]
    rules: list[RuleSpec]
    source_files: dict[str, str | None] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    validation_issues: list[ValidationIssue] = field(default_factory=list)
