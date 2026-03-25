from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def build_column_key(table_name: str, column_name: str) -> str:
    # `table_name::column_name` is the stable identifier inside one namespace.
    # Keep this format simple and explicit so it can be reused across loaders,
    # prompt bundles, rule metadata, and future external adapters.
    return f"{table_name}::{column_name}"


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
    def column_key(self) -> str:
        return build_column_key(self.table_name, self.column_name)


@dataclass(frozen=True)
class ValueSpec:
    table_name: str
    column_name: str
    raw_value: str
    value_gloss: str
    synonyms: list[str] = field(default_factory=list)
    business_tags: list[str] = field(default_factory=list)

    @property
    def column_key(self) -> str:
        return build_column_key(self.table_name, self.column_name)


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    description: str
    candidate_columns: list[str] = field(default_factory=list)
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
    # A catalog is namespace-scoped on purpose so one database can host many
    # datasource-specific metadata graphs without global name collisions.
    namespace: str
    columns: list[ColumnSpec]
    values: list[ValueSpec]
    rules: list[RuleSpec]
    source_files: dict[str, str | None] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    validation_issues: list[ValidationIssue] = field(default_factory=list)
