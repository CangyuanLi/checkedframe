from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Optional


class ColumnNotFoundError(Exception):
    """Raised when a required column is not found in the Schema."""


class ValidationError(Exception):
    """Raised when a check fails."""

    def __init__(self, check):
        self.check = check

        super().__init__(f"{check.name} failed: {check.description}")


@dataclasses.dataclass
class _ErrorStore:
    missing_column: Optional[ColumnNotFoundError] = None
    invalid_nulls: Optional[ValueError] = None
    invalid_dtype: Optional[TypeError] = None
    failed_checks: list[ValidationError] = dataclasses.field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.missing_column is not None
            or self.invalid_nulls is not None
            or self.invalid_dtype is not None
            or len(self.failed_checks) > 0
        )


class SchemaError(Exception):
    """Raised when the given DataFrame does not match the given Schema."""

    def __init__(
        self, errors: Mapping[str, _ErrorStore], failed_checks: list[ValidationError]
    ):
        self.errors = errors
        self.failed_checks = failed_checks

        super().__init__(self)

    def __str__(self) -> str:
        def _wrap_err(e) -> str:
            return f"    - {e}"

        output = []
        total_error_count = 0
        for col, error_store in self.errors.items():
            if error_store.is_empty():
                continue

            bullets = []
            error_count = 0

            if (e := error_store.missing_column) is not None:
                bullets.append(_wrap_err(e))
                error_count += 1

            if (e := error_store.invalid_dtype) is not None:
                bullets.append(_wrap_err(e))
                error_count += 1

            if (e := error_store.invalid_nulls) is not None:
                bullets.append(_wrap_err(e))
                error_count += 1

            if len(error_store.failed_checks) > 0:
                for fc in error_store.failed_checks:
                    bullets.append(_wrap_err(fc))
                    error_count += 1

            output.append(f"  {col}: {error_count} error(s)")
            output.extend(bullets)

            total_error_count += error_count

        for fc in self.failed_checks:
            output.append(f"  * {fc}")
            total_error_count += 1

        error_summary = [f"Found {total_error_count} error(s)"]

        return "\n".join(error_summary + output)

    def is_empty(self) -> bool:
        if len(self.failed_checks) > 0:
            return False

        for k, e in self.errors.items():
            if not e.is_empty():
                return False

        return True
