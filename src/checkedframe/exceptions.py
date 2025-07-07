from __future__ import annotations

from typing import Any


class CastError(Exception):
    def __init__(self, msg: str, element_passes: Any):
        super().__init__(msg)

        self.msg = msg
        self.element_passes = element_passes


class SchemaError(Exception):
    """Raised when the given DataFrame does not match the given Schema."""
