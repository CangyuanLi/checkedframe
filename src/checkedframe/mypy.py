from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

from mypy.nodes import CallExpr, Decorator, MemberExpr, NameExpr, TypeInfo
from mypy.plugin import (
    ClassDefContext,
    MethodContext,
    MethodSigContext,
    Plugin,
    SemanticAnalyzerPluginInterface,
)

SCHEMA_FULLNAME = "checkedframe._core.Schema"
CHECK_DECORATOR_FULLNAME = "checkedframe._checks.Check"


def mark_checks_as_staticmethod(ctx: ClassDefContext) -> None:
    for sym in ctx.cls.info.names.values():
        if not isinstance(sym.node, Decorator):
            continue

        for decorator in sym.node.original_decorators:
            if isinstance(decorator, CallExpr):
                dec = decorator.callee

                if not isinstance(dec, (NameExpr, MemberExpr)):
                    continue
            elif isinstance(decorator, (NameExpr, MemberExpr)):
                dec = decorator
            else:
                continue

            dec_name = dec.fullname

            if dec_name == CHECK_DECORATOR_FULLNAME:
                sym.node.func.is_static = True


class CheckedframePlugin(Plugin):
    def __init__(self, options) -> None:
        super().__init__(options)

    def get_base_class_hook(
        self, fullname: str
    ) -> Callable[[ClassDefContext], None] | None:
        sym = self.lookup_fully_qualified(fullname)
        if sym is not None and isinstance(sym.node, TypeInfo):
            if any(base.fullname == SCHEMA_FULLNAME for base in sym.node.mro):

                def _hook(ctx: ClassDefContext) -> None:
                    mark_checks_as_staticmethod(ctx)

                return _hook

        return None


def plugin(version: str) -> type[Plugin]:
    return CheckedframePlugin
