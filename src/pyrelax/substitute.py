"""
Generic substitution over the AST.

RelaX-style ``nome = expr`` assignments are a *name for an expression*,
not a materialised, independently-aliased relation — referencing ``x``
later should behave exactly as if you'd pasted ``x``'s definition in
its place (this is what the reference RelaX calculator's own
"replaceVariables" step does). That's what makes

    x = sigma grade >= 7 enrollment
    y = student left outer join student.student_id = enrollment.student_id (x)

behave the same as writing the join directly against
``sigma grade >= 7 enrollment`` inline: the alias ``enrollment`` used in
the join condition has to still resolve to an actual ``Relation("enrollment")``
node somewhere inside whatever `x` expands to.

``substitute`` walks any dataclass-based AST (works for both ``RelExpr``
and ``ValueExpr`` trees, and any structure combining them) and replaces
every ``ast_nodes.Relation`` leaf whose name is a key in ``env`` with the
(already-substituted) subtree stored there. Names that aren't in ``env``
are left alone — those are real base tables.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

from . import ast_nodes as ast


def substitute(node: Any, env: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Relation) and node.name in env:
        return env[node.name]

    if dataclasses.is_dataclass(node):
        changes = {}
        for f in dataclasses.fields(node):
            old_val = getattr(node, f.name)
            new_val = _substitute_value(old_val, env)
            if new_val is not old_val:
                changes[f.name] = new_val
        return dataclasses.replace(node, **changes) if changes else node

    return node


def _substitute_value(value: Any, env: Dict[str, Any]) -> Any:
    if dataclasses.is_dataclass(value):
        return substitute(value, env)
    if isinstance(value, list):
        new_items = [_substitute_value(v, env) for v in value]
        return new_items if any(a is not b for a, b in zip(new_items, value)) else value
    return value
