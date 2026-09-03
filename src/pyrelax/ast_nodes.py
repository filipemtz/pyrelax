"""
AST node definitions.

Two small families of nodes:

  * ``ValueExpr``   — scalar expressions used inside σ / π / γ / join
                       conditions (constants, columns, function calls).
  * ``RelExpr``      — relational-algebra expressions (π σ ρ τ γ, joins,
                       set operators, ...), built out of ``ValueExpr``
                       where needed.

Every node is a plain, immutable-by-convention ``@dataclass``. This gives
you free ``__repr__``/``__eq__``, static typing, and a natural place for
the lark ``Transformer`` to build results incrementally (one method per
node type — see ``relalg/parser.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional
from typing import Union as _TypingUnion

# ==========================================================================
# Value expressions
# ==========================================================================


@dataclass(frozen=True)
class Const:
    """A literal: number, string, boolean or NULL."""
    value: Any
    datatype: str = "null"  # 'number' | 'string' | 'boolean' | 'null'


@dataclass(frozen=True)
class Column:
    """A column reference, optionally qualified by a relation alias
    (``alias.name``); also doubles as a plain projection item."""
    name: str
    rel_alias: Optional[str] = None


@dataclass(frozen=True)
class Call:
    """A function/operator call, e.g. ``add(a, b)``, ``upper(x)``,
    ``and(a, b)``, ``caseWhen(c1, r1, c2, r2, ...)``."""
    func: str
    args: List["ValueExpr"] = field(default_factory=list)


ValueExpr = _TypingUnion[Const, Column, Call]


# ==========================================================================
# Projection items (π argument list)
# ==========================================================================


@dataclass(frozen=True)
class Star:
    """``*`` or ``alias.*`` inside a projection list."""
    rel_alias: Optional[str] = None


@dataclass(frozen=True)
class NamedExpr:
    """A projected, possibly-renamed, derived value: ``expr -> name``."""
    expr: ValueExpr
    name: Optional[str] = None


ProjectionItem = _TypingUnion[Star, Column, NamedExpr]


# ==========================================================================
# ρ (rename) arguments
# ==========================================================================


@dataclass(frozen=True)
class ColAssignment:
    """One ``dst <- src`` pair inside ``ρ dst<-src, ... (R)``."""
    dst: str
    src: Column


# ==========================================================================
# τ (order by) arguments
# ==========================================================================


@dataclass(frozen=True)
class OrderArg:
    col: Column
    ascending: bool = True


# ==========================================================================
# γ (group by) arguments
# ==========================================================================


@dataclass(frozen=True)
class AggCall:
    """One aggregate, e.g. ``SUM(qty) -> total`` or ``COUNT(*) -> n``."""
    agg_func: str  # 'SUM' | 'COUNT' | 'COUNT_ALL' | 'AVG' | 'MIN' | 'MAX'
    col: Optional[Column]
    name: str


# ==========================================================================
# Relational-algebra expressions
# ==========================================================================


@dataclass(frozen=True)
class Relation:
    name: str


@dataclass(frozen=True)
class TableDum:
    """``{}`` — zero columns, zero rows."""


@dataclass(frozen=True)
class TableDee:
    """``{()}`` — zero columns, one row."""


@dataclass(frozen=True)
class Projection:
    child: "RelExpr"
    columns: List[ProjectionItem]


@dataclass(frozen=True)
class Selection:
    child: "RelExpr"
    cond: ValueExpr


@dataclass(frozen=True)
class RenameRelation:
    child: "RelExpr"
    new_alias: str


@dataclass(frozen=True)
class RenameColumns:
    child: "RelExpr"
    assignments: List[ColAssignment]


@dataclass(frozen=True)
class GroupBy:
    child: "RelExpr"
    group: List[Column]
    aggregate: List[AggCall]


@dataclass(frozen=True)
class OrderBy:
    child: "RelExpr"
    args: List[OrderArg]


@dataclass(frozen=True)
class Union:
    left: "RelExpr"
    right: "RelExpr"


@dataclass(frozen=True)
class Intersect:
    left: "RelExpr"
    right: "RelExpr"


@dataclass(frozen=True)
class Difference:
    left: "RelExpr"
    right: "RelExpr"


@dataclass(frozen=True)
class CrossJoin:
    left: "RelExpr"
    right: "RelExpr"


@dataclass(frozen=True)
class ThetaJoin:
    left: "RelExpr"
    right: "RelExpr"
    cond: ValueExpr


@dataclass(frozen=True)
class NaturalJoin:
    left: "RelExpr"
    right: "RelExpr"


@dataclass(frozen=True)
class LeftOuterJoin:
    left: "RelExpr"
    right: "RelExpr"
    cond: Optional[ValueExpr] = None


@dataclass(frozen=True)
class RightOuterJoin:
    left: "RelExpr"
    right: "RelExpr"
    cond: Optional[ValueExpr] = None


@dataclass(frozen=True)
class FullOuterJoin:
    left: "RelExpr"
    right: "RelExpr"
    cond: Optional[ValueExpr] = None


@dataclass(frozen=True)
class LeftSemiJoin:
    left: "RelExpr"
    right: "RelExpr"


@dataclass(frozen=True)
class RightSemiJoin:
    left: "RelExpr"
    right: "RelExpr"


@dataclass(frozen=True)
class AntiJoin:
    left: "RelExpr"
    right: "RelExpr"
    cond: Optional[ValueExpr] = None


@dataclass(frozen=True)
class Division:
    left: "RelExpr"
    right: "RelExpr"


RelExpr = _TypingUnion[
    Relation, TableDum, TableDee, Projection, Selection, RenameRelation,
    RenameColumns, GroupBy, OrderBy, Union, Intersect, Difference,
    CrossJoin, ThetaJoin, NaturalJoin, LeftOuterJoin, RightOuterJoin,
    FullOuterJoin, LeftSemiJoin, RightSemiJoin, AntiJoin, Division,
]

# Node types whose *own* table alias(es) stay visible to whatever composes
# them directly in a FROM clause (used by the SQL compiler).
FUSIBLE_TYPES = (
    Relation, RenameRelation, Selection, CrossJoin, ThetaJoin, NaturalJoin,
    LeftOuterJoin, RightOuterJoin, FullOuterJoin, LeftSemiJoin,
    RightSemiJoin, AntiJoin,
)
