"""
Compiles a relational-algebra AST (``ast_nodes.py``) into DuckDB SQL text.

Design
------
DuckDB already implements everything hard about relational algebra
correctly and efficiently — joins (incl. NATURAL/SEMI/ANTI/OUTER),
set operators, NULL-aware three-valued logic, ``SELECT * RENAME/EXCLUDE``.
So rather than re-implementing joins/aggregation by hand (as a pandas-only
engine must), this compiler just *translates* each AST node to the SQL
construct that already means the same thing, and lets DuckDB execute it.

Two kinds of node, for alias visibility
----------------------------------------
Relational-algebra lets a σ or a join condition refer to a relation alias
established *anywhere below it in the same "join scope"* (e.g. ``σ
L.a = R.b (L ⨝ R)``). SQL only offers that visibility within a single
``SELECT``'s FROM clause. So:

* "fusible" nodes (``Relation``, ``RenameRelation``, ``Selection``, and
  every join type — see ``ast_nodes.FUSIBLE_TYPES``) are compiled by
  ``compile_from()`` into a *FROM-clause fragment*, keeping every
  relation alias they establish visible to whatever composes them.
* everything else (``Projection``, ``RenameColumns``, ``GroupBy``,
  ``OrderBy``, set operators, ``Division``, the degenerate tables) is a
  "boundary": once you project/aggregate/etc., the original aliases are
  gone anyway (real RA semantics), so these are compiled by
  ``compile_select()`` into a complete, self-contained ``SELECT`` and,
  when they're used as an operand of a join above them, wrapped in a
  parenthesised subquery with a fresh alias.

The one operator SQL has no equivalent for is relational **division**;
that is generated using the classic double-``NOT EXISTS`` idiom, which
needs to know the dividend/divisor's column *names* — cheaply obtained
from DuckDB itself via ``relation.columns`` (a schema lookup, not an
execution) rather than tracked by hand.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import duckdb

from . import ast_nodes as ast
from .errors import RelAlgExecutionError

_alias_counter = itertools.count()


def _new_alias() -> str:
    return f"_t{next(_alias_counter)}"


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _and(clauses: List[str]) -> str:
    return " AND ".join(f"({c})" for c in clauses)


# ==========================================================================
# value expressions -> SQL
# ==========================================================================

_CAST_TYPES = {"number": "DOUBLE", "string": "VARCHAR", "date": "DATE", "boolean": "BOOLEAN"}

_UNARY_SQL_FUNC = {
    "upper": "UPPER", "ucase": "UPPER", "lower": "LOWER", "lcase": "LOWER",
    "reverse": "REVERSE", "length": "LENGTH", "abs": "ABS", "floor": "FLOOR",
    "ceil": "CEIL", "round": "ROUND", "sqrt": "SQRT", "exp": "EXP", "ln": "LN",
}
_DATE_PART_FUNC = {
    "year": "year", "month": "month", "day": "day", "dayofmonth": "day",
    "hour": "hour", "minute": "minute", "second": "second",
}


def compile_value(node: ast.ValueExpr) -> str:
    if isinstance(node, ast.Const):
        return _compile_const(node)
    if isinstance(node, ast.Column):
        return _compile_column_ref(node)
    if isinstance(node, ast.Call):
        return _compile_call(node)
    raise RelAlgExecutionError(f"cannot compile value expression: {node!r}")


def _compile_const(node: ast.Const) -> str:
    if node.value is None:
        return "NULL"
    if node.datatype == "boolean":
        return "TRUE" if node.value else "FALSE"
    if node.datatype == "number":
        return repr(node.value)
    return quote_literal(node.value)


def _compile_column_ref(node: ast.Column) -> str:
    if node.rel_alias:
        return f"{quote_ident(node.rel_alias)}.{quote_ident(node.name)}"
    return quote_ident(node.name)


def _compile_call(node: ast.Call) -> str:
    f = node.func
    args = node.args

    if f in ("and", "or"):
        a, b = (compile_value(x) for x in args)
        return f"({a} {f.upper()} {b})"
    if f == "xor":
        a, b = (compile_value(x) for x in args)
        return f"({a} <> {b})"  # NULL-propagating boolean xor
    if f == "not":
        return f"(NOT {compile_value(args[0])})"
    if f == "minus":
        return f"(-{compile_value(args[0])})"
    if f == "concat":
        return "(" + " || ".join(compile_value(a) for a in args) + ")"

    if f in ("=", "!=", ">", "<", ">=", "<="):
        left, right = args
        if f in ("=", "!=") and isinstance(right, ast.Const) and right.value is None:
            return f"({compile_value(left)} IS {'NOT ' if f == '!=' else ''}NULL)"
        return f"({compile_value(left)} {f} {compile_value(right)})"

    if f == "like":
        return f"({compile_value(args[0])} LIKE {compile_value(args[1])})"
    if f == "ilike":
        return f"({compile_value(args[0])} ILIKE {compile_value(args[1])})"
    if f == "regexp":
        return f"regexp_matches({compile_value(args[0])}, {compile_value(args[1])})"

    if f == "between":
        a, lo, hi = (compile_value(x) for x in args)
        return f"({a} BETWEEN {lo} AND {hi})"
    if f == "notBetween":
        a, lo, hi = (compile_value(x) for x in args)
        return f"({a} NOT BETWEEN {lo} AND {hi})"

    if f in ("caseWhen", "caseWhenElse"):
        parts = ["CASE"]
        pairs = list(zip(args[0::2], args[1::2]))
        for cond, res in pairs[:-1] if f == "caseWhenElse" else pairs:
            parts.append(f"WHEN {compile_value(cond)} THEN {compile_value(res)}")
        if f == "caseWhenElse":
            parts.append(f"ELSE {compile_value(pairs[-1][1])}")
        parts.append("END")
        return "(" + " ".join(parts) + ")"

    if f == "coalesce":
        return "COALESCE(" + ", ".join(compile_value(a) for a in args) + ")"
    if f == "replace":
        return "REPLACE(" + ", ".join(compile_value(a) for a in args) + ")"
    if f == "repeat":
        return "REPEAT(" + ", ".join(compile_value(a) for a in args) + ")"

    if f == "substring":
        if len(args) == 2:
            return f"SUBSTRING({compile_value(args[0])} FROM {compile_value(args[1])})"
        return (f"SUBSTRING({compile_value(args[0])} FROM {compile_value(args[1])} "
                f"FOR {compile_value(args[2])})")

    if f == "cast":
        target = args[1].value
        sql_type = _CAST_TYPES.get(target)
        if sql_type is None:
            raise RelAlgExecutionError(f'unknown CAST target type "{target}"')
        return f"CAST({compile_value(args[0])} AS {sql_type})"

    if f in _UNARY_SQL_FUNC:
        return f"{_UNARY_SQL_FUNC[f]}({compile_value(args[0])})"
    if f == "date":
        return f"CAST({compile_value(args[0])} AS DATE)"
    if f in _DATE_PART_FUNC:
        return f"date_part('{_DATE_PART_FUNC[f]}', {compile_value(args[0])})"
    if f == "adddate":
        return f"({compile_value(args[0])} + CAST({compile_value(args[1])} AS BIGINT) * INTERVAL 1 DAY)"
    if f == "subdate":
        return f"({compile_value(args[0])} - CAST({compile_value(args[1])} AS BIGINT) * INTERVAL 1 DAY)"

    if f in ("add", "sub", "mul", "div", "mod"):
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%"}[f]
        return f"({compile_value(args[0])} {op} {compile_value(args[1])})"
    if f == "power":
        return f"POWER({compile_value(args[0])}, {compile_value(args[1])})"
    if f == "log":
        return f"(LN({compile_value(args[1])}) / LN({compile_value(args[0])}))"

    if f == "rand":
        return "RANDOM()"
    if f in ("now", "current_timestamp", "sysdate", "transaction_timestamp",
             "statement_timestamp", "clock_timestamp"):
        return "NOW()"
    if f == "rownum":
        raise RelAlgExecutionError(
            "rownum() is only supported directly as a projected column, e.g. "
            '`\u03c0 rownum() -> n (R)`, not inside other expressions')

    raise RelAlgExecutionError(f'unsupported function/operator "{f}"')


# ==========================================================================
# relational-algebra nodes -> SQL
# ==========================================================================

class SqlCompiler:
    """Compiles a RelExpr AST to SQL text, given a live DuckDB connection
    (needed only to introspect column names for ``Division``)."""

    def __init__(self, con: "duckdb.DuckDBPyConnection"):
        self.con = con

    # ---- public entry point -----------------------------------------------
    def compile_select(self, node: ast.RelExpr) -> str:
        if isinstance(node, ast.Projection):
            return self._compile_projection(node)
        if isinstance(node, ast.RenameColumns):
            return self._compile_rename_columns(node)
        if isinstance(node, ast.GroupBy):
            return self._compile_group_by(node)
        if isinstance(node, ast.OrderBy):
            return self._compile_order_by(node)
        if isinstance(node, ast.Union):
            return self._compile_set_op(node, "UNION")
        if isinstance(node, ast.Intersect):
            return self._compile_set_op(node, "INTERSECT")
        if isinstance(node, ast.Difference):
            return self._compile_set_op(node, "EXCEPT")
        if isinstance(node, ast.Division):
            return self._compile_division(node)
        if isinstance(node, ast.TableDum):
            return 'SELECT NULL AS "_dummy" WHERE FALSE'
        if isinstance(node, ast.TableDee):
            return 'SELECT 1 AS "_dummy"'
        # fusible node used as a top-level query: SELECT * FROM <from-frag> [WHERE ...]
        from_frag, wheres = self.compile_from(node)
        sql = f"SELECT * FROM {from_frag}"
        if wheres:
            sql += f" WHERE {_and(wheres)}"
        return sql

    # ---- FROM-clause fragments (alias-preserving) -------------------------
    def compile_from(self, node: ast.RelExpr) -> Tuple[str, List[str]]:
        if isinstance(node, ast.Relation):
            return quote_ident(node.name), []

        if isinstance(node, ast.RenameRelation):
            child = node.child
            if isinstance(child, ast.FUSIBLE_TYPES):
                inner_frag, inner_wheres = self.compile_from(child)
                body = inner_frag if not inner_wheres else f"(SELECT * FROM {inner_frag} WHERE {_and(inner_wheres)})"
            else:
                body = f"({self.compile_select(child)})"  # no alias yet -- we supply our own below
            return f"{body} AS {quote_ident(node.new_alias)}", []

        if isinstance(node, ast.Selection):
            inner_frag, inner_wheres = self.compile_from(node.child)
            return inner_frag, inner_wheres + [compile_value(node.cond)]

        if isinstance(node, ast.CrossJoin):
            l_frag, l_wheres = self.compile_from(node.left)
            r_frag, r_wheres = self.compile_from(node.right)
            return f"{l_frag} CROSS JOIN {r_frag}", l_wheres + r_wheres

        if isinstance(node, ast.ThetaJoin):
            l_frag, l_wheres = self.compile_from(node.left)
            r_frag, r_wheres = self.compile_from(node.right)
            on_sql = compile_value(node.cond)
            return f"{l_frag} JOIN {r_frag} ON {on_sql}", l_wheres + r_wheres

        if isinstance(node, ast.NaturalJoin):
            l_frag, l_wheres = self.compile_from(node.left)
            r_frag, r_wheres = self.compile_from(node.right)
            return f"{l_frag} NATURAL JOIN {r_frag}", l_wheres + r_wheres

        if isinstance(node, (ast.LeftOuterJoin, ast.RightOuterJoin, ast.FullOuterJoin,
                              ast.LeftSemiJoin, ast.RightSemiJoin, ast.AntiJoin)):
            return self._compile_join_boundary(node), []

        # boundary node used directly as a FROM source
        return self._wrap_as_subquery(node), []

    # ---- outer / semi / anti joins ------------------------------------------
    # These always materialise their two operands first (any pending
    # selection conditions get applied *before* the join, since hoisting a
    # WHERE across an outer/semi/anti join would silently change its
    # semantics into an inner join). The join itself still contributes a
    # normal FROM fragment, so its own two aliases stay visible to anything
    # composed immediately around it (e.g. a σ wrapping the whole join).
    def _compile_join_boundary(self, node) -> str:
        left_body = self._materialize(node.left)
        right_body = self._materialize(node.right)

        if isinstance(node, ast.LeftOuterJoin):
            kw, cond = "LEFT OUTER JOIN", node.cond
        elif isinstance(node, ast.RightOuterJoin):
            kw, cond = "RIGHT OUTER JOIN", node.cond
        elif isinstance(node, ast.FullOuterJoin):
            kw, cond = "FULL OUTER JOIN", node.cond
        elif isinstance(node, ast.LeftSemiJoin):
            kw, cond = "SEMI JOIN", None
        elif isinstance(node, ast.RightSemiJoin):
            # DuckDB's SEMI JOIN keeps the *left* side's rows/columns; to get
            # a right-semi-join (keep the right side) we just swap operands.
            left_body, right_body = right_body, left_body
            kw, cond = "SEMI JOIN", None
        elif isinstance(node, ast.AntiJoin):
            kw, cond = "ANTI JOIN", node.cond
        else:  # pragma: no cover - exhaustive above
            raise RelAlgExecutionError(f"not a join boundary node: {node!r}")

        if cond is not None:
            return f"{left_body} {kw} {right_body} ON {compile_value(cond)}"
        if kw in ("SEMI JOIN", "ANTI JOIN"):
            return f"{left_body} NATURAL {kw} {right_body}"
        return f"{left_body} NATURAL {kw} {right_body}"

    def _materialize(self, node: ast.RelExpr) -> str:
        """Compile `node` into something directly usable as a join operand,
        applying any pending selection conditions first. Reuses the child's
        own alias (base relation name / ρ target) when there's an obvious
        single one, so σ-conditions above still read naturally; falls back
        to a synthetic alias otherwise."""
        frag, wheres = self.compile_from(node)
        if not wheres:
            return frag
        alias = self._single_root_alias(node) or _new_alias()
        return f"(SELECT * FROM {frag} WHERE {_and(wheres)}) AS {quote_ident(alias)}"

    @staticmethod
    def _single_root_alias(node: ast.RelExpr) -> Optional[str]:
        while isinstance(node, ast.Selection):
            node = node.child
        if isinstance(node, ast.Relation):
            return node.name
        if isinstance(node, ast.RenameRelation):
            return node.new_alias
        return None

    def _wrap_as_subquery(self, node: ast.RelExpr) -> str:
        return f"({self.compile_select(node)}) AS {quote_ident(_new_alias())}"

    # ---- projection ----------------------------------------------------
    def _compile_projection(self, node: ast.Projection) -> str:
        from_frag, wheres = self.compile_from(node.child)
        items = []
        for col in node.columns:
            if isinstance(col, ast.Star):
                items.append(f"{quote_ident(col.rel_alias)}.*" if col.rel_alias else "*")
            elif isinstance(col, ast.Column):
                items.append(_compile_column_ref(col))
            elif isinstance(col, ast.NamedExpr):
                if isinstance(col.expr, ast.Call) and col.expr.func == "rownum" and not col.expr.args:
                    expr_sql = "ROW_NUMBER() OVER ()"
                else:
                    expr_sql = compile_value(col.expr)
                name = col.name or _default_name(col.expr)
                items.append(f"{expr_sql} AS {quote_ident(name)}")
            else:  # pragma: no cover
                raise RelAlgExecutionError(f"bad projection item: {col!r}")
        sql = f"SELECT {', '.join(items)} FROM {from_frag}"
        if wheres:
            sql += f" WHERE {_and(wheres)}"
        return sql

    # ---- rename columns --------------------------------------------------
    def _compile_rename_columns(self, node: ast.RenameColumns) -> str:
        from_frag, wheres = self.compile_from(node.child)
        renames = ", ".join(
            f"{quote_ident(a.src.name)} AS {quote_ident(a.dst)}" for a in node.assignments
        )
        sql = f"SELECT * RENAME ({renames}) FROM {from_frag}"
        if wheres:
            sql += f" WHERE {_and(wheres)}"
        return sql

    # ---- group by / aggregate --------------------------------------------
    def _compile_group_by(self, node: ast.GroupBy) -> str:
        from_frag, wheres = self.compile_from(node.child)
        select_items = [_compile_column_ref(c) for c in node.group]
        for agg in node.aggregate:
            select_items.append(f"{self._compile_agg(agg)} AS {quote_ident(agg.name)}")
        sql = f"SELECT {', '.join(select_items)} FROM {from_frag}"
        if wheres:
            sql += f" WHERE {_and(wheres)}"
        if node.group:
            group_sql = ", ".join(_compile_column_ref(c) for c in node.group)
            sql += f" GROUP BY {group_sql}"
        return sql

    @staticmethod
    def _compile_agg(agg: ast.AggCall) -> str:
        if agg.agg_func == "COUNT_ALL":
            return "COUNT(*)"
        col_sql = _compile_column_ref(agg.col)
        return f"{agg.agg_func}({col_sql})"

    # ---- order by -----------------------------------------------------
    def _compile_order_by(self, node: ast.OrderBy) -> str:
        from_frag, wheres = self.compile_from(node.child)
        sql = f"SELECT * FROM {from_frag}"
        if wheres:
            sql += f" WHERE {_and(wheres)}"
        order_sql = ", ".join(
            f"{_compile_column_ref(a.col)} {'ASC' if a.ascending else 'DESC'}" for a in node.args
        )
        sql += f" ORDER BY {order_sql}"
        return sql

    # ---- set operators -----------------------------------------------------
    def _compile_set_op(self, node, sql_op: str) -> str:
        left_sql = self.compile_select(node.left)
        right_sql = self.compile_select(node.right)
        return f"({left_sql}) {sql_op} ({right_sql})"

    # ---- division ------------------------------------------------------
    def _compile_division(self, node: ast.Division) -> str:
        left_sql = self.compile_select(node.left)
        right_sql = self.compile_select(node.right)
        left_cols = self.con.sql(f"SELECT * FROM ({left_sql}) AS _l LIMIT 0").columns
        right_cols = self.con.sql(f"SELECT * FROM ({right_sql}) AS _r LIMIT 0").columns

        s_cols = list(right_cols)
        diff_cols = [c for c in left_cols if c not in s_cols]
        if not diff_cols or len(diff_cols) + len(s_cols) != len(left_cols):
            raise RelAlgExecutionError(
                "division: the divisor's columns must be a proper, name-matching "
                "subset of the dividend's columns")

        diff_list = ", ".join(quote_ident(c) for c in diff_cols)
        diff_eq = _and([f'r.{quote_ident(c)} = d.{quote_ident(c)}' for c in diff_cols])
        s_eq = _and([f'r.{quote_ident(c)} = s.{quote_ident(c)}' for c in s_cols])

        return (
            f"SELECT DISTINCT {diff_list} FROM ({left_sql}) AS d "
            f"WHERE NOT EXISTS ("
            f"  SELECT 1 FROM ({right_sql}) AS s "
            f"  WHERE NOT EXISTS ("
            f"    SELECT 1 FROM ({left_sql}) AS r WHERE {diff_eq} AND {s_eq}"
            f"  )"
            f")"
        )


def _default_name(expr: ast.ValueExpr) -> str:
    if isinstance(expr, ast.Column):
        return expr.name
    if isinstance(expr, ast.Call):
        return expr.func
    return "expr"
