"""
Public entry point: ``RelAlgEngine``.

Registering a pandas DataFrame with DuckDB is one line
(``con.register(name, df)``) — exactly as easy as creating the DataFrame
itself — so that's the whole "setup cost" of this backend.

Multi-statement scripts (``nome = expr`` assignments followed by a final
expression) are handled by *substituting* each assignment's AST into
later statements wherever its name is referenced (see ``substitute.py``),
rather than materialising it as an opaque intermediate relation. This
matters: relational-algebra conditions can reference a relation alias
from anywhere in the same "join scope" (e.g. ``enrollment.student_id``
inside a join condition), and that only keeps working after
``x = sigma ... (enrollment)`` if referencing ``x`` later is exactly as
if ``sigma ... (enrollment)`` had been pasted in place — i.e. assignment
is a name for an expression, not a rename to a new opaque relation.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import duckdb
import pandas as pd

from . import ast_nodes as ast
from .errors import RelAlgExecutionError, RelAlgSyntaxError
from .parser import parse_relalg_expression
from .sql_compiler import SqlCompiler
from .substitute import substitute

_ASSIGNMENT_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|<-|\u2190|=)\s*(.+)$")


class RelAlgEngine:
    """Holds a set of named pandas DataFrames ("relations") and evaluates
    relational-algebra queries against them, using DuckDB as the execution
    backend.

    >>> engine = RelAlgEngine({"Livro": livro_df, "Autor": autor_df})
    >>> engine.query("\u03c0 titulo, ano (\u03c3 ano > 2000 (Livro))")
    """

    def __init__(self, tables: Optional[Dict[str, pd.DataFrame]] = None):
        self.tables: Dict[str, pd.DataFrame] = dict(tables or {})
        self._con = duckdb.connect(database=":memory:")

    def add_relation(self, name: str, df: pd.DataFrame) -> None:
        self.tables[name] = df

    def query(self, ra_text: str, eliminate_duplicates: bool = True) -> pd.DataFrame:
        """Run a (possibly multi-line) relational-algebra script.

        Each *statement* is either:
          - ``name = expr`` / ``name <- expr`` / ``name := expr``  (assignment), or
          - a bare relational-algebra expression.

        A new statement starts only on a line that looks like an
        assignment (or on the very first line); any other line is treated
        as a continuation of the previous statement, so a single
        expression can be wrapped across multiple lines for readability.
        The result of the last non-assignment statement is returned as a
        pandas DataFrame. Lines starting with ``--`` are comments.
        """
        statements = self._split_statements(ra_text)
        if not statements:
            raise RelAlgExecutionError("empty query: no expression to evaluate")

        for name, df in self.tables.items():
            self._con.register(name, df)

        env: Dict[str, ast.RelExpr] = {}
        last_ast: Optional[ast.RelExpr] = None
        for var_name, expr_text in statements:
            raw_ast = parse_relalg_expression(expr_text)
            last_ast = substitute(raw_ast, env)
            if var_name is not None:
                env[var_name] = last_ast

        compiler = SqlCompiler(self._con)
        try:
            body_sql = compiler.compile_select(last_ast)
        except duckdb.Error as e:
            raise RelAlgExecutionError(str(e)) from e

        select_kw = "SELECT DISTINCT" if eliminate_duplicates else "SELECT"
        final_sql = f"{select_kw} * FROM ({body_sql}) AS _result"
        if isinstance(last_ast, ast.OrderBy):
            # An ORDER BY inside `body_sql` isn't guaranteed to survive
            # being wrapped by the DISTINCT/final SELECT above, so
            # re-apply the same ordering at the outermost level too.
            final_sql += f" ORDER BY {SqlCompiler.order_by_clause(last_ast)}"

        try:
            return self._con.sql(final_sql).df()
        except duckdb.Error as e:
            raise RelAlgExecutionError(str(e)) from e

    @staticmethod
    def _split_statements(ra_text: str):
        lines = [
            ln.strip() for ln in ra_text.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
        ]
        statements = []  # list of (var_name_or_None, text)
        i = 0
        while i < len(lines):
            m = _ASSIGNMENT_LINE_RE.match(lines[i])
            var_name = m.group(1) if m else None
            buf = m.group(2) if m else lines[i]
            i += 1
            # Greedily try to parse what we have; only pull in more lines
            # (as a continuation of the same expression) if it doesn't
            # stand on its own yet and the next line isn't itself the
            # start of a new assignment.
            while True:
                try:
                    parse_relalg_expression(buf)
                    break
                except RelAlgSyntaxError:
                    if i < len(lines) and not _ASSIGNMENT_LINE_RE.match(lines[i]):
                        buf += " " + lines[i]
                        i += 1
                    else:
                        break  # let the real parse error surface later
            statements.append((var_name, buf))
        return statements


def execute_query(query: str, tables: Dict[str, pd.DataFrame], eliminate_duplicates: bool = True) -> pd.DataFrame:
    """Convenience one-shot function: parse + execute `query` against `tables`."""
    return RelAlgEngine(tables).query(query, eliminate_duplicates=eliminate_duplicates)
