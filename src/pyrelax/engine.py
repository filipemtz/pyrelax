"""
Public entry point: ``RelAlgEngine``.

Registering a pandas DataFrame with DuckDB is one line
(``con.register(name, df)``) — exactly as easy as creating the DataFrame
itself — so that's the whole "setup cost" of this backend.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import duckdb
import pandas as pd

from .errors import RelAlgExecutionError, RelAlgSyntaxError
from .parser import parse_relalg_expression
from .sql_compiler import SqlCompiler, quote_ident

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

        registered: Dict[str, Any] = dict(self.tables)  # name -> DataFrame | _AsView
        last_sql: Optional[str] = None
        for var_name, expr_text in statements:
            sql = self._compile(expr_text, registered)
            last_sql = sql
            if var_name is not None:
                registered[var_name] = _AsView(sql)

        final_sql = f"SELECT * FROM ({last_sql}) AS _result"
        if eliminate_duplicates:
            final_sql = f"SELECT DISTINCT * FROM ({last_sql}) AS _result"
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

    # ---- internals -------------------------------------------------------
    def _compile(self, expr_text: str, registered: Dict[str, Any]) -> str:
        for name, value in registered.items():
            if isinstance(value, pd.DataFrame):
                self._con.register(name, value)
            else:
                # a previously-assigned intermediate relation: expose it as
                # a view so later statements can reference it by name too.
                try:
                    self._con.execute(f"CREATE OR REPLACE TEMP VIEW {quote_ident(name)} AS {value.sql}")
                except duckdb.Error as e:
                    raise RelAlgExecutionError(str(e)) from e

        ast_node = parse_relalg_expression(expr_text)
        compiler = SqlCompiler(self._con)
        try:
            return compiler.compile_select(ast_node)
        except duckdb.Error as e:
            raise RelAlgExecutionError(str(e)) from e


class _AsView:
    """Marks a registered name as "already compiled SQL for a temp view"
    rather than a raw DataFrame."""

    def __init__(self, sql: str):
        self.sql = sql


def execute_query(query: str, tables: Dict[str, pd.DataFrame], eliminate_duplicates: bool = True) -> pd.DataFrame:
    """Convenience one-shot function: parse + execute `query` against `tables`."""
    return RelAlgEngine(tables).query(query, eliminate_duplicates=eliminate_duplicates)
