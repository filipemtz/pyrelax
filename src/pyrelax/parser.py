"""
Turns RelaX-syntax query text into an AST (see ``ast_nodes.py``).

Parsing is done in two steps, the standard lark workflow:
  1. ``grammar.lark`` + the Earley algorithm builds a generic parse tree.
  2. ``_ToAst``, a ``lark.Transformer``, walks that tree bottom-up and
     turns each rule into the corresponding ``@dataclass`` node.

Earley (rather than LALR) is used because the language has a couple of
genuine local ambiguities inherited from the original RelaX grammar —
most notably "JOIN <name>", which is only disambiguated by whether a
*condition and then another operand* follows (natural join) or just the
final operand (theta join whose condition happens to be a bare column
reference). Earley explores every reading and keeps only the one that
yields a complete, valid parse.
"""

from __future__ import annotations

from pathlib import Path

from lark import Lark, Transformer, v_args
from lark.exceptions import LarkError

from . import ast_nodes as ast
from .errors import RelAlgSyntaxError

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
_PARSER = Lark(
    _GRAMMAR_PATH.read_text(encoding="utf-8"),
    start="start",
    parser="earley",
    ambiguity="resolve",
)


@v_args(inline=True)
class _ToAst(Transformer):
    # ---- leaves -----------------------------------------------------
    def relation(self, name):
        return ast.Relation(str(name))

    def table_dum(self):
        return ast.TableDum()

    def table_dee(self):
        return ast.TableDee()

    def unqualified_column(self, name):
        return ast.Column(str(name), None)

    def qualified_column(self, alias, name):
        return ast.Column(str(name), str(alias))

    def column_ref(self, col):
        return col

    # ---- set operators -------------------------------------------------
    def union(self, left, _tok, right):
        return ast.Union(left, right)

    def intersect(self, left, _tok, right):
        return ast.Intersect(left, right)

    def difference(self, left, _tok, right):
        return ast.Difference(left, right)

    # ---- joins / division -----------------------------------------------
    def cross_join(self, left, _tok, right):
        return ast.CrossJoin(left, right)

    def theta_join(self, left, _tok, cond, right):
        return ast.ThetaJoin(left, right, cond)

    def natural_join(self, left, _tok, right):
        return ast.NaturalJoin(left, right)

    def left_outer_join(self, left, _tok, *rest):
        cond, right = (rest[0], rest[1]) if len(rest) == 2 else (None, rest[0])
        return ast.LeftOuterJoin(left, right, cond)

    def right_outer_join(self, left, _tok, *rest):
        cond, right = (rest[0], rest[1]) if len(rest) == 2 else (None, rest[0])
        return ast.RightOuterJoin(left, right, cond)

    def full_outer_join(self, left, _tok, *rest):
        cond, right = (rest[0], rest[1]) if len(rest) == 2 else (None, rest[0])
        return ast.FullOuterJoin(left, right, cond)

    def left_semi_join(self, left, _tok, right):
        return ast.LeftSemiJoin(left, right)

    def right_semi_join(self, left, _tok, right):
        return ast.RightSemiJoin(left, right)

    def anti_join(self, left, _tok, *rest):
        cond, right = (rest[0], rest[1]) if len(rest) == 2 else (None, rest[0])
        return ast.AntiJoin(left, right, cond)

    def division(self, left, _tok, right):
        return ast.Division(left, right)

    # ---- unary prefix operators -------------------------------------------
    def order_by(self, _tok, args, child):
        return ast.OrderBy(child, args)

    def group_by(self, _tok, group_and_aggs, child):
        group, aggs = group_and_aggs
        return ast.GroupBy(child, group, aggs)

    def rename_relation(self, _tok, name, child):
        return ast.RenameRelation(child, str(name))

    def rename_columns(self, _tok, assignments, child):
        return ast.RenameColumns(child, assignments)

    def selection(self, _tok, cond, child):
        return ast.Selection(child, cond)

    def projection(self, _tok, columns, child):
        return ast.Projection(child, columns)

    # ---- pi argument -----------------------------------------------------
    def named_column_expr_list(self, *items):
        return list(items)

    def star(self):
        return ast.Star(None)

    def star_qualified(self, alias):
        return ast.Star(str(alias))

    def named_expr_left(self, name, _tok, expr):
        return ast.NamedExpr(expr, str(name))

    def named_expr_right(self, expr, _tok, name):
        return ast.NamedExpr(expr, str(name))

    # ---- rho argument ------------------------------------------------------
    def col_assignment_list(self, *items):
        return list(items)

    def col_assignment_left(self, dst, _tok, src):
        return ast.ColAssignment(str(dst), src)

    def col_assignment_right(self, src, _tok, dst):
        return ast.ColAssignment(str(dst), src)

    # ---- tau argument -----------------------------------------------------
    def order_by_args(self, *items):
        return list(items)

    def order_by_asc(self, col, *_asc):
        return ast.OrderArg(col, True)

    def order_by_desc(self, col, _tok):
        return ast.OrderArg(col, False)

    # ---- gamma argument -----------------------------------------------------
    def group_by_full(self, *items):
        *cols, aggs = items
        return (list(cols), aggs)

    def group_by_aggs_only(self, aggs):
        return ([], aggs)

    def agg_call_list(self, *items):
        return list(items)

    def agg_call_left(self, name, _tok, func):
        return ast.AggCall(func.agg_func, func.col, str(name))

    def agg_call_right(self, func, _tok, name):
        return ast.AggCall(func.agg_func, func.col, str(name))

    def agg_func_col(self, fname, col):
        return ast.AggCall(str(fname).upper(), col, name=None)

    def agg_func_count_all(self, _tok):
        return ast.AggCall("COUNT_ALL", None, name=None)

    # ---- value expressions --------------------------------------------------
    def or_(self, a, _tok, b):
        return ast.Call("or", [a, b])

    def concat(self, a, _tok, b):
        return ast.Call("concat", [a, b])

    def xor_(self, a, _tok, b):
        return ast.Call("xor", [a, b])

    def and_(self, a, _tok, b):
        return ast.Call("and", [a, b])

    def case_when(self, _case, *rest):
        whens = [r for r in rest if isinstance(r, tuple) and len(r) == 2 and r[0] == "when"]
        elses = [r for r in rest if isinstance(r, tuple) and len(r) == 2 and r[0] == "else"]
        args = []
        for _, (cond, res) in whens:
            args += [cond, res]
        if elses:
            args += [ast.Const(True, "boolean"), elses[0][1]]
            return ast.Call("caseWhenElse", args)
        return ast.Call("caseWhen", args)

    def when_clause(self, _when, cond, _then, res):
        return ("when", (cond, res))

    def else_clause(self, _else, res):
        return ("else", res)

    def not_between(self, a, _not, _between, lo, _and, hi):
        return ast.Call("notBetween", [a, lo, hi])

    def between(self, a, _between, lo, _and, hi):
        return ast.Call("between", [a, lo, hi])

    def comparison(self, a, op, b):
        norm = {"<>": "!=", "\u2260": "!=", "\u2265": ">=", "\u2264": "<="}
        return ast.Call(norm.get(str(op), str(op)), [a, b])

    def like_op(self, a, _tok, b):
        return ast.Call("like", [a, b])

    def ilike_op(self, a, _tok, b):
        return ast.Call("ilike", [a, b])

    def regexp_op(self, a, _tok, b):
        return ast.Call("regexp", [a, b])

    def add(self, a, b):
        return ast.Call("add", [a, b])

    def sub(self, a, b):
        return ast.Call("sub", [a, b])

    def mul(self, a, b):
        return ast.Call("mul", [a, b])

    def div(self, a, b):
        return ast.Call("div", [a, b])

    def mod(self, a, b):
        return ast.Call("mod", [a, b])

    def negate(self, a):
        return ast.Call("minus", [a])

    def not_(self, _tok, a):
        return ast.Call("not", [a])

    def number_const(self, tok):
        raw = str(tok)
        val = float(raw) if "." in raw else int(raw)
        return ast.Const(val, "number")

    def string_const(self, tok):
        return ast.Const(str(tok)[1:-1], "string")

    def true_const(self, _tok):
        return ast.Const(True, "boolean")

    def false_const(self, _tok):
        return ast.Const(False, "boolean")

    def null_const(self, _tok):
        return ast.Const(None, "null")

    # ---- function calls ---------------------------------------------------
    def call_nullary(self, fname, *_parens):
        return ast.Call(str(fname).lower(), [])

    def call_unary(self, fname, arg):
        return ast.Call(str(fname).lower(), [arg])

    def call_binary(self, fname, a, b):
        return ast.Call(str(fname).lower(), [a, b])

    def call_nary(self, fname, *args):
        return ast.Call(str(fname).lower(), list(args))

    def call_substring2(self, *args):
        v9_args = [a for a in args if not (hasattr(a, "type"))]
        return ast.Call("substring", list(v9_args))

    def call_substring3(self, *args):
        v9_args = [a for a in args if not (hasattr(a, "type"))]
        return ast.Call("substring", list(v9_args))

    def call_cast(self, value, _as, typename):
        return ast.Call("cast", [value, ast.Const(str(typename).lower(), "string")])


def parse_relalg_expression(text: str) -> ast.RelExpr:
    """Parse a single relational-algebra expression and return its AST."""
    try:
        tree = _PARSER.parse(text)
    except LarkError as e:
        raise RelAlgSyntaxError(str(e)) from e
    return _ToAst().transform(tree)
