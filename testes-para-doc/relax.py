"""
relalg.py — a relational-algebra language, implemented on top of pandas.

This is a Python re-implementation of the query language used by the RelaX
relational algebra calculator (https://dbis-uibk.github.io/relax/), covering
its symbols, its ASCII keyword equivalents and its value-expression language,
executed against pandas DataFrames instead of an in-browser JS engine.

Dependency: `lark` (install with `pip install lark`).

--------------------------------------------------------------------------
SUPPORTED RELATIONAL OPERATORS  (symbol | ascii keyword)
--------------------------------------------------------------------------
    projection        π  a,b,c (R)          | pi a,b,c (R)
    selection          σ  cond (R)           | sigma cond (R)
    rename columns     ρ  new<-old,... (R)    | rho new<-old,... (R)
    rename relation     ρ  newName (R)         | rho newName (R)
    group / aggregate   γ  a,b ; SUM(c)->s (R) | gamma a,b ; SUM(c)->s (R)
    order by            τ  a asc, b desc (R)   | tau a asc, b desc (R)
    union               R ∪ S                  | R union S
    intersect            R ∩ S                  | R intersect S
    difference           R − S   (also '\\')     | R except S
    cross join           R ⨯ S                   | R cross join S
    inner/theta join     R ⨝ cond S / R ⋈ cond S  | R join cond S
    natural join         R ⨝ S / R ⋈ S            | R natural join S
    left outer join      R ⟕ [cond] S             | R left [outer] join S
    right outer join     R ⟖ [cond] S             | R right [outer] join S
    full outer join      R ⟗ [cond] S             | R full [outer] join S
    left semi join       R ⋉ S                    | R left semi join S
    right semi join      R ⋊ S                    | R right semi join S
    anti join            R ▷ [cond] S             | R anti join S
    division             R ÷ S                    | R / S
    relation reference   R
    inline empty rel.     {}   (TABLE_DUM)
    inline unit rel.       {()}  (TABLE_DEE)
    parentheses           ( expr )

--------------------------------------------------------------------------
SUPPORTED VALUE EXPRESSIONS (used inside sigma / pi / join conditions / gamma)
--------------------------------------------------------------------------
    constants: numbers, 'strings', true/false, null
    columns:  name  or  relalias.name
    arithmetic:  +  -  *  /  %  (and unary -)
    comparisons: = != <> >= <= > <     (also  IS NULL / IS NOT NULL  via  = null / != null)
    boolean:  and, or, xor, not / ! / ¬        (also  ∧ ∨ ⊻)
    string concat: ||
    LIKE / ILIKE / REGEXP / RLIKE
    BETWEEN ... AND ...  /  NOT BETWEEN ... AND ...
    CASE WHEN ... THEN ... [WHEN ... THEN ...]* [ELSE ...] END
    functions:
        unary:   upper, lower, ucase, lcase, reverse, length, abs, floor, ceil,
                 round, sqrt, exp, ln, date, year, month, day, dayofmonth,
                 hour, minute, second
        binary:  adddate, subdate, mod, add, sub, mul, div(power? no), power, log,
                 repeat, cast( x as string|number|date|boolean ),
                 substring(str, start) / substring(str from start)
        ternary: substring(str, start, len) / substring(str from start for len)
        n-ary:   coalesce(...), concat(...), replace(a,b,c)
        nullary: rand(), rownum(), now(), current_timestamp(), sysdate(), ...
    aggregate functions (only inside gamma): SUM, COUNT, AVG, MIN, MAX, COUNT(*)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    import pandas as pd

from lark import Lark, UnexpectedInput
    from relalg import RelAlgEngine

    engine = RelAlgEngine({
        "Livro":  pd.DataFrame(...),
        "Autor":  pd.DataFrame(...),
    })

    result_df = engine.query('''
        \u03c0 titulo, ano (\u03c3 ano > 2000 (Livro))
    ''')

Multi-statement scripts are also supported, one statement per line, with
`nome = expressão` (or `nome <- expressão` / `nome := expressão`) assigning
an intermediate result that can be referenced by later lines; the value of
the last non-assignment line is returned.

NOTE ON SCOPE: inline relation literals with typed rows, e.g.
`{ a:number, b:string | 1, 'x' \\n 2, 'y' }`, are not supported (only the
degenerate TABLE_DUM `{}` and TABLE_DEE `{()}` are). Build a pandas
DataFrame and pass it in via the `tables` dict instead.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from lark import Lark, UnexpectedInput


# ==========================================================================
# Errors
# ==========================================================================

class RelAlgError(Exception):
    """Base class for all errors raised by this module."""


class RelAlgSyntaxError(RelAlgError):
    """Raised when the query text cannot be parsed."""


class RelAlgExecutionError(RelAlgError):
    """Raised when a syntactically valid query cannot be executed
    (unknown relation, unknown column, ambiguous column, schema mismatch...)."""


# ==========================================================================
# Tokenizer
# ==========================================================================

@dataclass
class Token:
    type: str   # 'SYMBOL' | 'IDENT' | 'NUMBER' | 'STRING' | 'EOF'
    value: Any
    pos: int

    def __repr__(self):
        return f'{self.type}:{self.value!r}'


_LEXER_GRAMMAR = r"""
start: (MULTI | SYMBOL | NUMBER | STRING | IDENT)*

MULTI.10: "<-" | "->" | ":=" | "!=" | "<>" | ">=" | "<=" | "||"

// Unicode relational-algebra symbols and ASCII punctuation/operators.
SYMBOL: /[πσρτγψ←→∪∩÷⨯⨝⋈⋉⋊▷⟕⟖⟗∧∨⊻≠≥≤()\[\],.;=<>+\-*\/%{}|\\!¬]/
NUMBER: /\d+(\.\d+)?/
STRING: /'[^'\n]*'/
IDENT: /[a-zA-Z_][a-zA-Z0-9_]*/

%import common.WS_INLINE
%ignore WS_INLINE
%ignore /\r?\n/
%ignore /--[^\n]*/
%ignore /\/\*.*?\*\//s
"""

_LEXER = Lark(_LEXER_GRAMMAR, parser="lalr", lexer="basic", maybe_placeholders=False)


def tokenize(text: str) -> List[Token]:
    """Tokenize the query using Lark's battle-tested lexer.

    The parser itself still consumes the small Token abstraction used by the
    evaluator. Keeping that boundary makes the lexical implementation easier
    to maintain without forcing changes throughout the AST/evaluator code.
    """
    try:
        parsed = _LEXER.parse(text)
    except UnexpectedInput as exc:
        pos = getattr(exc, "pos_in_stream", None)
        if pos is None:
            pos = 0
        ch = text[pos] if 0 <= pos < len(text) else ""
        raise RelAlgSyntaxError(
            f"unexpected character {ch!r} at position {pos}"
        ) from exc

    tokens: List[Token] = []
    for tok in parsed.children:
        token_type = tok.type
        if token_type == "MULTI":
            token_type = "SYMBOL"
        if token_type == "NUMBER":
            raw = str(tok)
            value = float(raw) if "." in raw else int(raw)
        elif token_type == "STRING":
            value = str(tok)[1:-1]
        else:
            value = str(tok)

        # Lark exposes the exact character offset of each token.
        tokens.append(Token(token_type, value, tok.start_pos))

    tokens.append(Token("EOF", None, len(text)))
    return tokens


# ==========================================================================
# AST node helpers
# (nodes are plain dicts with a 'type'/'func' key; simple + language-native)
# ==========================================================================

def _const(value, datatype='null'):
    return {'kind': 'const', 'datatype': datatype, 'value': value}


def _col(name, rel_alias=None):
    return {'kind': 'column', 'name': name, 'relAlias': rel_alias}


def _call(func, args):
    return {'kind': 'call', 'func': func, 'args': args}


# ==========================================================================
# Parser
# ==========================================================================

_RESERVED = {
    'pi', 'sigma', 'rho', 'tau', 'gamma', 'psi', 'and', 'or', 'not', 'xor',
    'union', 'intersect', 'except', 'join', 'cross', 'inner', 'left',
    'right', 'outer', 'full', 'natural', 'semi', 'anti', 'desc', 'asc',
    'case', 'when', 'then', 'else', 'end', 'true', 'false', 'null', 'is',
    'like', 'ilike', 'regexp', 'rlike', 'between', 'from', 'for', 'as',
}

_COMPARISON_KEYWORDS_AS_SYMBOLS = {
    '≠': '!=', '≥': '>=', '≤': '<=',
}


class Parser:
    def __init__(self, tokens: List[Token], known_relations: Optional[set] = None):
        self.tokens = tokens
        self.pos = 0
        self.known_relations = known_relations or set()

    # ---- low level helpers -------------------------------------------------
    def peek(self, offset=0) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != 'EOF':
            self.pos += 1
        return tok

    def at_eof(self) -> bool:
        return self.peek().type == 'EOF'

    def save(self):
        return self.pos

    def restore(self, mark):
        self.pos = mark

    def is_symbol(self, sym) -> bool:
        t = self.peek()
        return t.type == 'SYMBOL' and t.value == sym

    def try_symbol(self, sym) -> bool:
        if self.is_symbol(sym):
            self.advance()
            return True
        return False

    def expect_symbol(self, sym):
        if not self.try_symbol(sym):
            got = self.peek()
            raise RelAlgSyntaxError(
                f"expected '{sym}' but found {got.type} {got.value!r} at position {got.pos}")

    def is_kw(self, word) -> bool:
        t = self.peek()
        return t.type == 'IDENT' and t.value.lower() == word.lower()

    def try_kw(self, word) -> bool:
        if self.is_kw(word):
            self.advance()
            return True
        return False

    def try_kw_seq(self, *words) -> bool:
        """Try to consume an exact sequence of keywords; all-or-nothing."""
        mark = self.save()
        for w in words:
            if not self.try_kw(w):
                self.restore(mark)
                return False
        return True

    def expect_ident(self) -> str:
        t = self.peek()
        if t.type != 'IDENT':
            raise RelAlgSyntaxError(f'expected identifier at position {t.pos}, found {t.type} {t.value!r}')
        if t.value.lower() in _RESERVED:
            raise RelAlgSyntaxError(f"'{t.value}' is a reserved keyword, cannot be used as a name (position {t.pos})")
        self.advance()
        return t.value

    # ---- symbol-or-keyword operator matchers -------------------------------
    # each returns True (and consumes) or False (and consumes nothing)

    def match_pi(self):
        return self.try_symbol('π') or self.try_kw('pi')

    def match_sigma(self):
        return self.try_symbol('σ') or self.try_kw('sigma')

    def match_rho(self):
        return self.try_symbol('ρ') or self.try_kw('rho')

    def match_tau(self):
        return self.try_symbol('τ') or self.try_kw('tau')

    def match_gamma(self):
        return self.try_symbol('γ') or self.try_kw('gamma')

    def match_arrow_left(self):
        return self.try_symbol('←') or self.try_symbol('<-')

    def match_arrow_right(self):
        return self.try_symbol('→') or self.try_symbol('->')

    def match_union(self):
        return self.try_symbol('∪') or self.try_kw('union')

    def match_intersect(self):
        return self.try_symbol('∩') or self.try_kw('intersect')

    def match_difference(self):
        return self.try_symbol('-') or self.try_symbol('\\') or self.try_kw('except')

    def match_division(self):
        return self.try_symbol('÷') or self.try_symbol('/')

    def match_cross_join(self):
        if self.try_symbol('⨯'):
            return True
        if self.try_kw_seq('cross', 'join'):
            return True
        # RelaX-compatible ASCII shorthand. The token is interpreted as an
        # operator only while parsing the infix-operator position, so a
        # relation actually named "x" remains valid as either operand.
        return self.try_kw('x')

    def match_inner_join_symbol(self):
        return self.try_symbol('⨝') or self.try_symbol('⋈')

    def match_inner_join_kw(self):
        return self.try_kw_seq('inner', 'join') or self.try_kw('join')

    def match_natural_prefix_kw(self):
        return self.try_kw_seq('natural', 'join')

    def match_left_outer_join(self):
        if self.try_symbol('⟕'):
            return True
        return (self.try_kw_seq('left', 'outer', 'join') or self.try_kw_seq('left', 'join'))

    def match_right_outer_join(self):
        if self.try_symbol('⟖'):
            return True
        return (self.try_kw_seq('right', 'outer', 'join') or self.try_kw_seq('right', 'join'))

    def match_full_outer_join(self):
        if self.try_symbol('⟗'):
            return True
        return (self.try_kw_seq('full', 'outer', 'join') or self.try_kw_seq('full', 'join'))

    def match_left_semi_join(self):
        return self.try_symbol('⋉') or self.try_kw_seq('left', 'semi', 'join')

    def match_right_semi_join(self):
        return self.try_symbol('⋊') or self.try_kw_seq('right', 'semi', 'join')

    def match_anti_join(self):
        if self.try_symbol('▷'):
            return True
        return (self.try_kw_seq('anti', 'semi', 'join') or self.try_kw_seq('anti', 'join'))

    # ---- program (multi-statement) ----------------------------------------
    def parse_expression_only(self):
        node = self.parse_expr_p4()
        if not self.at_eof():
            t = self.peek()
            raise RelAlgSyntaxError(f'unexpected trailing input {t.type} {t.value!r} at position {t.pos}')
        return node

    # ---- expression_precedence4: union / difference ------------------------
    def parse_expr_p4(self):
        node = self.parse_expr_p3()
        while True:
            mark = self.save()
            if self.match_union():
                right = self.parse_expr_p3()
                node = {'kind': 'union', 'left': node, 'right': right}
                continue
            self.restore(mark)
            if self.match_difference():
                right = self.parse_expr_p3()
                node = {'kind': 'difference', 'left': node, 'right': right}
                continue
            self.restore(mark)
            break
        return node

    # ---- expression_precedence3: intersect ---------------------------------
    def parse_expr_p3(self):
        node = self.parse_expr_p2()
        while True:
            mark = self.save()
            if self.match_intersect():
                right = self.parse_expr_p2()
                node = {'kind': 'intersect', 'left': node, 'right': right}
                continue
            self.restore(mark)
            break
        return node

    # ---- expression_precedence2: joins / division --------------------------
    def parse_expr_p2(self):
        node = self.parse_expr_p1()
        while True:
            mark = self.save()

            if self.match_cross_join():
                right = self.parse_expr_p1()
                node = {'kind': 'crossJoin', 'left': node, 'right': right}
                continue
            self.restore(mark)

            if self.match_inner_join_symbol() or self.match_inner_join_kw():
                cond = self._maybe_parse_join_condition()
                right = self.parse_expr_p1()
                if cond is not None:
                    node = {'kind': 'thetaJoin', 'left': node, 'right': right, 'cond': cond}
                else:
                    node = {'kind': 'naturalJoin', 'left': node, 'right': right}
                continue
            self.restore(mark)

            if self.match_natural_prefix_kw():
                right = self.parse_expr_p1()
                node = {'kind': 'naturalJoin', 'left': node, 'right': right}
                continue
            self.restore(mark)

            if self.match_left_outer_join():
                cond = self._maybe_parse_join_condition()
                right = self.parse_expr_p1()
                node = {'kind': 'leftOuterJoin', 'left': node, 'right': right, 'cond': cond}
                continue
            self.restore(mark)

            if self.match_right_outer_join():
                cond = self._maybe_parse_join_condition()
                right = self.parse_expr_p1()
                node = {'kind': 'rightOuterJoin', 'left': node, 'right': right, 'cond': cond}
                continue
            self.restore(mark)

            if self.match_full_outer_join():
                cond = self._maybe_parse_join_condition()
                right = self.parse_expr_p1()
                node = {'kind': 'fullOuterJoin', 'left': node, 'right': right, 'cond': cond}
                continue
            self.restore(mark)

            if self.match_left_semi_join():
                right = self.parse_expr_p1()
                node = {'kind': 'leftSemiJoin', 'left': node, 'right': right}
                continue
            self.restore(mark)

            if self.match_right_semi_join():
                right = self.parse_expr_p1()
                node = {'kind': 'rightSemiJoin', 'left': node, 'right': right}
                continue
            self.restore(mark)

            if self.match_anti_join():
                cond = self._maybe_parse_join_condition()
                right = self.parse_expr_p1()
                node = {'kind': 'antiJoin', 'left': node, 'right': right, 'cond': cond}
                continue
            self.restore(mark)

            if self.match_division():
                right = self.parse_expr_p1()
                node = {'kind': 'division', 'left': node, 'right': right}
                continue
            self.restore(mark)

            break
        return node

    def _maybe_parse_join_condition(self):
        """Try to parse an optional value-expression join condition.
        Disambiguates from a natural join followed directly by a bare
        relation name (mirrors the original grammar's lookahead trick)."""
        mark = self.save()
        try:
            cond = self.parse_value_expr()
        except RelAlgSyntaxError:
            self.restore(mark)
            return None
        if (cond.get('kind') == 'column' and cond.get('relAlias') is None
                and cond.get('name') in self.known_relations):
            # this was actually the start of the right-hand relation, not a condition
            self.restore(mark)
            return None
        return cond

    # ---- expression_precedence1: unary prefix operators --------------------
    def parse_expr_p1(self):
        if self.match_tau():
            args = self.parse_list_of_order_by_args()
            child = self.parse_expr_p1()
            return {'kind': 'orderBy', 'child': child, 'args': args}

        if self.match_gamma():
            group, aggs = self.parse_group_by_args()
            child = self.parse_expr_p1()
            return {'kind': 'groupBy', 'child': child, 'group': group, 'aggregate': aggs}

        if self.match_rho():
            mark = self.save()
            # renameRelation: rho NAME ( child )   -- NAME directly followed by '('
            if self.peek().type == 'IDENT' and self.peek().value.lower() not in _RESERVED:
                name = self.peek().value
                nxt = self.peek(1)
                if nxt.type == 'SYMBOL' and nxt.value == '(':
                    self.advance()  # consume name
                    child = self.parse_expr_p1()
                    return {'kind': 'renameRelation', 'child': child, 'newRelAlias': name}
            self.restore(mark)
            assigns = self.parse_list_of_col_assignments()
            child = self.parse_expr_p1()
            return {'kind': 'renameColumns', 'child': child, 'assignments': assigns}

        if self.match_sigma():
            cond = self.parse_value_expr()
            child = self.parse_expr_p1()
            return {'kind': 'selection', 'child': child, 'cond': cond}

        if self.match_pi():
            cols = self.parse_list_of_named_column_exprs()
            child = self.parse_expr_p1()
            return {'kind': 'projection', 'child': child, 'columns': cols}

        return self.parse_expr_p0()

    # ---- expression_precedence0: table / relation / ( expr ) ---------------
    def parse_expr_p0(self):
        if self.is_symbol('{'):
            return self.parse_inline_table()
        if self.try_symbol('('):
            node = self.parse_expr_p4()
            self.expect_symbol(')')
            return node
        name = self.expect_ident()
        return {'kind': 'relation', 'name': name}

    def parse_inline_table(self):
        self.expect_symbol('{')
        if self.try_symbol('}'):
            return {'kind': 'tableDum'}
        if self.try_symbol('('):
            self.expect_symbol(')')
            self.expect_symbol('}')
            return {'kind': 'tableDee'}
        raise RelAlgSyntaxError(
            'inline relation literals with rows (e.g. "{ a:number | 1 }") are not '
            'supported; build a pandas DataFrame and pass it in the `tables` dict instead')

    # ---- pi argument: list of named column expressions ---------------------
    def parse_list_of_named_column_exprs(self):
        items = [self.parse_named_column_expr()]
        while self.try_symbol(','):
            items.append(self.parse_named_column_expr())
        return items

    def parse_named_column_expr(self):
        # asterisk:  *  or  alias.*
        mark = self.save()
        if self.peek().type == 'IDENT' and self.peek(1).type == 'SYMBOL' and self.peek(1).value == '.' \
                and self.peek(2).type == 'SYMBOL' and self.peek(2).value == '*':
            alias = self.advance().value
            self.advance()  # '.'
            self.advance()  # '*'
            return {'kind': 'star', 'relAlias': alias}
        self.restore(mark)
        if self.try_symbol('*'):
            return {'kind': 'star', 'relAlias': None}

        # dst <- valueExpr
        mark = self.save()
        if self.peek().type == 'IDENT' and self.peek().value.lower() not in _RESERVED:
            nxt = self.peek(1)
            if nxt.type == 'SYMBOL' and nxt.value in ('<-', '←'):
                dst = self.advance().value
                self.advance()
                expr = self.parse_value_expr()
                return {'kind': 'namedExpr', 'name': dst, 'child': expr}
        self.restore(mark)

        expr = self.parse_value_expr()
        if self.match_arrow_right():
            dst = self.expect_unqualified_name()
            return {'kind': 'namedExpr', 'name': dst, 'child': expr}
        if expr.get('kind') == 'column':
            return expr
        # anonymous derived column: name it from its textual form
        return {'kind': 'namedExpr', 'name': None, 'child': expr}

    def expect_unqualified_name(self):
        return self.expect_ident()

    # ---- rho argument: list of column assignments --------------------------
    def parse_list_of_col_assignments(self):
        items = [self.parse_col_assignment()]
        while self.try_symbol(','):
            items.append(self.parse_col_assignment())
        return items

    def parse_col_assignment(self):
        mark = self.save()
        if self.peek().type == 'IDENT':
            nxt = self.peek(1)
            if nxt.type == 'SYMBOL' and nxt.value in ('<-', '←'):
                dst = self.advance().value
                self.advance()
                src = self.parse_column_name()
                return {'dst': dst, 'src': src}
        self.restore(mark)
        src = self.parse_column_name()
        self.match_arrow_right() or self._expect_arrow_right()
        dst = self.expect_unqualified_name()
        return {'dst': dst, 'src': src}

    def _expect_arrow_right(self):
        if not self.match_arrow_right():
            raise RelAlgSyntaxError(f'expected "->" at position {self.peek().pos}')
        return True

    # ---- tau argument: order-by list ---------------------------------------
    def parse_list_of_order_by_args(self):
        items = [self.parse_order_by_arg()]
        while self.try_symbol(','):
            items.append(self.parse_order_by_arg())
        return items

    def parse_order_by_arg(self):
        col = self.parse_column_name()
        asc = True
        if self.try_kw('asc'):
            asc = True
        elif self.try_kw('desc'):
            asc = False
        return {'col': col, 'asc': asc}

    # ---- gamma argument: group columns + aggregate list ---------------------
    def parse_group_by_args(self):
        group = []
        mark = self.save()
        # try: listOfColumns ';' listOfAggFunctionArguments
        try:
            first_col = self.parse_column_name()
            cols = [first_col]
            while self.try_symbol(','):
                cols.append(self.parse_column_name())
            if self.try_symbol(';'):
                group = cols
                aggs = self.parse_list_of_agg_function_arguments()
                return group, aggs
        except RelAlgSyntaxError:
            pass
        self.restore(mark)
        self.try_symbol(';')
        aggs = self.parse_list_of_agg_function_arguments()
        return [], aggs

    _AGG_FUNCS = {'sum', 'count', 'avg', 'min', 'max'}

    def parse_list_of_agg_function_arguments(self):
        items = [self.parse_agg_function_argument()]
        while self.try_symbol(','):
            items.append(self.parse_agg_function_argument())
        return items

    def parse_agg_function_argument(self):
        mark = self.save()
        if self.peek().type == 'IDENT':
            nxt = self.peek(1)
            if nxt.type == 'SYMBOL' and nxt.value in ('<-', '←'):
                name = self.advance().value
                self.advance()
                func = self.parse_agg_function()
                func['name'] = name
                return func
        self.restore(mark)
        func = self.parse_agg_function()
        self._expect_arrow_right()
        name = self.expect_unqualified_name()
        func['name'] = name
        return func

    def parse_agg_function(self):
        t = self.peek()
        if t.type != 'IDENT' or t.value.lower() not in self._AGG_FUNCS:
            raise RelAlgSyntaxError(f'expected aggregate function (SUM/COUNT/AVG/MIN/MAX) at position {t.pos}')
        fname = self.advance().value.upper()
        self.expect_symbol('(')
        if fname == 'COUNT' and self.try_symbol('*'):
            self.expect_symbol(')')
            return {'aggFunc': 'COUNT_ALL', 'col': None}
        col = self.parse_column_name()
        self.expect_symbol(')')
        return {'aggFunc': fname, 'col': col}

    # ---- column name / relalias.name ---------------------------------------
    def parse_column_name(self):
        mark = self.save()
        if self.peek().type == 'IDENT' and self.peek(1).type == 'SYMBOL' and self.peek(1).value == '.':
            alias = self.advance().value
            self.advance()
            name = self.expect_unqualified_name()
            return _col(name, alias)
        self.restore(mark)
        name = self.expect_unqualified_name()
        return _col(name, None)

    # =========================================================================
    # value expressions (precedence 9 .. 0, loosest to tightest)
    # =========================================================================
    def parse_value_expr(self):
        return self.parse_v9()

    def parse_v9(self):  # or, ||
        node = self.parse_v8()
        while True:
            if self.try_kw('or') or self.try_symbol('∨'):
                right = self.parse_v8()
                node = _call('or', [node, right])
                continue
            if self.try_symbol('||'):
                right = self.parse_v8()
                node = _call('concat', [node, right])
                continue
            break
        return node

    def parse_v8(self):  # xor
        node = self.parse_v7()
        while self.try_kw('xor') or self.try_symbol('⊻'):
            right = self.parse_v7()
            node = _call('xor', [node, right])
        return node

    def parse_v7(self):  # and
        node = self.parse_v6()
        while self.try_kw('and') or self.try_symbol('∧'):
            right = self.parse_v6()
            node = _call('and', [node, right])
        return node

    def parse_v6(self):  # case when
        if self.is_kw('case'):
            return self.parse_case_when()
        return self.parse_v5()

    def parse_case_when(self):
        self.advance()  # 'case'
        whens = []
        while self.try_kw('when'):
            cond = self.parse_value_expr()
            if not self.try_kw('then'):
                raise RelAlgSyntaxError(f'expected THEN at position {self.peek().pos}')
            res = self.parse_value_expr()
            whens.append((cond, res))
        if not whens:
            raise RelAlgSyntaxError(f'expected WHEN at position {self.peek().pos}')
        else_expr = None
        if self.try_kw('else'):
            else_expr = self.parse_v5()
        if not self.try_kw('end'):
            raise RelAlgSyntaxError(f'expected END at position {self.peek().pos}')
        args = []
        for cond, res in whens:
            args.append(cond)
            args.append(res)
        if else_expr is not None:
            args.append(_const(True, 'boolean'))
            args.append(else_expr)
            return _call('caseWhenElse', args)
        return _call('caseWhen', args)

    def parse_v5(self):  # comparisons, is [not] null, like, between
        node = self.parse_v4()
        while True:
            mark = self.save()
            if self.try_kw('between') or self.try_kw_seq('not', 'between'):
                negated = self.tokens[mark].type == 'IDENT' and self.tokens[mark].value.lower() == 'not'
                lower = self.parse_v4()
                if not self.try_kw('and'):
                    raise RelAlgSyntaxError(f'expected AND in BETWEEN at position {self.peek().pos}')
                upper = self.parse_v4()
                node = _call('notBetween' if negated else 'between', [node, lower, upper])
                continue
            self.restore(mark)

            op = self._try_comparison_op()
            if op is not None:
                right = self.parse_v4()
                node = _call(op, [node, right])
                continue

            if self.try_kw('like') or self.try_kw('ilike') or self.try_kw('regexp') or self.try_kw('rlike'):
                fname = self.tokens[self.pos - 1].value.lower()
                right = self.parse_v0_constant_only()
                node = _call(fname, [node, right])
                continue

            break
        return node

    def _try_comparison_op(self):
        t = self.peek()
        if t.type == 'SYMBOL':
            if t.value in ('=', '!=', '<>', '>=', '<=', '>', '<'):
                self.advance()
                return {'<>': '!='}.get(t.value, t.value)
            if t.value in _COMPARISON_KEYWORDS_AS_SYMBOLS:
                self.advance()
                return _COMPARISON_KEYWORDS_AS_SYMBOLS[t.value]
        return None

    def parse_v0_constant_only(self):
        return self.parse_v0()

    def parse_v4(self):  # + -
        node = self.parse_v3()
        while True:
            t = self.peek()
            if t.type == 'SYMBOL' and t.value in ('+', '-'):
                self.advance()
                right = self.parse_v3()
                node = _call('add' if t.value == '+' else 'sub', [node, right])
                continue
            break
        return node

    def parse_v3(self):  # * / %
        node = self.parse_v2()
        while True:
            t = self.peek()
            if t.type == 'SYMBOL' and t.value in ('*', '/', '%'):
                self.advance()
                right = self.parse_v2()
                fname = {'*': 'mul', '/': 'div', '%': 'mod'}[t.value]
                node = _call(fname, [node, right])
                continue
            break
        return node

    def parse_v2(self):  # unary minus
        if self.is_symbol('-'):
            self.advance()
            operand = self.parse_v2()
            return _call('minus', [operand])
        return self.parse_v1()

    def parse_v1(self):  # not / ! / ¬
        if self.try_symbol('!') or self.try_symbol('¬') or self.try_kw('not'):
            operand = self.parse_v5()
            return _call('not', [operand])
        return self.parse_v0()

    def parse_v0(self):
        t = self.peek()

        if t.type == 'NUMBER':
            self.advance()
            return _const(t.value, 'number')
        if t.type == 'STRING':
            self.advance()
            return _const(t.value, 'string')
        if self.is_kw('true'):
            self.advance()
            return _const(True, 'boolean')
        if self.is_kw('false'):
            self.advance()
            return _const(False, 'boolean')
        if self.is_kw('null'):
            self.advance()
            return _const(None, 'null')

        if self.is_symbol('('):
            self.advance()
            node = self.parse_v9()
            self.expect_symbol(')')
            return node

        if t.type == 'IDENT':
            fname = t.value.lower()
            if fname in _VALUE_FUNCS_NULLARY and self._peek_is_call(1):
                self.advance()
                self.expect_symbol('(')
                self.expect_symbol(')')
                return _call(fname, [])
            if fname in _VALUE_FUNCS_UNARY and self._peek_is_call(1):
                self.advance()
                self.expect_symbol('(')
                arg0 = self.parse_v9()
                self.expect_symbol(')')
                return _call(fname, [arg0])
            if fname == 'substring' and self._peek_is_call(1):
                return self.parse_substring()
            if fname in _VALUE_FUNCS_BINARY and self._peek_is_call(1):
                self.advance()
                self.expect_symbol('(')
                arg0 = self.parse_v9()
                self.expect_symbol(',')
                arg1 = self.parse_v9()
                self.expect_symbol(')')
                return _call(fname, [arg0, arg1])
            if fname == 'cast' and self._peek_is_call(1):
                self.advance()
                self.expect_symbol('(')
                arg0 = self.parse_v9()
                if not self.try_kw('as'):
                    raise RelAlgSyntaxError(f'expected AS in CAST at position {self.peek().pos}')
                typ = self.expect_ident_no_reserved_check().lower()
                self.expect_symbol(')')
                return _call('cast', [arg0, _const(typ, 'string')])
            if fname in _VALUE_FUNCS_NARY and self._peek_is_call(1):
                self.advance()
                self.expect_symbol('(')
                args = [self.parse_v9()]
                while self.try_symbol(','):
                    args.append(self.parse_v9())
                self.expect_symbol(')')
                return _call(fname, args)

            # plain column reference: name  or  alias.name
            return self.parse_column_name()

        raise RelAlgSyntaxError(f'unexpected token {t.type} {t.value!r} at position {t.pos}')

    def expect_ident_no_reserved_check(self):
        t = self.advance()
        return t.value

    def parse_substring(self):
        self.advance()  # 'substring'
        self.expect_symbol('(')
        arg0 = self.parse_v9()
        if self.try_symbol(','):
            arg1 = self.parse_v9()
            if self.try_symbol(','):
                arg2 = self.parse_v9()
                self.expect_symbol(')')
                return _call('substring', [arg0, arg1, arg2])
            self.expect_symbol(')')
            return _call('substring', [arg0, arg1])
        if self.try_kw('from'):
            arg1 = self.parse_v9()
            if self.try_kw('for'):
                arg2 = self.parse_v9()
                self.expect_symbol(')')
                return _call('substring', [arg0, arg1, arg2])
            self.expect_symbol(')')
            return _call('substring', [arg0, arg1])
        raise RelAlgSyntaxError(f'malformed SUBSTRING() at position {self.peek().pos}')

    def _peek_is_call(self, offset):
        t = self.peek(offset)
        return t.type == 'SYMBOL' and t.value == '('


_VALUE_FUNCS_UNARY = {
    'upper', 'ucase', 'lower', 'lcase', 'reverse', 'length', 'abs', 'floor',
    'ceil', 'round', 'sqrt', 'exp', 'ln', 'date', 'year', 'month', 'day',
    'dayofmonth', 'hour', 'minute', 'second',
}
_VALUE_FUNCS_BINARY = {
    'adddate', 'subdate', 'mod', 'add', 'sub', 'mul', 'div', 'power', 'log', 'repeat',
}
_VALUE_FUNCS_NARY = {'coalesce', 'concat', 'replace'}
_VALUE_FUNCS_NULLARY = {
    'rand', 'rownum', 'now', 'current_timestamp', 'transaction_timestamp',
    'statement_timestamp', 'clock_timestamp', 'sysdate',
}


def parse_relalg_expression(text: str, known_relations: Optional[set] = None):
    tokens = tokenize(text)
    parser = Parser(tokens, known_relations)
    return parser.parse_expression_only()


# ==========================================================================
# Schema / Relation model
#
# Every physical pandas column is given a globally-unique internal id
# ("c0", "c1", ...); a separate lightweight schema tracks, for each id,
# the (relation-alias, display-name) pair. This mirrors the separation
# of Schema/Table in the original RelaX implementation and makes column
# resolution (incl. ambiguity detection) independent of how pandas
# happens to label columns.
# ==========================================================================

_colid_counter = itertools.count()


def _new_colid() -> str:
    return f'c{next(_colid_counter)}'


@dataclass
class ColumnInfo:
    colid: str
    rel_alias: Optional[str]
    name: Any  # usually str


@dataclass
class Relation:
    df: pd.DataFrame
    columns: List[ColumnInfo]

    def colids(self) -> List[str]:
        return [c.colid for c in self.columns]

    def find(self, name, rel_alias=None) -> List[ColumnInfo]:
        matches = []
        for c in self.columns:
            if c.name != name:
                continue
            if rel_alias is not None and c.rel_alias != rel_alias:
                continue
            matches.append(c)
        return matches

    def resolve(self, name, rel_alias=None) -> ColumnInfo:
        matches = self.find(name, rel_alias)
        if not matches:
            label = f'{rel_alias}.{name}' if rel_alias else str(name)
            raise RelAlgExecutionError(f'unknown column "{label}"')
        if len(matches) > 1:
            label = f'{rel_alias}.{name}' if rel_alias else str(name)
            raise RelAlgExecutionError(
                f'column "{label}" is ambiguous ({len(matches)} matches) — qualify it with a relation alias')
        return matches[0]

    def display_frame(self) -> pd.DataFrame:
        """Return a copy of the underlying dataframe with human friendly
        column labels (alias.name if alias is set and needed for uniqueness,
        else just name)."""
        names = [c.name for c in self.columns]
        dupes = {n for n in names if names.count(n) > 1}
        labels = []
        for c in self.columns:
            if c.name in dupes and c.rel_alias:
                labels.append(f'{c.rel_alias}.{c.name}')
            else:
                labels.append(str(c.name))
        out = self.df[self.colids()].copy()
        out.columns = labels
        return out

    def clone_with_alias(self, alias: Optional[str]) -> 'Relation':
        new_cols = [ColumnInfo(c.colid, alias, c.name) for c in self.columns]
        return Relation(self.df, new_cols)


def relation_from_dataframe(name: str, df: pd.DataFrame) -> Relation:
    colids = [_new_colid() for _ in df.columns]
    new_df = df.copy()
    new_df.columns = colids
    new_df = new_df.reset_index(drop=True)
    cols = [ColumnInfo(cid, name, col) for cid, col in zip(colids, df.columns)]
    return Relation(new_df, cols)


# ==========================================================================
# Value expression evaluation (vectorised over a Relation's dataframe)
# ==========================================================================

def _to_bool_series(x, index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype('boolean')
    return pd.Series([bool(x)] * len(index) if x is not None else [pd.NA] * len(index),
                      index=index, dtype='boolean')


def _series_like(x, index):
    if isinstance(x, pd.Series):
        return x
    return pd.Series([x] * len(index), index=index)


def _like_to_regex(pattern: str) -> str:
    out = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == '%':
            out.append('.*')
        elif ch == '_':
            out.append('.')
        else:
            out.append(re.escape(ch))
        i += 1
    return '^' + ''.join(out) + '$'


def _apply_elementwise(fn, *series_or_scalars, index):
    """Apply a python scalar function element-wise, propagating NaN/None."""
    series_args = [_series_like(a, index) for a in series_or_scalars]
    n = len(index)

    def combine(i):
        vals = [s.iloc[i] for s in series_args]
        if any((v is None) or (isinstance(v, float) and math.isnan(v)) for v in vals):
            return None
        return fn(*vals)

    return pd.Series([combine(i) for i in range(n)], index=index)


class ValueEvaluator:
    def __init__(self, relation: Relation):
        self.relation = relation
        self.index = relation.df.index

    def eval(self, node):
        kind = node.get('kind')
        if kind == 'const':
            return node['value']
        if kind == 'column':
            col = self.relation.resolve(node['name'], node['relAlias'])
            return self.relation.df[col.colid]
        if kind == 'call':
            return self.eval_call(node['func'], node['args'])
        raise RelAlgExecutionError(f'cannot evaluate expression node: {node}')

    def eval_call(self, func, arg_nodes):
        if func == 'rownum':
            if arg_nodes:
                raise RelAlgExecutionError('rownum() does not accept arguments')
            return pd.Series(np.arange(1, len(self.index) + 1), index=self.index)

        handler = _FUNC_TABLE.get(func)
        if handler is None:
            raise RelAlgExecutionError(f'unsupported function/operator "{func}"')
        args = [self.eval(a) for a in arg_nodes]
        return handler(self, args)

    def eval_bool_mask(self, node) -> pd.Series:
        result = self.eval(node)
        mask = _to_bool_series(result, self.index)
        return mask.fillna(False).astype(bool)


# ---- function table -------------------------------------------------------

def _f_and(ev, args):
    a, b = (_to_bool_series(x, ev.index) for x in args)
    return a & b


def _f_or(ev, args):
    a, b = (_to_bool_series(x, ev.index) for x in args)
    return a | b


def _f_xor(ev, args):
    a, b = (_to_bool_series(x, ev.index) for x in args)
    return a ^ b


def _f_not(ev, args):
    (a,) = args
    return ~_to_bool_series(a, ev.index)


def _numeric_binop(op):
    def handler(ev, args):
        a, b = args
        sa, sb = _series_like(a, ev.index), _series_like(b, ev.index)
        return op(sa, sb)
    return handler


def _f_minus(ev, args):
    (a,) = args
    return -_series_like(a, ev.index)


def _comparison(op, na_result=pd.NA):
    def handler(ev, args):
        a, b = args
        if op == '=' and b is None:
            return _series_like(a, ev.index).isna()
        if op == '!=' and b is None:
            return _series_like(a, ev.index).notna()
        sa, sb = _series_like(a, ev.index), _series_like(b, ev.index)
        ops = {
            '=': lambda x, y: x == y, '!=': lambda x, y: x != y,
            '>': lambda x, y: x > y, '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y, '<=': lambda x, y: x <= y,
        }
        result = ops[op](sa, sb)
        return result.astype('boolean')
    return handler


def _f_like(case_insensitive=False, use_regex=False):
    def handler(ev, args):
        a, pattern = args
        s = _series_like(a, ev.index).astype('string')
        if use_regex:
            regex = pattern
        else:
            regex = _like_to_regex(pattern)
        flags = re.IGNORECASE if case_insensitive else 0
        return s.str.match(regex, flags=flags).astype('boolean')
    return handler


def _f_between(negate=False):
    def handler(ev, args):
        a, lo, hi = args
        sa = _series_like(a, ev.index)
        slo, shi = _series_like(lo, ev.index), _series_like(hi, ev.index)
        result = (sa >= slo) & (sa <= shi)
        if negate:
            result = ~result
        return result.astype('boolean')
    return handler


def _f_case_when(ev, args):
    n = len(args) // 2
    default = None
    result = pd.Series([default] * len(ev.index), index=ev.index, dtype='object')
    filled = pd.Series([False] * len(ev.index), index=ev.index)
    for i in range(n):
        cond = _to_bool_series(args[2 * i], ev.index).fillna(False).astype(bool)
        val = _series_like(args[2 * i + 1], ev.index)
        take = cond & (~filled)
        result = result.where(~take, val)
        filled = filled | cond
    return result


def _f_coalesce(ev, args):
    result = _series_like(args[0], ev.index)
    for a in args[1:]:
        s = _series_like(a, ev.index)
        result = result.where(result.notna(), s)
    return result


def _f_concat(ev, args):
    parts = [_series_like(a, ev.index).astype('string') for a in args]
    out = parts[0]
    for p in parts[1:]:
        out = out.str.cat(p)
    return out


def _f_replace(ev, args):
    s, old, new = args
    return _apply_elementwise(lambda v, o, n: str(v).replace(str(o), str(n)), s, old, new, index=ev.index)


def _f_substring(ev, args):
    if len(args) == 2:
        s, start = args
        return _apply_elementwise(lambda v, st: str(v)[max(int(st) - 1, 0):], s, start, index=ev.index)
    s, start, length = args
    return _apply_elementwise(
        lambda v, st, ln: str(v)[max(int(st) - 1, 0): max(int(st) - 1, 0) + int(ln)], s, start, length, index=ev.index)


def _f_cast(ev, args):
    value, typ_node = args
    typ = typ_node
    s = _series_like(value, ev.index)
    if typ == 'number':
        return pd.to_numeric(s, errors='coerce')
    if typ == 'string':
        return s.astype('string')
    if typ == 'boolean':
        return s.astype('boolean')
    if typ == 'date':
        return pd.to_datetime(s, errors='coerce')
    raise RelAlgExecutionError(f'unknown cast target type "{typ}"')


def _unary_numeric(fn):
    def handler(ev, args):
        (a,) = args
        return _series_like(a, ev.index).astype(float).apply(fn)
    return handler


def _unary_string(fn):
    def handler(ev, args):
        (a,) = args
        return _series_like(a, ev.index).astype('string').apply(lambda v: fn(v) if pd.notna(v) else v)
    return handler


def _date_part(part):
    def handler(ev, args):
        (a,) = args
        s = pd.to_datetime(_series_like(a, ev.index), errors='coerce')
        return getattr(s.dt, part)
    return handler


def _f_date(ev, args):
    (a,) = args
    return pd.to_datetime(_series_like(a, ev.index), errors='coerce')


def _f_adddate(ev, args):
    a, days = args
    base = pd.to_datetime(_series_like(a, ev.index), errors='coerce')
    d = _series_like(days, ev.index).astype(int)
    return base + pd.to_timedelta(d, unit='D')


def _f_subdate(ev, args):
    a, days = args
    base = pd.to_datetime(_series_like(a, ev.index), errors='coerce')
    d = _series_like(days, ev.index).astype(int)
    return base - pd.to_timedelta(d, unit='D')


def _nullary(fn):
    def handler(ev, args):
        return pd.Series([fn()] * len(ev.index), index=ev.index)
    return handler


import random as _random
import datetime as _datetime

_FUNC_TABLE = {
    'and': _f_and, 'or': _f_or, 'xor': _f_xor, 'not': _f_not,
    'add': _numeric_binop(lambda a, b: a + b),
    'sub': _numeric_binop(lambda a, b: a - b),
    'mul': _numeric_binop(lambda a, b: a * b),
    'div': _numeric_binop(lambda a, b: a / b),
    'mod': _numeric_binop(lambda a, b: a % b),
    'power': _numeric_binop(lambda a, b: a ** b),
    'log': _numeric_binop(lambda a, b: np.log(b) / np.log(a)),
    'minus': _f_minus,
    '=': _comparison('='), '!=': _comparison('!='),
    '>': _comparison('>'), '<': _comparison('<'),
    '>=': _comparison('>='), '<=': _comparison('<='),
    'like': _f_like(), 'ilike': _f_like(case_insensitive=True),
    'regexp': _f_like(use_regex=True), 'rlike': _f_like(use_regex=True),
    'between': _f_between(), 'notBetween': _f_between(negate=True),
    'caseWhen': _f_case_when, 'caseWhenElse': _f_case_when,
    'coalesce': _f_coalesce, 'concat': _f_concat, 'replace': _f_replace,
    'repeat': lambda ev, args: _apply_elementwise(lambda v, n: str(v) * int(n), args[0], args[1], index=ev.index),
    'substring': _f_substring,
    'cast': _f_cast,
    'upper': _unary_string(str.upper), 'ucase': _unary_string(str.upper),
    'lower': _unary_string(str.lower), 'lcase': _unary_string(str.lower),
    'reverse': _unary_string(lambda s: s[::-1]),
    'length': lambda ev, args: _series_like(args[0], ev.index).astype('string').str.len(),
    'abs': _unary_numeric(abs), 'floor': _unary_numeric(math.floor),
    'ceil': _unary_numeric(math.ceil), 'round': _unary_numeric(round),
    'sqrt': _unary_numeric(math.sqrt), 'exp': _unary_numeric(math.exp),
    'ln': _unary_numeric(math.log),
    'date': _f_date, 'adddate': _f_adddate, 'subdate': _f_subdate,
    'year': _date_part('year'), 'month': _date_part('month'),
    'day': _date_part('day'), 'dayofmonth': _date_part('day'),
    'hour': _date_part('hour'), 'minute': _date_part('minute'), 'second': _date_part('second'),
    'rand': _nullary(_random.random),
    'now': _nullary(_datetime.datetime.now), 'current_timestamp': _nullary(_datetime.datetime.now),
    'transaction_timestamp': _nullary(_datetime.datetime.now),
    'statement_timestamp': _nullary(_datetime.datetime.now),
    'clock_timestamp': _nullary(_datetime.datetime.now),
    'sysdate': _nullary(_datetime.datetime.now),
    # rownum() is handled directly by ValueEvaluator because it depends on row position.
}


# ==========================================================================
# Relational operator evaluation
# ==========================================================================

class Evaluator:
    def __init__(self, base_relations: Dict[str, pd.DataFrame]):
        self._base_frames = base_relations

    def evaluate(self, node) -> Relation:
        kind = node['kind']
        method = getattr(self, f'_eval_{kind}', None)
        if method is None:
            raise RelAlgExecutionError(f'unsupported relational operator "{kind}"')
        return method(node)

    # ---- leaves -------------------------------------------------------
    def _eval_relation(self, node) -> Relation:
        name = node['name']
        if name not in self._base_frames:
            raise RelAlgExecutionError(f'unknown relation "{name}"')
        df = self._base_frames[name]
        if isinstance(df, Relation):
            return Relation(df.df.copy(), list(df.columns))
        return relation_from_dataframe(name, df)

    def _eval_tableDum(self, node) -> Relation:
        return Relation(pd.DataFrame(index=pd.RangeIndex(0)), [])

    def _eval_tableDee(self, node) -> Relation:
        return Relation(pd.DataFrame(index=pd.RangeIndex(1)), [])

    # ---- projection -----------------------------------------------------
    def _eval_projection(self, node) -> Relation:
        child = self.evaluate(node['child'])
        ev = ValueEvaluator(child)
        new_cols = []
        new_data = {}
        for item in node['columns']:
            if item['kind'] == 'star':
                alias = item['relAlias']
                for c in child.columns:
                    if alias is not None and c.rel_alias != alias:
                        continue
                    cid = _new_colid()
                    new_data[cid] = child.df[c.colid]
                    new_cols.append(ColumnInfo(cid, c.rel_alias, c.name))
            elif item['kind'] == 'column':
                c = child.resolve(item['name'], item['relAlias'])
                cid = _new_colid()
                new_data[cid] = child.df[c.colid]
                new_cols.append(ColumnInfo(cid, c.rel_alias, c.name))
            elif item['kind'] == 'namedExpr':
                value = ev.eval(item['child'])
                cid = _new_colid()
                series = _series_like(value, child.df.index)
                new_data[cid] = series
                name = item['name'] or _expr_default_name(item['child'])
                new_cols.append(ColumnInfo(cid, None, name))
            else:
                raise RelAlgExecutionError(f'bad projection item: {item}')
        out_df = pd.DataFrame(new_data, index=child.df.index)
        rel = Relation(out_df, new_cols)
        return _dedup_rows(rel)

    # ---- selection --------------------------------------------------------
    def _eval_selection(self, node) -> Relation:
        child = self.evaluate(node['child'])
        ev = ValueEvaluator(child)
        mask = ev.eval_bool_mask(node['cond'])
        out_df = child.df[mask].reset_index(drop=True)
        return Relation(out_df, list(child.columns))

    # ---- rename -------------------------------------------------------
    def _eval_renameRelation(self, node) -> Relation:
        child = self.evaluate(node['child'])
        return child.clone_with_alias(node['newRelAlias'])

    def _eval_renameColumns(self, node) -> Relation:
        child = self.evaluate(node['child'])
        cols = list(child.columns)
        for a in node['assignments']:
            src = child.resolve(a['src']['name'], a['src']['relAlias'])
            for i, c in enumerate(cols):
                if c.colid == src.colid:
                    cols[i] = ColumnInfo(c.colid, c.rel_alias, a['dst'])
        return Relation(child.df, cols)

    # ---- group by / aggregate ----------------------------------------------
    def _eval_groupBy(self, node) -> Relation:
        child = self.evaluate(node['child'])
        group_cols = [child.resolve(g['name'], g['relAlias']) for g in node['group']]
        df = child.df

        new_cols = []
        if not group_cols:
            agg_data = {}
            for agg in node['aggregate']:
                cid = _new_colid()
                agg_data[cid] = [_compute_agg(df, child, agg)]
                new_cols.append(ColumnInfo(cid, None, agg['name']))
            out_df = pd.DataFrame(agg_data, index=pd.RangeIndex(1)) if agg_data else \
                pd.DataFrame(index=pd.RangeIndex(1))
            return Relation(out_df, new_cols)

        gkey_ids = [c.colid for c in group_cols]
        grouped = df.groupby(gkey_ids, dropna=False, sort=False)

        rows = []
        keys_seen = []
        for key, sub in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            row = list(key)
            for agg in node['aggregate']:
                row.append(_compute_agg(sub, child, agg))
            rows.append(row)

        for c in group_cols:
            new_cols.append(ColumnInfo(_new_colid(), c.rel_alias, c.name))
        for agg in node['aggregate']:
            new_cols.append(ColumnInfo(_new_colid(), None, agg['name']))

        out_df = pd.DataFrame(rows, columns=[c.colid for c in new_cols]) if rows else \
            pd.DataFrame(columns=[c.colid for c in new_cols])
        return Relation(out_df.reset_index(drop=True), new_cols)

    # ---- order by -------------------------------------------------------
    def _eval_orderBy(self, node) -> Relation:
        child = self.evaluate(node['child'])
        by, ascending = [], []
        for arg in node['args']:
            c = child.resolve(arg['col']['name'], arg['col']['relAlias'])
            by.append(c.colid)
            ascending.append(arg['asc'])
        out_df = child.df.sort_values(by=by, ascending=ascending, kind='mergesort').reset_index(drop=True)
        return Relation(out_df, list(child.columns))

    # ---- set operators (union / intersect / difference) ---------------------
    def _eval_union(self, node) -> Relation:
        return self._set_op(node, 'union')

    def _eval_intersect(self, node) -> Relation:
        return self._set_op(node, 'intersect')

    def _eval_difference(self, node) -> Relation:
        return self._set_op(node, 'difference')

    def _set_op(self, node, op) -> Relation:
        left = self.evaluate(node['left'])
        right = self.evaluate(node['right'])
        if len(left.columns) != len(right.columns):
            raise RelAlgExecutionError(
                f'{op}: relations must have the same degree '
                f'({len(left.columns)} vs {len(right.columns)})')

        left_names = [c.name for c in left.columns]
        right_names = [c.name for c in right.columns]
        if left_names != right_names:
            raise RelAlgExecutionError(
                f'{op}: relations must have compatible attribute names '
                f'({left_names!r} vs {right_names!r})')

        left_ids = left.colids()
        right_ids = right.colids()
        ldf = left.df[left_ids].copy()
        rdf = right.df[right_ids].copy()
        rdf.columns = left_ids

        if op == 'union':
            out = pd.concat([ldf, rdf], ignore_index=True).drop_duplicates().reset_index(drop=True)
        elif op == 'intersect':
            merged = ldf.merge(rdf.drop_duplicates(), how='inner')
            out = merged.drop_duplicates().reset_index(drop=True)
        else:  # difference
            merged = ldf.merge(rdf.drop_duplicates(), how='left', indicator=True)
            out = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge']).reset_index(drop=True)
        return Relation(out, list(left.columns))

    # ---- cross / theta / natural joins -----------------------------------
    def _eval_crossJoin(self, node) -> Relation:
        left = self.evaluate(node['left'])
        right = self.evaluate(node['right'])
        return _cross(left, right)

    def _eval_thetaJoin(self, node) -> Relation:
        left = self.evaluate(node['left'])
        right = self.evaluate(node['right'])
        cross = _cross(left, right)
        mask = ValueEvaluator(cross).eval_bool_mask(node['cond'])
        out_df = cross.df[mask].reset_index(drop=True)
        return Relation(out_df, list(cross.columns))

    def _eval_naturalJoin(self, node) -> Relation:
        left = self.evaluate(node['left'])
        right = self.evaluate(node['right'])
        return _natural_join(left, right, how='inner')

    # ---- outer joins --------------------------------------------------------
    def _eval_leftOuterJoin(self, node) -> Relation:
        return self._outer_join(node, 'left')

    def _eval_rightOuterJoin(self, node) -> Relation:
        return self._outer_join(node, 'right')

    def _eval_fullOuterJoin(self, node) -> Relation:
        return self._outer_join(node, 'full')

    def _outer_join(self, node, side) -> Relation:
        left = self.evaluate(node['left'])
        right = self.evaluate(node['right'])
        cond = node.get('cond')
        if cond is None:
            return _natural_join(left, right, how=side)
        return _theta_outer_join(left, right, cond, side)

    # ---- semi / anti joins --------------------------------------------------
    def _eval_leftSemiJoin(self, node) -> Relation:
        left = self.evaluate(node['left'])
        right = self.evaluate(node['right'])
        keep_mask = _natural_match_mask(left, right)
        out_df = left.df[keep_mask].reset_index(drop=True)
        return Relation(out_df, list(left.columns))

    def _eval_rightSemiJoin(self, node) -> Relation:
        left = self.evaluate(node['left'])
        right = self.evaluate(node['right'])
        keep_mask = _natural_match_mask(right, left)
        out_df = right.df[keep_mask].reset_index(drop=True)
        return Relation(out_df, list(right.columns))

    def _eval_antiJoin(self, node) -> Relation:
        left = self.evaluate(node['left'])
        right = self.evaluate(node['right'])
        cond = node.get('cond')
        if cond is None:
            keep_mask = ~_natural_match_mask(left, right)
        else:
            cross = _cross(left, right)
            cross_df = cross.df.copy()
            n_left, n_right = len(left.df), len(right.df)
            cross_df['__lpos__'] = np.repeat(np.arange(n_left), n_right) if n_right else pd.Series([], dtype=int)
            cross_with_pos = Relation(cross_df, list(cross.columns))
            match_mask = ValueEvaluator(cross_with_pos).eval_bool_mask(cond)
            matched_left_positions = set(cross_df.loc[match_mask, '__lpos__'])
            keep_mask = ~left.df.index.to_series().reset_index(drop=True).isin(matched_left_positions)
            keep_mask = keep_mask.values
        out_df = left.df[keep_mask].reset_index(drop=True)
        return Relation(out_df, list(left.columns))

    # ---- division -------------------------------------------------------
    def _eval_division(self, node) -> Relation:
        left = self.evaluate(node['left'])
        right = self.evaluate(node['right'])

        right_names = [c.name for c in right.columns]
        left_names = [c.name for c in left.columns]
        if len(set(right_names)) != len(right_names):
            raise RelAlgExecutionError('division: divisor attributes must have unique names')
        if any(name not in left_names for name in right_names):
            raise RelAlgExecutionError(
                'division: every divisor attribute must occur in the dividend')

        diff_cols = [c for c in left.columns if c.name not in set(right_names)]
        if not diff_cols:
            raise RelAlgExecutionError(
                'division: the dividend must contain at least one attribute not in the divisor')

        x_names = [c.name for c in diff_cols]
        y_names = right_names
        x_ids = [c.colid for c in diff_cols]
        y_ids = [next(c.colid for c in left.columns if c.name == name) for name in y_names]
        right_ids = [c.colid for c in right.columns]

        # For each X tuple in the dividend, require that every Y tuple from the
        # divisor occurs together with X in the dividend.  This is the direct
        # definition of relational division and avoids depending on column order.
        candidates = left.df[x_ids].drop_duplicates().reset_index(drop=True)
        candidates.columns = [f'__x{i}' for i in range(len(x_ids))]

        divisor = right.df[right_ids].drop_duplicates().reset_index(drop=True)
        divisor.columns = [f'__y{i}' for i in range(len(y_ids))]

        if divisor.empty:
            # In relational algebra, R / empty-set is the projection on X.
            out = candidates.copy()
        else:
            combinations = candidates.merge(divisor, how='cross')

            dividend = left.df[x_ids + y_ids].copy()
            dividend.columns = list(candidates.columns) + list(divisor.columns)
            dividend = dividend.drop_duplicates()

            matched = combinations.merge(
                dividend,
                on=list(candidates.columns) + list(divisor.columns),
                how='inner',
            )
            count_by_x = matched.groupby(list(candidates.columns), dropna=False).size()
            needed = len(divisor)
            qualifying = count_by_x[count_by_x == needed].reset_index()[list(candidates.columns)]
            out = qualifying

        new_cols = [ColumnInfo(_new_colid(), c.rel_alias, c.name) for c in diff_cols]
        out.columns = [c.colid for c in new_cols]
        return Relation(out.reset_index(drop=True), new_cols)


def _expr_default_name(node):
    if node.get('kind') == 'column':
        return node['name']
    if node.get('kind') == 'call':
        return node['func']
    return 'expr'


def _dedup_rows(rel: Relation) -> Relation:
    if not rel.columns:
        return rel
    out = rel.df.drop_duplicates(subset=rel.colids()).reset_index(drop=True)
    return Relation(out, list(rel.columns))


def _compute_agg(df: pd.DataFrame, relation: Relation, agg: dict):
    func = agg['aggFunc']
    if func == 'COUNT_ALL':
        return len(df)
    c = relation.resolve(agg['col']['name'], agg['col']['relAlias'])
    series = df[c.colid]
    if func == 'SUM':
        return series.sum()
    if func == 'COUNT':
        return series.count()
    if func == 'AVG':
        return series.mean()
    if func == 'MIN':
        return series.min()
    if func == 'MAX':
        return series.max()
    raise RelAlgExecutionError(f'unknown aggregate function {func}')


def _cross(left: Relation, right: Relation) -> Relation:
    ldf = left.df.copy()
    rdf = right.df.copy()
    ldf['__k__'] = 1
    rdf['__k__'] = 1
    merged = ldf.merge(rdf, on='__k__', how='outer', suffixes=('', '')).drop(columns='__k__')
    merged = merged.reset_index(drop=True)
    return Relation(merged, list(left.columns) + list(right.columns))


def _natural_join(left: Relation, right: Relation, how: str) -> Relation:
    shared = []
    for lc in left.columns:
        for rc in right.columns:
            if lc.name == rc.name:
                shared.append((lc, rc))
    if not shared:
        # With no common attributes, natural join is a Cartesian product.
        # For an outer join, however, an empty opposite input must still
        # preserve the non-empty side.
        if how == 'inner' or (len(left.df) and len(right.df)):
            return _cross(left, right)

        if how == 'left' and len(left.df):
            filler = pd.DataFrame({c.colid: [None] * len(left.df) for c in right.columns})
            out = pd.concat([left.df.reset_index(drop=True), filler], axis=1)
            return Relation(out, list(left.columns) + list(right.columns))

        if how == 'right' and len(right.df):
            filler = pd.DataFrame({c.colid: [None] * len(right.df) for c in left.columns})
            out = pd.concat([filler, right.df.reset_index(drop=True)], axis=1)
            return Relation(out, list(left.columns) + list(right.columns))

        if how == 'full':
            if len(left.df):
                filler = pd.DataFrame({c.colid: [None] * len(left.df) for c in right.columns})
                out = pd.concat([left.df.reset_index(drop=True), filler], axis=1)
                return Relation(out, list(left.columns) + list(right.columns))
            if len(right.df):
                filler = pd.DataFrame({c.colid: [None] * len(right.df) for c in left.columns})
                out = pd.concat([filler, right.df.reset_index(drop=True)], axis=1)
                return Relation(out, list(left.columns) + list(right.columns))

        return Relation(
            pd.DataFrame(columns=[c.colid for c in left.columns + right.columns]),
            list(left.columns) + list(right.columns),
        )

    l_ids = [lc.colid for lc, _ in shared]
    r_ids = [rc.colid for _, rc in shared]
    ldf = left.df.copy()
    rdf = right.df.copy()
    rdf_renamed = rdf.copy()
    rename_map = dict(zip(r_ids, l_ids))
    rdf_renamed_join_keys = rdf_renamed.rename(columns=rename_map)

    pandas_how = {'inner': 'inner', 'left': 'left', 'right': 'right', 'full': 'outer'}[how]
    other_r_ids = [c.colid for c in right.columns if c.colid not in r_ids]

    right_for_merge = rdf_renamed_join_keys[l_ids + other_r_ids]
    merged = ldf.merge(right_for_merge, on=l_ids, how=pandas_how)
    merged = merged.reset_index(drop=True)

    new_cols = list(left.columns) + [c for c in right.columns if c.colid not in r_ids]
    return Relation(merged, new_cols)


def _natural_match_mask(a: Relation, b: Relation) -> np.ndarray:
    """boolean mask over a.df: True where the row of `a` has >=1 natural-join
    partner in `b` (equality on all columns whose *name* is shared)."""
    shared = [(ac, bc) for ac in a.columns for bc in b.columns if ac.name == bc.name]
    if not shared:
        return np.ones(len(a.df), dtype=bool) if len(b.df) > 0 else np.zeros(len(a.df), dtype=bool)
    a_ids = [ac.colid for ac, _ in shared]
    b_ids = [bc.colid for _, bc in shared]
    bkey = b.df[b_ids].copy()
    bkey.columns = a_ids
    bkey = bkey.drop_duplicates()
    akey = a.df[a_ids].copy()
    akey['__pos__'] = range(len(akey))
    merged = akey.merge(bkey, on=a_ids, how='left', indicator=True)
    matched_positions = set(merged.loc[merged['_merge'] == 'both', '__pos__'])
    return np.array([i in matched_positions for i in range(len(a.df))])


def _theta_outer_join(left: Relation, right: Relation, cond, side: str) -> Relation:
    cross = _cross(left, right)
    cross_df = cross.df.copy()
    cross_df['__lpos__'] = np.repeat(np.arange(len(left.df)), len(right.df)) if len(right.df) else \
        pd.Series([], dtype=int)
    cross_df['__rpos__'] = np.tile(np.arange(len(right.df)), len(left.df)) if len(left.df) else \
        pd.Series([], dtype=int)
    cross_with_pos = Relation(cross_df, list(cross.columns))
    mask = ValueEvaluator(cross_with_pos).eval_bool_mask(cond)
    matched = cross_df[mask]

    matched_out = matched.drop(columns=['__lpos__', '__rpos__'], errors='ignore')
    matched_l = set(matched['__lpos__']) if len(matched) else set()
    matched_r = set(matched['__rpos__']) if len(matched) else set()

    pieces = [matched_out]
    right_col_ids = [c.colid for c in right.columns]
    left_col_ids = [c.colid for c in left.columns]

    if side in ('left', 'full'):
        unmatched_left_positions = [i for i in range(len(left.df)) if i not in matched_l]
        if unmatched_left_positions:
            left_rows = left.df.iloc[unmatched_left_positions].reset_index(drop=True)
            filler = pd.DataFrame({cid: [None] * len(left_rows) for cid in right_col_ids})
            piece = pd.concat([left_rows.reset_index(drop=True), filler], axis=1)
            pieces.append(piece)

    if side in ('right', 'full'):
        unmatched_right_positions = [i for i in range(len(right.df)) if i not in matched_r]
        if unmatched_right_positions:
            right_rows = right.df.iloc[unmatched_right_positions].reset_index(drop=True)
            filler = pd.DataFrame({cid: [None] * len(right_rows) for cid in left_col_ids})
            piece = pd.concat([filler, right_rows.reset_index(drop=True)], axis=1)
            pieces.append(piece)

    out_df = pd.concat(pieces, ignore_index=True) if pieces else cross_df.drop(
        columns=['__lpos__', '__rpos__'], errors='ignore')
    out_df = out_df[[c.colid for c in cross.columns]]
    return Relation(out_df.reset_index(drop=True), list(cross.columns))


# ==========================================================================
# Top level API
# ==========================================================================

def _split_statements(text: str) -> List[str]:
    """Split a script at top-level newlines/semicolons.

    Newlines inside relational expressions are treated as whitespace, so a
    query such as ``pi ... (\n    sigma ...\n)`` remains one statement. String literals and
    comments are ignored while tracking parenthesis depth.
    """
    statements: List[str] = []
    current: List[str] = []
    depth = 0
    i = 0
    n = len(text)
    in_string = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                if depth == 0:
                    stmt = "".join(current).strip()
                    if stmt:
                        statements.append(stmt)
                    current = []
                else:
                    current.append("\n")
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if not in_string and ch == "-" and nxt == "-":
            in_line_comment = True
            i += 2
            continue

        if not in_string and ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if ch == "'":
            in_string = not in_string
            current.append(ch)
            i += 1
            continue

        if not in_string:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    raise RelAlgSyntaxError("unmatched ')' in query")

            if depth == 0 and ch in "\n;":
                stmt = "".join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    if in_string:
        raise RelAlgSyntaxError("unterminated string literal")
    if in_block_comment:
        raise RelAlgSyntaxError("unterminated block comment")
    if depth != 0:
        raise RelAlgSyntaxError("unbalanced parentheses in query")

    stmt = "".join(current).strip()
    if stmt:
        statements.append(stmt)
    return statements


_ASSIGNMENT_LINE_RE = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|<-|←|=)\s*(.+)$',
    re.DOTALL,
)


class RelAlgEngine:
    """Holds a set of named pandas DataFrames ("relations") and evaluates
    relational-algebra queries against them."""

    def __init__(self, tables: Optional[Dict[str, pd.DataFrame]] = None):
        self.tables: Dict[str, pd.DataFrame] = dict(tables or {})

    def add_relation(self, name: str, df: pd.DataFrame):
        self.tables[name] = df

    def query(self, ra_text: str, eliminate_duplicates: bool = True) -> pd.DataFrame:
        """Run a (possibly multi-line) relational-algebra script.

        Each non-blank line is either:
          - `name = expr` / `name <- expr` / `name := expr`   (assignment), or
          - a bare relational-algebra expression.

        The result of the last non-assignment line is returned as a
        pandas DataFrame. Lines beginning with `--` are comments.
        """
        env: Dict[str, pd.DataFrame] = dict(self.tables)
        last_result: Optional[Relation] = None

        for line in _split_statements(ra_text):
            m = _ASSIGNMENT_LINE_RE.match(line)
            if m:
                var_name, expr_text = m.group(1), m.group(2)
                relation = self._run_single(expr_text, env)
                env[var_name] = relation
                last_result = relation
            else:
                last_result = self._run_single(line, env)

        if last_result is None:
            raise RelAlgExecutionError('empty query: no expression to evaluate')

        if eliminate_duplicates:
            last_result = _dedup_relation(last_result)

        return last_result.display_frame()

    def _run_single(self, expr_text: str, env: Dict[str, Any]) -> Relation:
        known = {k for k in env.keys()}
        ast = parse_relalg_expression(expr_text, known_relations=known)
        evaluator = Evaluator(env)
        return evaluator.evaluate(ast)


def _dedup_relation(rel: Relation) -> Relation:
    if not rel.columns:
        return rel
    out = rel.df.drop_duplicates(subset=rel.colids()).reset_index(drop=True)
    return Relation(out, list(rel.columns))


def execute_query(query: str, tables: Dict[str, pd.DataFrame], eliminate_duplicates: bool = True) -> pd.DataFrame:
    """Convenience one-shot function: parse + execute `query` against `tables`."""
    return RelAlgEngine(tables).query(query, eliminate_duplicates=eliminate_duplicates)
