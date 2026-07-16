"""The Nomos law DSL: tokenizer, parser, AST.

A law file declares channels (per-cell float fields) and rules. Each step,
every rule's `when` condition is evaluated against a snapshot of the world at
the start of the step; where it holds, the rule's actions are applied.
See README.md for the full language reference — that document is the spec.

AST is plain tuples:
    ("num", 1.5)
    ("name", "energy")
    ("un", "neg"|"not", expr)
    ("bin", "+"|"-"|"*"|"/"|"%"|"=="|"!="|"<"|"<="|">"|">="|"and"|"or", l, r)
    ("call", "sum8", [expr, ...])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class NomosError(Exception):
    """Any error in parsing or executing a law file."""


KEYWORDS = {"channel", "init", "rule", "when", "set", "add", "and", "or", "not"}

_TOKEN_RE = re.compile(
    r"""
      (?P<ws>[ \t\r\n]+)
    | (?P<comment>\#[^\n]*)
    | (?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+)
    | (?P<name>[A-Za-z_]\w*)
    | (?P<op>==|!=|<=|>=|[-+*/%<>=(),:])
    """,
    re.VERBOSE,
)


@dataclass
class Token:
    kind: str  # "num" | "name" | "op" | "eof"
    value: str
    pos: int


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            line = text.count("\n", 0, pos) + 1
            raise NomosError(f"line {line}: unexpected character {text[pos]!r}")
        kind = m.lastgroup
        if kind not in ("ws", "comment"):
            tokens.append(Token(kind, m.group(), pos))
        pos = m.end()
    tokens.append(Token("eof", "", pos))
    return tokens


@dataclass
class Channel:
    name: str
    init: tuple | None  # expression AST, None means zeros


@dataclass
class Rule:
    name: str
    when: tuple
    actions: list[tuple]  # ("set"|"add", channel_name, expr)


@dataclass
class Program:
    channels: list[Channel] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)

    def channel_names(self) -> list[str]:
        return [c.name for c in self.channels]


# Infix binding powers (left, right). Higher binds tighter.
_INFIX: dict[str, tuple[int, int]] = {
    "or": (1, 2),
    "and": (3, 4),
    "==": (5, 6), "!=": (5, 6), "<": (5, 6), "<=": (5, 6), ">": (5, 6), ">=": (5, 6),
    "+": (7, 8), "-": (7, 8),
    "*": (9, 10), "/": (9, 10), "%": (9, 10),
}
_NOT_BP = 5    # operand of `not` includes comparisons, excludes and/or
_NEG_BP = 11   # unary minus binds tighter than * /


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize(text)
        self.i = 0

    def _peek(self) -> Token:
        return self.tokens[self.i]

    def _next(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _error(self, tok: Token, msg: str) -> NomosError:
        line = self.text.count("\n", 0, tok.pos) + 1
        return NomosError(f"line {line}: {msg}")

    def _expect_op(self, op: str) -> None:
        tok = self._next()
        if tok.kind != "op" or tok.value != op:
            raise self._error(tok, f"expected {op!r}, got {tok.value!r}")

    def _expect_name(self, what: str) -> str:
        tok = self._next()
        if tok.kind != "name" or tok.value in KEYWORDS:
            raise self._error(tok, f"expected {what}, got {tok.value!r}")
        return tok.value

    def _skip_colon(self) -> None:
        if self._peek().kind == "op" and self._peek().value == ":":
            self._next()

    def _at_keyword(self, *words: str) -> bool:
        tok = self._peek()
        return tok.kind == "name" and tok.value in words

    # ---- statements ----

    def parse_program(self) -> Program:
        prog = Program()
        while self._peek().kind != "eof":
            if self._at_keyword("channel"):
                self._next()
                name = self._expect_name("channel name")
                self._skip_colon()
                init = None
                if self._at_keyword("init"):
                    self._next()
                    self._skip_colon()
                    init = self.parse_expr(0)
                prog.channels.append(Channel(name, init))
            elif self._at_keyword("rule"):
                self._next()
                name = self._expect_name("rule name")
                self._skip_colon()
                tok = self._next()
                if not (tok.kind == "name" and tok.value == "when"):
                    raise self._error(tok, f"expected 'when' in rule {name!r}")
                when = self.parse_expr(0)
                actions = []
                while self._at_keyword("set", "add"):
                    op = self._next().value
                    ch = self._expect_name("channel name")
                    self._expect_op("=")
                    actions.append((op, ch, self.parse_expr(0)))
                if not actions:
                    raise self._error(tok, f"rule {name!r} has no set/add actions")
                prog.rules.append(Rule(name, when, actions))
            else:
                raise self._error(
                    self._peek(),
                    f"expected 'channel' or 'rule', got {self._peek().value!r}",
                )
        return prog

    # ---- expressions (Pratt) ----

    def parse_expr(self, min_bp: int) -> tuple:
        tok = self._next()
        if tok.kind == "num":
            left: tuple = ("num", float(tok.value))
        elif tok.kind == "name" and tok.value == "not":
            left = ("un", "not", self.parse_expr(_NOT_BP))
        elif tok.kind == "name" and tok.value in KEYWORDS:
            raise self._error(tok, f"unexpected keyword {tok.value!r} in expression")
        elif tok.kind == "name":
            if self._peek().kind == "op" and self._peek().value == "(":
                self._next()
                args = []
                if not (self._peek().kind == "op" and self._peek().value == ")"):
                    args.append(self.parse_expr(0))
                    while self._peek().kind == "op" and self._peek().value == ",":
                        self._next()
                        args.append(self.parse_expr(0))
                self._expect_op(")")
                left = ("call", tok.value, args)
            else:
                left = ("name", tok.value)
        elif tok.kind == "op" and tok.value == "(":
            left = self.parse_expr(0)
            self._expect_op(")")
        elif tok.kind == "op" and tok.value == "-":
            left = ("un", "neg", self.parse_expr(_NEG_BP))
        else:
            raise self._error(tok, f"unexpected token {tok.value!r} in expression")

        while True:
            tok = self._peek()
            op = tok.value
            if (tok.kind == "op" or (tok.kind == "name" and op in ("and", "or"))) and op in _INFIX:
                lbp, rbp = _INFIX[op]
                if lbp < min_bp:
                    break
                self._next()
                left = ("bin", op, left, self.parse_expr(rbp))
            else:
                break
        return left


def parse_program(text: str) -> Program:
    """Parse a full law file into a Program."""
    return _Parser(text).parse_program()


def parse_expression(text: str) -> tuple:
    """Parse a single expression (used in tests and tooling)."""
    parser = _Parser(text)
    expr = parser.parse_expr(0)
    tok = parser._peek()
    if tok.kind != "eof":
        raise parser._error(tok, f"trailing input after expression: {tok.value!r}")
    return expr


def walk_names(expr: tuple):
    """Yield every channel reference ("name" node) in an expression tree."""
    kind = expr[0]
    if kind == "name":
        yield expr[1]
    elif kind == "un":
        yield from walk_names(expr[2])
    elif kind == "bin":
        yield from walk_names(expr[2])
        yield from walk_names(expr[3])
    elif kind == "call":
        for arg in expr[2]:
            yield from walk_names(arg)


def walk_calls(expr: tuple):
    """Yield every function name used in an expression tree."""
    kind = expr[0]
    if kind == "call":
        yield expr[1]
        for arg in expr[2]:
            yield from walk_calls(arg)
    elif kind == "un":
        yield from walk_calls(expr[2])
    elif kind == "bin":
        yield from walk_calls(expr[2])
        yield from walk_calls(expr[3])
