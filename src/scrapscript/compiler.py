#!/usr/bin/env python3.10
import base64
import dataclasses
import enum
import http.server
import json
import logging
import os
import typing
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import auto
from types import FunctionType
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


def is_identifier_char(c: str) -> bool:
    return c.isalnum() or c in ("$", "'", "_")


@dataclass(eq=True)
class Token:
    lineno: int = dataclasses.field(default=-1, init=False, compare=False)


@dataclass(eq=True)
class IntLit(Token):
    value: int


@dataclass(eq=True)
class FloatLit(Token):
    value: float


@dataclass(eq=True)
class StringLit(Token):
    value: str


@dataclass(eq=True)
class BytesLit(Token):
    value: str
    base: int


@dataclass(eq=True)
class Operator(Token):
    value: str


@dataclass(eq=True)
class Name(Token):
    value: str


@dataclass(eq=True)
class LeftParen(Token):
    # (
    pass


@dataclass(eq=True)
class RightParen(Token):
    # )
    pass


@dataclass(eq=True)
class LeftBrace(Token):
    # {
    pass


@dataclass(eq=True)
class RightBrace(Token):
    # }
    pass


@dataclass(eq=True)
class LeftBracket(Token):
    # [
    pass


@dataclass(eq=True)
class RightBracket(Token):
    # ]
    pass


@dataclass(eq=True)
class Juxt(Token):
    # The space between other tokens that indicates function application.
    pass


@dataclass(eq=True)
class SymbolToken(Token):
    value: str


@dataclass(eq=True)
class EOF(Token):
    pass


class Lexer:
    def __init__(self, text: str):
        self.text: str = text
        self.idx: int = 0
        self.lineno: int = 1
        self.colno: int = 1
        self.line: str = ""

    def has_input(self) -> bool:
        return self.idx < len(self.text)

    def read_char(self) -> str:
        c = self.peek_char()
        if c == "\n":
            self.lineno += 1
            self.colno = 1
            self.line = ""
        else:
            self.line += c
            self.colno += 1
        self.idx += 1
        return c

    def peek_char(self) -> str:
        if not self.has_input():
            raise UnexpectedEOFError("while reading token")
        return self.text[self.idx]

    def make_token(self, cls: type, *args: Any) -> Token:
        result: Token = cls(*args)
        result.lineno = self.lineno
        return result

    def read_one(self) -> Token:
        while self.has_input():
            c = self.read_char()
            if not c.isspace():
                break
        else:
            return self.make_token(EOF)
        if c == '"':
            return self.read_string()
        if c == "-":
            if self.has_input() and self.peek_char() == "-":
                self.read_comment()
                return self.read_one()
            return self.read_op(c)
        if c == "#":
            value = self.read_one()
            if isinstance(value, EOF):
                raise UnexpectedEOFError("while reading symbol")
            if not isinstance(value, Name):
                raise ParseError(f"expected name after #, got {value!r}")
            return self.make_token(SymbolToken, value.value)
        if c == "~":
            if self.has_input() and self.peek_char() == "~":
                self.read_char()
                return self.read_bytes()
            raise ParseError(f"unexpected token {c!r}")
        if c.isdigit():
            return self.read_number(c)
        if c in "()[]{}":
            custom = {
                "(": LeftParen,
                ")": RightParen,
                "{": LeftBrace,
                "}": RightBrace,
                "[": LeftBracket,
                "]": RightBracket,
            }
            return self.make_token(custom[c])
        if c in OPER_CHARS:
            return self.read_op(c)
        if is_identifier_char(c):
            return self.read_var(c)
        raise ParseError(f"unexpected token {c!r}", ("<input>", self.lineno, self.colno, self.line))

    def read_string(self) -> Token:
        buf = ""
        while self.has_input():
            if (c := self.read_char()) == '"':
                break
            buf += c
        else:
            raise UnexpectedEOFError("while reading string")
        return self.make_token(StringLit, buf)

    def read_comment(self) -> None:
        while self.has_input() and self.read_char() != "\n":
            pass

    def read_number(self, first_digit: str) -> Token:
        # TODO: Support floating point numbers with no integer part
        buf = first_digit
        has_decimal = False
        while self.has_input():
            c = self.peek_char()
            if c == ".":
                if has_decimal:
                    raise ParseError(f"unexpected token {c!r}")
                has_decimal = True
            elif not c.isdigit():
                break
            self.read_char()
            buf += c

        if has_decimal:
            return self.make_token(FloatLit, float(buf))
        return self.make_token(IntLit, int(buf))

    def _starts_operator(self, buf: str) -> bool:
        # TODO(max): Rewrite using trie
        return any(op.startswith(buf) for op in PS.keys())

    def read_op(self, first_char: str) -> Token:
        buf = first_char
        while self.has_input():
            c = self.peek_char()
            if not self._starts_operator(buf + c):
                break
            self.read_char()
            buf += c
        if buf in PS.keys():
            return self.make_token(Operator, buf)
        raise ParseError(f"unexpected token {buf!r}")

    def read_var(self, first_char: str) -> Token:
        buf = first_char
        while self.has_input() and is_identifier_char(c := self.peek_char()):
            self.read_char()
            buf += c
        return self.make_token(Name, buf)

    def read_bytes(self) -> Token:
        buf = ""
        while self.has_input():
            if (c := self.read_char()).isspace():
                break
            buf += c
        base, _, value = buf.rpartition("'")
        return self.make_token(BytesLit, value, int(base) if base else 64)


def tokenize(x: str) -> typing.List[Token]:
    lexer = Lexer(x)
    tokens = []
    while (token := lexer.read_one()) and not isinstance(token, EOF):
        tokens.append(token)
    return tokens


@dataclass(frozen=True)
class Prec:
    pl: float
    pr: float


def lp(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, n - 0.1)


def rp(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, n + 0.1)


def np(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, n)


def xp(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, 0)


PS = {
    "::": lp(2000),
    "@": rp(1001),
    "": rp(1000),
    ">>": lp(14),
    "<<": lp(14),
    "^": rp(13),
    "*": lp(12),
    "/": lp(12),
    "//": lp(12),
    "%": lp(12),
    "+": lp(11),
    "-": lp(11),
    ">*": rp(10),
    "++": rp(10),
    ">+": lp(10),
    "+<": rp(10),
    "==": np(9),
    "/=": np(9),
    "<": np(9),
    ">": np(9),
    "<=": np(9),
    ">=": np(9),
    "&&": rp(8),
    "||": rp(7),
    "|>": rp(6),
    "<|": lp(6),
    "->": lp(5),
    "|": rp(4.5),
    ":": lp(4.5),
    "=": rp(4),
    "!": lp(3),
    ".": rp(3),
    "?": rp(3),
    ",": xp(1),
    # TODO: Fix precedence for spread
    "...": xp(0),
}


HIGHEST_PREC: float = max(max(p.pl, p.pr) for p in PS.values())


OPER_CHARS = set("".join(PS.keys()))
assert " " not in OPER_CHARS


class ParseError(SyntaxError):
    pass


# TODO(max): Replace with EOFError?
class UnexpectedEOFError(ParseError):
    pass


def parse_assign(tokens: typing.List[Token], p: float = 0) -> "Assign":
    assign = parse(tokens, p)
    if isinstance(assign, Spread):
        return Assign(Var("..."), assign)
    if not isinstance(assign, Assign):
        raise ParseError("failed to parse variable assignment in record constructor")
    return assign


def parse(tokens: typing.List[Token], p: float = 0) -> "Object":
    if not tokens:
        raise UnexpectedEOFError("unexpected end of input")
    token = tokens.pop(0)
    l: Object
    if isinstance(token, IntLit):
        l = Int(token.value)
    elif isinstance(token, FloatLit):
        l = Float(token.value)
    elif isinstance(token, Name):
        # TODO: Handle kebab case vars
        l = Var(token.value)
    elif isinstance(token, SymbolToken):
        l = Symbol(token.value)
    elif isinstance(token, BytesLit):
        base = token.base
        if base == 85:
            l = Bytes(base64.b85decode(token.value))
        elif base == 64:
            l = Bytes(base64.b64decode(token.value))
        elif base == 32:
            l = Bytes(base64.b32decode(token.value))
        elif base == 16:
            l = Bytes(base64.b16decode(token.value))
        else:
            raise ParseError(f"unexpected base {base!r} in {token!r}")
    elif isinstance(token, StringLit):
        l = String(token.value)
    elif token == Operator("..."):
        if tokens and isinstance(tokens[0], Name):
            name = tokens[0].value
            tokens.pop(0)
            l = Spread(name)
        else:
            l = Spread()
    elif token == Operator("|"):
        expr = parse(tokens, PS["|"].pr)  # TODO: make this work for larger arities
        if not isinstance(expr, Function):
            raise ParseError(f"expected function in match expression {expr!r}")
        cases = [MatchCase(expr.arg, expr.body)]
        while tokens and tokens[0] == Operator("|"):
            tokens.pop(0)
            expr = parse(tokens, PS["|"].pr)  # TODO: make this work for larger arities
            if not isinstance(expr, Function):
                raise ParseError(f"expected function in match expression {expr!r}")
            cases.append(MatchCase(expr.arg, expr.body))
        l = MatchFunction(cases)
    elif isinstance(token, LeftParen):
        if isinstance(tokens[0], RightParen):
            l = Hole()
        else:
            l = parse(tokens)
        tokens.pop(0)
    elif isinstance(token, LeftBracket):
        l = List([])
        token = tokens[0]
        if isinstance(token, RightBracket):
            tokens.pop(0)
        else:
            l.items.append(parse(tokens, 2))
            while not isinstance(tokens.pop(0), RightBracket):
                if isinstance(l.items[-1], Spread):
                    raise ParseError("spread must come at end of list match")
                # TODO: Implement .. operator
                l.items.append(parse(tokens, 2))
    elif isinstance(token, LeftBrace):
        l = Record({})
        token = tokens[0]
        if isinstance(token, RightBrace):
            tokens.pop(0)
        else:
            assign = parse_assign(tokens, 2)
            l.data[assign.name.name] = assign.value
            while not isinstance(tokens.pop(0), RightBrace):
                if isinstance(assign.value, Spread):
                    raise ParseError("spread must come at end of record match")
                # TODO: Implement .. operator
                assign = parse_assign(tokens, 2)
                l.data[assign.name.name] = assign.value
    elif token == Operator("-"):
        # Unary minus
        # Precedence was chosen to be higher than binary ops so that -a op
        # b is (-a) op b and not -(a op b).
        # Precedence was chosen to be higher than function application so that
        # -a b is (-a) b and not -(a b).
        r = parse(tokens, HIGHEST_PREC + 1)
        l = Binop(BinopKind.SUB, Int(0), r)
    else:
        raise ParseError(f"unexpected token {token!r}")

    while True:
        if not tokens:
            break
        op = tokens[0]
        if isinstance(op, (RightParen, RightBracket, RightBrace)):
            break
        if not isinstance(op, Operator):
            prec = PS[""]
            pl, pr = prec.pl, prec.pr
            if pl < p:
                break
            l = Apply(l, parse(tokens, pr))
            continue
        prec = PS[op.value]
        pl, pr = prec.pl, prec.pr
        if pl < p:
            break
        tokens.pop(0)
        if op == Operator("="):
            if not isinstance(l, Var):
                raise ParseError(f"expected variable in assignment {l!r}")
            l = Assign(l, parse(tokens, pr))
        elif op == Operator("->"):
            l = Function(l, parse(tokens, pr))
        elif op == Operator("|>"):
            l = Apply(parse(tokens, pr), l)
        elif op == Operator("<|"):
            l = Apply(l, parse(tokens, pr))
        elif op == Operator(">>"):
            l = Compose(l, parse(tokens, pr))
        elif op == Operator("<<"):
            l = Compose(parse(tokens, pr), l)
        elif op == Operator("."):
            l = Where(l, parse(tokens, pr))
        elif op == Operator("?"):
            l = Assert(l, parse(tokens, pr))
        elif op == Operator("@"):
            # TODO: revisit whether to use @ or . for field access
            l = Access(l, parse(tokens, pr))
        else:
            assert not isinstance(op, Juxt)
            assert isinstance(op, Operator)
            l = Binop(BinopKind.from_str(op.value), l, parse(tokens, pr))
    return l


OBJECT_DESERIALIZERS: Dict[str, FunctionType] = {}
OBJECT_TYPES: Dict[str, type] = {}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Object:
    def __init_subclass__(cls, /, **kwargs: Dict[Any, Any]) -> None:
        super().__init_subclass__(**kwargs)
        OBJECT_TYPES[cls.__name__] = cls
        deserializer = cls.__dict__.get("deserialize", None)
        if deserializer:
            assert isinstance(deserializer, staticmethod)
            func = deserializer.__func__
            assert isinstance(func, FunctionType)
            OBJECT_DESERIALIZERS[cls.__name__] = func

    def serialize(self) -> Dict[str, object]:
        cls = type(self)
        result: Dict[str, object] = {"type": cls.__name__}
        for field in dataclasses.fields(cls):
            if issubclass(field.type, Object):
                value = getattr(self, field.name)
                result[field.name] = value.serialize()
            else:
                raise NotImplementedError("serializing non-Object fields; write your own serialize function")
        return result

    def _serialize(self, **kwargs: object) -> Dict[str, object]:
        return {"type": type(self).__name__, **kwargs}

    @staticmethod
    def deserialize(msg: Dict[str, Any]) -> "Object":
        assert "type" in msg, f"No idea what to do with {msg!r}"
        ty = msg["type"]
        assert isinstance(ty, str)
        deserializer = OBJECT_DESERIALIZERS.get(ty)
        if deserializer:
            result = deserializer(msg)
            assert isinstance(result, Object)
            return result
        cls = OBJECT_TYPES[ty]
        kwargs = {}
        for field in dataclasses.fields(cls):
            if issubclass(field.type, Object):
                kwargs[field.name] = Object.deserialize(msg[field.name])
            else:
                raise NotImplementedError("deserializing non-Object fields; write your own deserialize function")
        result = cls(**kwargs)
        assert isinstance(result, Object)
        return result

    def __str__(self) -> str:
        raise NotImplementedError("__str__ not implemented for superclass Object")


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Int(Object):
    value: int

    def serialize(self) -> Dict[str, object]:
        return self._serialize(value=self.value)

    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "Int":
        assert msg["type"] == "Int"
        assert isinstance(msg["value"], int)
        return Int(msg["value"])

    def __str__(self) -> str:
        return str(self.value)


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Float(Object):
    value: float

    def serialize(self) -> Dict[str, object]:
        raise NotImplementedError("serialization for Float is not supported")

    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "Float":
        raise NotImplementedError("serialization for Float is not supported")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class String(Object):
    value: str

    def serialize(self) -> Dict[str, object]:
        return {"type": "String", "value": self.value}

    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "String":
        assert msg["type"] == "String"
        assert isinstance(msg["value"], str)
        return String(msg["value"])

    def __str__(self) -> str:
        # TODO: handle nested quotes
        return f'"{self.value}"'


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Bytes(Object):
    value: bytes

    def serialize(self) -> Dict[str, object]:
        return {"type": "Bytes", "value": str(self)}

    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "Bytes":
        assert msg["type"] == "Bytes"
        assert isinstance(msg["value"], bytes)
        return Bytes(msg["value"])

    def __str__(self) -> str:
        return f"~~{base64.b64encode(self.value).decode()}"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Var(Object):
    name: str

    def serialize(self) -> Dict[str, object]:
        return {"type": "Var", "name": self.name}

    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "Var":
        assert msg["type"] == "Var"
        assert isinstance(msg["name"], str)
        return Var(msg["name"])

    def __str__(self) -> str:
        return self.name


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Hole(Object):
    def __str__(self) -> str:
        return "()"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Spread(Object):
    name: Optional[str] = None

    def __str__(self) -> str:
        return "..." if self.name is None else f"...{self.name}"


Env = Mapping[str, Object]


class BinopKind(enum.Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FLOOR_DIV = auto()
    EXP = auto()
    MOD = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    BOOL_AND = auto()
    BOOL_OR = auto()
    STRING_CONCAT = auto()
    LIST_CONS = auto()
    LIST_APPEND = auto()
    RIGHT_EVAL = auto()
    HASTYPE = auto()
    PIPE = auto()
    REVERSE_PIPE = auto()

    @classmethod
    def from_str(cls, x: str) -> "BinopKind":
        return {
            "+": cls.ADD,
            "-": cls.SUB,
            "*": cls.MUL,
            "/": cls.DIV,
            "//": cls.FLOOR_DIV,
            "^": cls.EXP,
            "%": cls.MOD,
            "==": cls.EQUAL,
            "/=": cls.NOT_EQUAL,
            "<": cls.LESS,
            ">": cls.GREATER,
            "<=": cls.LESS_EQUAL,
            ">=": cls.GREATER_EQUAL,
            "&&": cls.BOOL_AND,
            "||": cls.BOOL_OR,
            "++": cls.STRING_CONCAT,
            ">+": cls.LIST_CONS,
            "+<": cls.LIST_APPEND,
            "!": cls.RIGHT_EVAL,
            ":": cls.HASTYPE,
            "|>": cls.PIPE,
            "<|": cls.REVERSE_PIPE,
        }[x]

    @classmethod
    def to_str(cls, binop_kind: "BinopKind") -> str:
        return {
            cls.ADD: "+",
            cls.SUB: "-",
            cls.MUL: "*",
            cls.DIV: "/",
            cls.EXP: "^",
            cls.MOD: "%",
            cls.EQUAL: "==",
            cls.NOT_EQUAL: "/=",
            cls.LESS: "<",
            cls.GREATER: ">",
            cls.LESS_EQUAL: "<=",
            cls.GREATER_EQUAL: ">=",
            cls.BOOL_AND: "&&",
            cls.BOOL_OR: "||",
            cls.STRING_CONCAT: "++",
            cls.LIST_CONS: ">+",
            cls.LIST_APPEND: "+<",
            cls.RIGHT_EVAL: "!",
            cls.HASTYPE: ":",
            cls.PIPE: "|>",
            cls.REVERSE_PIPE: "<|",
        }[binop_kind]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Binop(Object):
    op: BinopKind
    left: Object
    right: Object

    def serialize(self) -> Dict[str, object]:
        return {
            "type": "Binop",
            "op": self.op.name,
            "left": self.left.serialize(),
            "right": self.right.serialize(),
        }

    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "Binop":
        assert msg["type"] == "Binop"
        opname = msg["op"]
        assert isinstance(opname, str)
        op = BinopKind[opname]
        assert isinstance(op, BinopKind)
        left_obj = msg["left"]
        assert isinstance(left_obj, dict)
        right_obj = msg["right"]
        assert isinstance(right_obj, dict)
        left = Object.deserialize(left_obj)
        right = Object.deserialize(right_obj)
        return Binop(op, left, right)

    def __str__(self) -> str:
        return f"{self.left} {BinopKind.to_str(self.op)} {self.right}"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class List(Object):
    items: typing.List[Object]

    def serialize(self) -> Dict[str, object]:
        return {"type": "List", "items": [item.serialize() for item in self.items]}

    def __str__(self) -> str:
        inner = ", ".join(str(item) for item in self.items)
        return f"[{inner}]"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Assign(Object):
    name: Var
    value: Object

    def __str__(self) -> str:
        return f"{self.name} = {self.value}"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Function(Object):
    arg: Object
    body: Object

    def __str__(self) -> str:
        # TODO: Better pretty printing for Function
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Apply(Object):
    func: Object
    arg: Object

    def __str__(self) -> str:
        # TODO: Better pretty printing for Apply
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Compose(Object):
    inner: Object
    outer: Object

    def __str__(self) -> str:
        # TODO: Better pretty printing for Compose
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Where(Object):
    body: Object
    binding: Object

    def __str__(self) -> str:
        # TODO: Better pretty printing for Where
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Assert(Object):
    value: Object
    cond: Object

    def __str__(self) -> str:
        # TODO: Better pretty printing for Assert
        return self.__repr__()


def serialize_env(env: Env) -> Dict[str, object]:
    return {key: value.serialize() for key, value in env.items()}


def deserialize_env(msg: Dict[str, Any]) -> Env:
    return {key: Object.deserialize(value) for key, value in msg.items()}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class EnvObject(Object):
    env: Env

    def serialize(self) -> Dict[str, object]:
        return self._serialize(env=serialize_env(self.env))

    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "EnvObject":
        assert msg["type"] == "EnvObject"
        env_obj = msg["env"]
        assert isinstance(env_obj, dict)
        env = deserialize_env(env_obj)
        return EnvObject(env)

    def __str__(self) -> str:
        return f"EnvObject(keys={self.env.keys()})"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class MatchCase(Object):
    pattern: Object
    body: Object

    def __str__(self) -> str:
        # TODO: Better pretty printing for MatchCase
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class MatchFunction(Object):
    cases: typing.List[MatchCase]

    def serialize(self) -> Dict[str, object]:
        return self._serialize(cases=[case.serialize() for case in self.cases])

    def __str__(self) -> str:
        # TODO: Better pretty printing for MatchFunction
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Relocation(Object):
    name: str

    def serialize(self) -> Dict[str, object]:
        return self._serialize(name=self.name)

    def __str__(self) -> str:
        # TODO: Better pretty printing for Relocation
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class NativeFunctionRelocation(Relocation):
    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "NativeFunction":
        # TODO(max): Should this return a Var or an actual
        # NativeFunctionRelocation object instead of doing the relocation in
        # the deserialization?
        assert msg["type"] == "NativeFunctionRelocation"
        name = msg["name"]
        assert isinstance(name, str)
        result = STDLIB[name]
        assert isinstance(result, NativeFunction)
        return result

    def __str__(self) -> str:
        # TODO: Better pretty printing for NativeFunctionRelocation
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class NativeFunction(Object):
    name: str
    func: Callable[[Object], Object]

    def serialize(self) -> Dict[str, object]:
        return NativeFunctionRelocation(self.name).serialize()

    def __str__(self) -> str:
        # TODO: Better pretty printing for NativeFunction
        return f"NativeFunction(name={self.name})"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Closure(Object):
    env: Env
    func: Union[Function, MatchFunction]

    def serialize(self) -> Dict[str, object]:
        return self._serialize(env=serialize_env(self.env), func=self.func.serialize())

    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "Closure":
        assert msg["type"] == "Closure"
        env_obj = msg["env"]
        assert isinstance(env_obj, dict)
        env = deserialize_env(env_obj)
        func_obj = msg["func"]
        assert isinstance(func_obj, dict)
        func = Object.deserialize(func_obj)
        assert isinstance(func, (Function, MatchFunction))
        return Closure(env, func)

    def __str__(self) -> str:
        # TODO: Better pretty printing for Closure
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Record(Object):
    data: Dict[str, Object]

    def serialize(self) -> Dict[str, object]:
        return self._serialize(data={key: value.serialize() for key, value in self.data.items()})

    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "Record":
        assert msg["type"] == "Record"
        data_obj = msg["data"]
        assert isinstance(data_obj, dict)
        data = {key: Object.deserialize(value) for key, value in data_obj.items()}
        return Record(data)

    def __str__(self) -> str:
        inner = ", ".join(f"{k} = {self.data[k]}" for k in self.data)
        return f"{{{inner}}}"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Access(Object):
    obj: Object
    at: Object

    def __str__(self) -> str:
        # TODO: Better pretty printing for Access
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Symbol(Object):
    value: str

    def serialize(self) -> Dict[str, object]:
        return self._serialize(value=self.value)

    @staticmethod
    def deserialize(msg: Dict[str, object]) -> "Symbol":
        assert msg["type"] == "Symbol"
        value_obj = msg["value"]
        assert isinstance(value_obj, str)
        return Symbol(value_obj)

    def __str__(self) -> str:
        return f"#{self.value}"


def unpack_number(obj: Object) -> Union[int, float]:
    if not isinstance(obj, (Int, Float)):
        raise TypeError(f"expected Int or Float, got {type(obj).__name__}")
    return obj.value


def eval_number(env: Env, exp: Object) -> Union[int, float]:
    result = eval_exp(env, exp)
    return unpack_number(result)


def eval_str(env: Env, exp: Object) -> str:
    result = eval_exp(env, exp)
    if not isinstance(result, String):
        raise TypeError(f"expected String, got {type(result).__name__}")
    return result.value


def eval_bool(env: Env, exp: Object) -> bool:
    result = eval_exp(env, exp)
    if isinstance(result, Symbol) and result.value in ("true", "false"):
        return result.value == "true"
    raise TypeError(f"expected #true or #false, got {type(result).__name__}")


def eval_list(env: Env, exp: Object) -> typing.List[Object]:
    result = eval_exp(env, exp)
    if not isinstance(result, List):
        raise TypeError(f"expected List, got {type(result).__name__}")
    return result.items


def make_bool(x: bool) -> Object:
    return Symbol("true" if x else "false")


def wrap_inferred_number_type(x: Union[int, float]) -> Object:
    # TODO: Since this is intended to be a reference implementation
    # we should avoid relying heavily on Python's implementation of
    # arithmetic operations, type inference, and multiple dispatch.
    # Update this to make the interpreter more language agnostic.
    if isinstance(x, int):
        return Int(x)
    return Float(x)


BINOP_HANDLERS: Dict[BinopKind, Callable[[Env, Object, Object], Object]] = {
    BinopKind.ADD: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) + eval_number(env, y)),
    BinopKind.SUB: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) - eval_number(env, y)),
    BinopKind.MUL: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) * eval_number(env, y)),
    BinopKind.DIV: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) / eval_number(env, y)),
    BinopKind.FLOOR_DIV: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) // eval_number(env, y)),
    BinopKind.EXP: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) ** eval_number(env, y)),
    BinopKind.MOD: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) % eval_number(env, y)),
    BinopKind.EQUAL: lambda env, x, y: make_bool(eval_exp(env, x) == eval_exp(env, y)),
    BinopKind.NOT_EQUAL: lambda env, x, y: make_bool(eval_exp(env, x) != eval_exp(env, y)),
    BinopKind.LESS: lambda env, x, y: make_bool(eval_number(env, x) < eval_number(env, y)),
    BinopKind.GREATER: lambda env, x, y: make_bool(eval_number(env, x) > eval_number(env, y)),
    BinopKind.LESS_EQUAL: lambda env, x, y: make_bool(eval_number(env, x) <= eval_number(env, y)),
    BinopKind.GREATER_EQUAL: lambda env, x, y: make_bool(eval_number(env, x) >= eval_number(env, y)),
    BinopKind.BOOL_AND: lambda env, x, y: make_bool(eval_bool(env, x) and eval_bool(env, y)),
    BinopKind.BOOL_OR: lambda env, x, y: make_bool(eval_bool(env, x) or eval_bool(env, y)),
    BinopKind.STRING_CONCAT: lambda env, x, y: String(eval_str(env, x) + eval_str(env, y)),
    BinopKind.LIST_CONS: lambda env, x, y: List([eval_exp(env, x)] + eval_list(env, y)),
    BinopKind.LIST_APPEND: lambda env, x, y: List(eval_list(env, x) + [eval_exp(env, y)]),
    BinopKind.RIGHT_EVAL: lambda env, x, y: eval_exp(env, y),
}


class MatchError(Exception):
    pass


def match(obj: Object, pattern: Object) -> Optional[Env]:
    if isinstance(pattern, Int):
        return {} if isinstance(obj, Int) and obj.value == pattern.value else None
    if isinstance(pattern, Float):
        raise MatchError("pattern matching is not supported for Floats")
    if isinstance(pattern, String):
        return {} if isinstance(obj, String) and obj.value == pattern.value else None
    if isinstance(pattern, Var):
        return {pattern.name: obj}
    if isinstance(pattern, Symbol):
        return {} if isinstance(obj, Symbol) and obj.value == pattern.value else None
    if isinstance(pattern, Record):
        if not isinstance(obj, Record):
            return None
        result: Env = {}
        use_spread = False
        for key, pattern_item in pattern.data.items():
            if isinstance(pattern_item, Spread):
                use_spread = True
                break
            obj_item = obj.data.get(key)
            if obj_item is None:
                return None
            part = match(obj_item, pattern_item)
            if part is None:
                return None
            assert isinstance(result, dict)  # for .update()
            result.update(part)
        if not use_spread and len(pattern.data) != len(obj.data):
            return None
        return result
    if isinstance(pattern, List):
        if not isinstance(obj, List):
            return None
        result: Env = {}  # type: ignore
        use_spread = False
        for i, pattern_item in enumerate(pattern.items):
            if isinstance(pattern_item, Spread):
                use_spread = True
                if pattern_item.name is not None:
                    assert isinstance(result, dict)  # for .update()
                    result.update({pattern_item.name: List(obj.items[i:])})
                break
            if i >= len(obj.items):
                return None
            obj_item = obj.items[i]
            part = match(obj_item, pattern_item)
            if part is None:
                return None
            assert isinstance(result, dict)  # for .update()
            result.update(part)
        if not use_spread and len(pattern.items) != len(obj.items):
            return None
        return result
    raise NotImplementedError(f"match not implemented for {type(pattern).__name__}")


def free_in(exp: Object) -> Set[str]:
    if isinstance(exp, (Int, Float, String, Bytes, Hole, NativeFunction, Symbol)):
        return set()
    if isinstance(exp, Var):
        return {exp.name}
    if isinstance(exp, Spread):
        if exp.name is not None:
            return {exp.name}
        return set()
    if isinstance(exp, Binop):
        return free_in(exp.left) | free_in(exp.right)
    if isinstance(exp, List):
        if not exp.items:
            return set()
        return set.union(*(free_in(item) for item in exp.items))
    if isinstance(exp, Record):
        if not exp.data:
            return set()
        return set.union(*(free_in(value) for key, value in exp.data.items()))
    if isinstance(exp, Function):
        assert isinstance(exp.arg, Var)
        return free_in(exp.body) - {exp.arg.name}
    if isinstance(exp, MatchFunction):
        if not exp.cases:
            return set()
        return set.union(*(free_in(case) for case in exp.cases))
    if isinstance(exp, MatchCase):
        return free_in(exp.body) - free_in(exp.pattern)
    if isinstance(exp, Apply):
        return free_in(exp.func) | free_in(exp.arg)
    if isinstance(exp, Access):
        return free_in(exp.obj) | free_in(exp.at)
    if isinstance(exp, Where):
        assert isinstance(exp.binding, Assign)
        return (free_in(exp.body) - {exp.binding.name.name}) | free_in(exp.binding)
    if isinstance(exp, Assign):
        return free_in(exp.value)
    if isinstance(exp, Closure):
        # TODO(max): Should this remove the set of keys in the closure env?
        return free_in(exp.func)
    raise NotImplementedError(("free_in", type(exp)))


def improve_closure(closure: Closure) -> Closure:
    freevars = free_in(closure.func)
    env = {boundvar: value for boundvar, value in closure.env.items() if boundvar in freevars}
    return Closure(env, closure.func)


def eval_exp(env: Env, exp: Object) -> Object:
    logger.debug(exp)
    if isinstance(exp, (Int, Float, String, Bytes, Hole, Closure, NativeFunction, Symbol)):
        return exp
    if isinstance(exp, Var):
        value = env.get(exp.name)
        if value is None:
            raise NameError(f"name '{exp.name}' is not defined")
        return value
    if isinstance(exp, Binop):
        handler = BINOP_HANDLERS.get(exp.op)
        if handler is None:
            raise NotImplementedError(f"no handler for {exp.op}")
        return handler(env, exp.left, exp.right)
    if isinstance(exp, List):
        return List([eval_exp(env, item) for item in exp.items])
    if isinstance(exp, Record):
        return Record({k: eval_exp(env, exp.data[k]) for k in exp.data})
    if isinstance(exp, Assign):
        # TODO(max): Rework this. There's something about matching that we need
        # to figure out and implement.
        assert isinstance(exp.name, Var)
        value = eval_exp(env, exp.value)
        if isinstance(value, Closure):
            # We want functions to be able to call themselves without using the
            # Y combinator or similar, so we bind functions (and only
            # functions) using a letrec-like strategy. We augment their
            # captured environment with a binding to themselves.
            assert isinstance(value.env, dict)
            value.env[exp.name.name] = value
            # We still improve_closure here even though we also did it on
            # Closure creation because the Closure might not need a binding for
            # itself (it might not be recursive).
            value = improve_closure(value)
        return EnvObject({**env, exp.name.name: value})
    if isinstance(exp, Where):
        assert isinstance(exp.binding, Assign)
        res_env = eval_exp(env, exp.binding)
        assert isinstance(res_env, EnvObject)
        new_env = {**env, **res_env.env}
        return eval_exp(new_env, exp.body)
    if isinstance(exp, Assert):
        cond = eval_exp(env, exp.cond)
        if cond != Symbol("true"):
            raise AssertionError(f"condition {exp.cond} failed")
        return eval_exp(env, exp.value)
    if isinstance(exp, Function):
        if not isinstance(exp.arg, Var):
            raise RuntimeError(f"expected variable in function definition {exp.arg}")
        value = Closure(env, exp)
        value = improve_closure(value)
        return value
    if isinstance(exp, MatchFunction):
        value = Closure(env, exp)
        value = improve_closure(value)
        return value
    if isinstance(exp, Apply):
        if isinstance(exp.func, Var) and exp.func.name == "$$quote":
            return exp.arg
        callee = eval_exp(env, exp.func)
        arg = eval_exp(env, exp.arg)
        if isinstance(callee, NativeFunction):
            return callee.func(arg)
        if not isinstance(callee, Closure):
            raise TypeError(f"attempted to apply a non-closure of type {type(callee).__name__}")
        if isinstance(callee.func, Function):
            assert isinstance(callee.func.arg, Var)
            new_env = {**callee.env, callee.func.arg.name: arg}
            return eval_exp(new_env, callee.func.body)
        elif isinstance(callee.func, MatchFunction):
            for case in callee.func.cases:
                m = match(arg, case.pattern)
                if m is None:
                    continue
                return eval_exp({**callee.env, **m}, case.body)
            raise MatchError("no matching cases")
        else:
            raise TypeError(f"attempted to apply a non-function of type {type(callee.func).__name__}")
    if isinstance(exp, Access):
        obj = eval_exp(env, exp.obj)
        if isinstance(obj, Record):
            if not isinstance(exp.at, Var):
                raise TypeError(f"cannot access record field using {type(exp.at).__name__}, expected a field name")
            if exp.at.name not in obj.data:
                raise NameError(f"no assignment to {exp.at.name} found in record")
            return obj.data[exp.at.name]
        elif isinstance(obj, List):
            access_at = eval_exp(env, exp.at)
            if not isinstance(access_at, Int):
                raise TypeError(f"cannot index into list using type {type(access_at).__name__}, expected integer")
            if access_at.value < 0 or access_at.value >= len(obj.items):
                raise ValueError(f"index {access_at.value} out of bounds for list")
            return obj.items[access_at.value]
        raise TypeError(f"attempted to access from type {type(obj).__name__}")
    if isinstance(exp, Compose):
        clo_inner = eval_exp(env, exp.inner)
        clo_outer = eval_exp(env, exp.outer)
        return Closure({}, Function(Var("x"), Apply(clo_outer, Apply(clo_inner, Var("x")))))
    elif isinstance(exp, Spread):
        raise RuntimeError("cannot evaluate a spread")
    raise NotImplementedError(f"eval_exp not implemented for {exp}")


def bencode(obj: object) -> bytes:
    assert not isinstance(obj, bool)
    if isinstance(obj, int):
        return b"i" + str(int(obj)).encode("ascii") + b"e"
    if isinstance(obj, str):
        return bencode(obj.encode("utf-8"))
    if isinstance(obj, bytes):
        return str(len(obj)).encode("ascii") + b":" + obj
    if isinstance(obj, list):
        return b"l" + b"".join(bencode(x) for x in obj) + b"e"
    if isinstance(obj, dict):
        sorted_items = sorted(obj.items(), key=lambda x: x[0])
        return b"d" + b"".join(bencode(k) + bencode(v) for k, v in sorted_items) + b"e"
    raise NotImplementedError(f"bencode not implemented for {type(obj)}")


class Bdecoder:
    def __init__(self, msg: str) -> None:
        self.msg: str = msg
        self.idx: int = 0

    def peek(self) -> str:
        return self.msg[self.idx]

    def read(self) -> str:
        c = self.peek()
        self.idx += 1
        return c

    def decode_int(self) -> int:
        buf = ""
        while (c := self.read()) != "e":
            buf += c
        return int(buf)

    def decode_list(self) -> typing.List[Any]:
        result = []
        while self.peek() != "e":
            result.append(self.decode())
        assert self.read() == "e"
        return result

    def decode_dict(self) -> typing.Dict[Any, Any]:
        result: Dict[Any, Any] = {}
        while self.peek() != "e":
            key = self.decode()
            value = self.decode()
            result[key] = value
        assert self.read() == "e"
        return result

    def decode_str(self, start: str) -> str:
        len_buf = start
        while (c := self.peek()) != ":":
            assert c.isdigit()
            len_buf += c
            self.read()
        assert self.read() == ":"
        buf = ""
        for _ in range(int(len_buf)):
            buf += self.read()
        return buf

    def decode(self) -> object:
        ty = self.read()
        if ty == "i":
            return self.decode_int()
        if ty == "l":
            return self.decode_list()
        if ty == "d":
            return self.decode_dict()
        if ty.isdigit():
            return self.decode_str(ty)
        raise NotImplementedError(ty)


def bdecode(msg: str) -> object:
    return Bdecoder(msg).decode()


def serialize(obj: Object) -> bytes:
    return bencode(obj.serialize())


def deserialize(msg: str) -> Object:
    logging.debug("deserialize %s", msg)
    decoded = bdecode(msg)
    assert isinstance(decoded, dict)
    return Object.deserialize(decoded)


class ScrapMonad:
    def __init__(self, env: Env) -> None:
        assert isinstance(env, dict)  # for .copy()
        self.env: Env = env.copy()

    def bind(self, exp: Object) -> Tuple[Object, "ScrapMonad"]:
        env = self.env
        result = eval_exp(env, exp)
        if isinstance(result, EnvObject):
            return result, ScrapMonad({**env, **result.env})
        return result, ScrapMonad({**env, "_": result})


ASSET_DIR = os.path.dirname(__file__)


class ScrapReplServer(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        logger.debug("GET %s", self.path)
        parsed_path = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)
        logging.debug("PATH %s", parsed_path)
        logging.debug("QUERY %s", query)
        if parsed_path.path == "/repl":
            return self.do_repl()
        if parsed_path.path == "/eval":
            try:
                return self.do_eval(query)
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
                return
        if parsed_path.path == "/style.css":
            self.send_response(200)
            self.send_header("Content-type", "text/css")
            self.end_headers()
            with open(os.path.join(ASSET_DIR, "style.css"), "rb") as f:
                self.wfile.write(f.read())
            return
        return self.do_404()

    def do_repl(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        with open(os.path.join(ASSET_DIR, "repl.html"), "rb") as f:
            self.wfile.write(f.read())
        return

    def do_404(self) -> None:
        self.send_response(404)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"""try hitting <a href="/repl">/repl</a>""")
        return

    def do_eval(self, query: Dict[str, Any]) -> None:
        exp = query.get("exp")
        if exp is None:
            raise TypeError("Need expression to evaluate")
        if len(exp) != 1:
            raise TypeError("Need exactly one expression to evaluate")
        exp = exp[0]
        tokens = tokenize(exp)
        ast = parse(tokens)
        env = query.get("env")
        if env is None:
            env = STDLIB
        else:
            if len(env) != 1:
                raise TypeError("Need exactly one env")
            env_object = deserialize(env[0])
            assert isinstance(env_object, EnvObject)
            env = env_object.env
        logging.debug("env is %s", env)
        monad = ScrapMonad(env)
        result, next_monad = monad.bind(ast)
        serialized = EnvObject(next_monad.env).serialize()
        encoded = bencode(serialized)
        response = {"env": encoded.decode("utf-8"), "result": str(result)}
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))
        return


def fetch(url: Object) -> Object:
    if not isinstance(url, String):
        raise TypeError(f"fetch expected String, but got {type(url).__name__}")
    with urllib.request.urlopen(url.value) as f:
        return String(f.read().decode("utf-8"))


def make_object(pyobj: object) -> Object:
    assert not isinstance(pyobj, Object)
    if isinstance(pyobj, int):
        return Int(pyobj)
    if isinstance(pyobj, str):
        return String(pyobj)
    if isinstance(pyobj, list):
        return List([make_object(o) for o in pyobj])
    if isinstance(pyobj, dict):
        # Assumed to only be called with JSON, so string keys.
        return Record({key: make_object(value) for key, value in pyobj.items()})
    raise NotImplementedError(type(pyobj))


def jsondecode(obj: Object) -> Object:
    if not isinstance(obj, String):
        raise TypeError(f"jsondecode expected String, but got {type(obj).__name__}")
    data = json.loads(obj.value)
    return make_object(data)


def listlength(obj: Object) -> Object:
    # TODO(max): Implement in scrapscript once list pattern matching is
    # implemented.
    if not isinstance(obj, List):
        raise TypeError(f"listlength expected List, but got {type(obj).__name__}")
    return Int(len(obj.items))


STDLIB = {
    "$$add": Closure({}, Function(Var("x"), Function(Var("y"), Binop(BinopKind.ADD, Var("x"), Var("y"))))),
    "$$fetch": NativeFunction("$$fetch", fetch),
    "$$jsondecode": NativeFunction("$$jsondecode", jsondecode),
    "$$serialize": NativeFunction("$$serialize", lambda obj: Bytes(serialize(obj))),
    "$$listlength": NativeFunction("$$listlength", listlength),
}


PRELUDE = """
id = x -> x

. quicksort =
  | [] -> []
  | [p, ...xs] -> (concat ((quicksort (ltp xs p)) +< p) (quicksort (gtp xs p))
    . gtp = xs -> p -> filter (x -> x >= p) xs
    . ltp = xs -> p -> filter (x -> x < p) xs)

. filter = f ->
  | [] -> []
  | [x, ...xs] -> f x |> | #true -> x >+ filter f xs
                         | #false -> filter f xs

. concat = xs ->
  | [] -> xs
  | [y, ...ys] -> concat (xs +< y) ys

. map = f ->
  | [] -> []
  | [x, ...xs] -> f x >+ map f xs

. range =
  | 1 -> [0]
  | i -> range (i - 1) +< (i - 1)

. foldr = f -> a ->
  | [] -> a
  | [x, ...xs] -> f x (foldr f a xs)

. take =
  | 0 -> xs -> []
  | n ->
    | [] -> []
    | [x, ...xs] -> x >+ take (n - 1) xs

. all = f ->
  | [] -> #true
  | [x, ...xs] -> f x && all f xs

. any = f ->
  | [] -> #false
  | [x, ...xs] -> f x || any f xs
"""
