#!/usr/bin/env python3.10
import base64
import dataclasses
import enum
import logging
import typing
from dataclasses import dataclass
from enum import auto
from types import FunctionType
from typing import Any, Dict, Optional, Union
from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

OBJECT_DESERIALIZERS: dict[str, FunctionType] = {}
OBJECT_TYPES: dict[str, type] = {}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Object:
    def __init_subclass__(cls, /, **kwargs: dict[Any, Any]) -> None:
        super().__init_subclass__(**kwargs)
        OBJECT_TYPES[cls.__name__] = cls
        deserializer = cls.__dict__.get("deserialize", None)
        if deserializer:
            assert isinstance(deserializer, staticmethod)
            func = deserializer.__func__
            assert isinstance(func, FunctionType)
            OBJECT_DESERIALIZERS[cls.__name__] = func

    def serialize(self) -> dict[str, object]:
        cls = type(self)
        result: dict[str, object] = {"type": cls.__name__}
        for field in dataclasses.fields(cls):
            if issubclass(field.type, Object):
                value = getattr(self, field.name)
                result[field.name] = value.serialize()
            else:
                raise NotImplementedError("serializing non-Object fields; write your own serialize function")
        return result

    def _serialize(self, **kwargs: object) -> dict[str, object]:
        return {"type": type(self).__name__, **kwargs}

    @staticmethod
    def deserialize(msg: dict[str, Any]) -> "Object":
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

    def serialize(self) -> dict[str, object]:
        return self._serialize(value=self.value)

    @staticmethod
    def deserialize(msg: dict[str, object]) -> "Int":
        assert msg["type"] == "Int"
        assert isinstance(msg["value"], int)
        return Int(msg["value"])

    def __str__(self) -> str:
        return str(self.value)


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Float(Object):
    value: float

    def serialize(self) -> dict[str, object]:
        raise NotImplementedError("serialization for Float is not supported")

    @staticmethod
    def deserialize(msg: dict[str, object]) -> "Float":
        raise NotImplementedError("serialization for Float is not supported")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class String(Object):
    value: str

    def serialize(self) -> dict[str, object]:
        return {"type": "String", "value": self.value}

    @staticmethod
    def deserialize(msg: dict[str, object]) -> "String":
        assert msg["type"] == "String"
        assert isinstance(msg["value"], str)
        return String(msg["value"])

    def __str__(self) -> str:
        # TODO: handle nested quotes
        return f'"{self.value}"'


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Bytes(Object):
    value: bytes

    def serialize(self) -> dict[str, object]:
        return {"type": "Bytes", "value": str(self)}

    @staticmethod
    def deserialize(msg: dict[str, object]) -> "Bytes":
        assert msg["type"] == "Bytes"
        assert isinstance(msg["value"], bytes)
        return Bytes(msg["value"])

    def __str__(self) -> str:
        return f"~~{base64.b64encode(self.value).decode()}"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Var(Object):
    name: str

    def serialize(self) -> dict[str, object]:
        return {"type": "Var", "name": self.name}

    @staticmethod
    def deserialize(msg: dict[str, object]) -> "Var":
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
    name: str | None = None

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

    def serialize(self) -> dict[str, object]:
        return {
            "type": "Binop",
            "op": self.op.name,
            "left": self.left.serialize(),
            "right": self.right.serialize(),
        }

    @staticmethod
    def deserialize(msg: dict[str, object]) -> "Binop":
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
    items: list[Object]

    def serialize(self) -> dict[str, object]:
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


def serialize_env(env: Env) -> dict[str, object]:
    return {key: value.serialize() for key, value in env.items()}


def deserialize_env(msg: dict[str, Any]) -> Env:
    return {key: Object.deserialize(value) for key, value in msg.items()}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class EnvObject(Object):
    env: Env

    def serialize(self) -> dict[str, object]:
        return self._serialize(env=serialize_env(self.env))

    @staticmethod
    def deserialize(msg: dict[str, object]) -> "EnvObject":
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
    cases: list[MatchCase]

    def serialize(self) -> dict[str, object]:
        return self._serialize(cases=[case.serialize() for case in self.cases])

    def __str__(self) -> str:
        # TODO: Better pretty printing for MatchFunction
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Relocation(Object):
    name: str

    def serialize(self) -> dict[str, object]:
        return self._serialize(name=self.name)

    def __str__(self) -> str:
        # TODO: Better pretty printing for Relocation
        return self.__repr__()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class NativeFunctionRelocation(Relocation):
    @staticmethod
    def deserialize(msg: dict[str, object]) -> "NativeFunction":
        # TODO(max): Should this return a Var or an actual
        # NativeFunctionRelocation object instead of doing the relocation in
        # the deserialization?
        assert msg["type"] == "NativeFunctionRelocation"
        name = msg["name"]
        assert isinstance(name, str)

        from .stdlib import STDLIB

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

    def serialize(self) -> dict[str, object]:
        return NativeFunctionRelocation(self.name).serialize()

    def __str__(self) -> str:
        # TODO: Better pretty printing for NativeFunction
        return f"NativeFunction(name={self.name})"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Closure(Object):
    env: Env
    func: Function | MatchFunction

    def serialize(self) -> dict[str, object]:
        return self._serialize(env=serialize_env(self.env), func=self.func.serialize())

    @staticmethod
    def deserialize(msg: dict[str, object]) -> "Closure":
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
    data: dict[str, Object]

    def serialize(self) -> dict[str, object]:
        return self._serialize(data={key: value.serialize() for key, value in self.data.items()})

    @staticmethod
    def deserialize(msg: dict[str, object]) -> "Record":
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

    def serialize(self) -> dict[str, object]:
        return self._serialize(value=self.value)

    @staticmethod
    def deserialize(msg: dict[str, object]) -> "Symbol":
        assert msg["type"] == "Symbol"
        value_obj = msg["value"]
        assert isinstance(value_obj, str)
        return Symbol(value_obj)

    def __str__(self) -> str:
        return f"#{self.value}"
