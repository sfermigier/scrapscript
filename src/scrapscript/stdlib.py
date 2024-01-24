from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from .ast import (Binop, BinopKind, Bytes, Closure, Function, Int, List,
                  NativeFunction, Object, Record, String, Var)


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

    def decode_list(self) -> list[Any]:
        result = []
        while self.peek() != "e":
            result.append(self.decode())
        assert self.read() == "e"
        return result

    def decode_dict(self) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
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
