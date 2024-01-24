import re
import unittest
from typing import Optional

from scrapscript.lexer import tokenize, Lexer, StringLit
from scrapscript.parser import *
from scrapscript.cli import boot_env
from scrapscript.compiler import *
from scrapscript.stdlib import *


class TokenizerTests(unittest.TestCase):
    def test_tokenize_digit(self) -> None:
        self.assertEqual(tokenize("1"), [IntLit(1)])

    def test_tokenize_multiple_digits(self) -> None:
        self.assertEqual(tokenize("123"), [IntLit(123)])

    def test_tokenize_negative_int(self) -> None:
        self.assertEqual(tokenize("-123"), [Operator("-"), IntLit(123)])

    def test_tokenize_float(self) -> None:
        self.assertEqual(tokenize("3.14"), [FloatLit(3.14)])

    def test_tokenize_negative_float(self) -> None:
        self.assertEqual(tokenize("-3.14"), [Operator("-"), FloatLit(3.14)])

    @unittest.skip("TODO: support floats with no integer part")
    def test_tokenize_float_with_no_integer_part(self) -> None:
        self.assertEqual(tokenize(".14"), [FloatLit(0.14)])

    def test_tokenize_float_with_no_decimal_part(self) -> None:
        self.assertEqual(tokenize("10."), [FloatLit(10.0)])

    def test_tokenize_float_with_multiple_decimal_points_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token '.'")):
            tokenize("1.0.1")

    def test_tokenize_binop(self) -> None:
        self.assertEqual(tokenize("1 + 2"), [IntLit(1), Operator("+"), IntLit(2)])

    def test_tokenize_binop_no_spaces(self) -> None:
        self.assertEqual(tokenize("1+2"), [IntLit(1), Operator("+"), IntLit(2)])

    def test_tokenize_two_oper_chars_returns_two_ops(self) -> None:
        self.assertEqual(tokenize(",:"), [Operator(","), Operator(":")])

    def test_tokenize_binary_sub_no_spaces(self) -> None:
        self.assertEqual(tokenize("1-2"), [IntLit(1), Operator("-"), IntLit(2)])

    def test_tokenize_binop_var(self) -> None:
        ops = ["+", "-", "*", "/", "^", "%", "==", "/=", "<", ">", "<=", ">=", "&&", "||", "++", ">+", "+<"]
        for op in ops:
            with self.subTest(op=op):
                self.assertEqual(tokenize(f"a {op} b"), [Name("a"), Operator(op), Name("b")])
                self.assertEqual(tokenize(f"a{op}b"), [Name("a"), Operator(op), Name("b")])

    def test_tokenize_var(self) -> None:
        self.assertEqual(tokenize("abc"), [Name("abc")])

    @unittest.skip("TODO: make this fail to tokenize")
    def test_tokenize_var_with_quote(self) -> None:
        self.assertEqual(tokenize("sha1'abc"), [Name("sha1'abc")])

    def test_tokenize_dollar_sha1_var(self) -> None:
        self.assertEqual(tokenize("$sha1'foo"), [Name("$sha1'foo")])

    def test_tokenize_dollar_dollar_var(self) -> None:
        self.assertEqual(tokenize("$$bills"), [Name("$$bills")])

    def test_tokenize_dot_dot_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token '..'")):
            tokenize("..")

    def test_tokenize_spread(self) -> None:
        self.assertEqual(tokenize("..."), [Operator("...")])

    def test_ignore_whitespace(self) -> None:
        self.assertEqual(tokenize("1\n+\t2"), [IntLit(1), Operator("+"), IntLit(2)])

    def test_ignore_line_comment(self) -> None:
        self.assertEqual(tokenize("-- 1\n2"), [IntLit(2)])

    def test_tokenize_string(self) -> None:
        self.assertEqual(tokenize('"hello"'), [StringLit("hello")])

    def test_tokenize_string_with_spaces(self) -> None:
        self.assertEqual(tokenize('"hello world"'), [StringLit("hello world")])

    def test_tokenize_string_missing_end_quote_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(UnexpectedEOFError, "while reading string"):
            tokenize('"hello')

    def test_tokenize_with_trailing_whitespace(self) -> None:
        self.assertEqual(tokenize("- "), [Operator("-")])
        self.assertEqual(tokenize("-- "), [])
        self.assertEqual(tokenize("+ "), [Operator("+")])
        self.assertEqual(tokenize("123 "), [IntLit(123)])
        self.assertEqual(tokenize("abc "), [Name("abc")])
        self.assertEqual(tokenize("[ "), [LeftBracket()])
        self.assertEqual(tokenize("] "), [RightBracket()])

    def test_tokenize_empty_list(self) -> None:
        self.assertEqual(tokenize("[ ]"), [LeftBracket(), RightBracket()])

    def test_tokenize_empty_list_with_spaces(self) -> None:
        self.assertEqual(tokenize("[ ]"), [LeftBracket(), RightBracket()])

    def test_tokenize_list_with_items(self) -> None:
        self.assertEqual(tokenize("[ 1 , 2 ]"), [LeftBracket(), IntLit(1), Operator(","), IntLit(2), RightBracket()])

    def test_tokenize_list_with_no_spaces(self) -> None:
        self.assertEqual(tokenize("[1,2]"), [LeftBracket(), IntLit(1), Operator(","), IntLit(2), RightBracket()])

    def test_tokenize_function(self) -> None:
        self.assertEqual(
            tokenize("a -> b -> a + b"),
            [Name("a"), Operator("->"), Name("b"), Operator("->"), Name("a"), Operator("+"), Name("b")],
        )

    def test_tokenize_function_with_no_spaces(self) -> None:
        self.assertEqual(
            tokenize("a->b->a+b"),
            [Name("a"), Operator("->"), Name("b"), Operator("->"), Name("a"), Operator("+"), Name("b")],
        )

    def test_tokenize_where(self) -> None:
        self.assertEqual(tokenize("a . b"), [Name("a"), Operator("."), Name("b")])

    def test_tokenize_assert(self) -> None:
        self.assertEqual(tokenize("a ? b"), [Name("a"), Operator("?"), Name("b")])

    def test_tokenize_hastype(self) -> None:
        self.assertEqual(tokenize("a : b"), [Name("a"), Operator(":"), Name("b")])

    def test_tokenize_minus_returns_minus(self) -> None:
        self.assertEqual(tokenize("-"), [Operator("-")])

    def test_tokenize_tilde_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token '~'"):
            tokenize("~")

    def test_tokenize_tilde_equals_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token '~'"):
            tokenize("~=")

    def test_tokenize_tilde_tilde_returns_empty_bytes(self) -> None:
        self.assertEqual(tokenize("~~"), [BytesLit("", 64)])

    def test_tokenize_bytes_returns_bytes_base64(self) -> None:
        self.assertEqual(tokenize("~~QUJD"), [BytesLit("QUJD", 64)])

    def test_tokenize_bytes_base85(self) -> None:
        self.assertEqual(tokenize("~~85'K|(_"), [BytesLit("K|(_", 85)])

    def test_tokenize_bytes_base64(self) -> None:
        self.assertEqual(tokenize("~~64'QUJD"), [BytesLit("QUJD", 64)])

    def test_tokenize_bytes_base32(self) -> None:
        self.assertEqual(tokenize("~~32'IFBEG==="), [BytesLit("IFBEG===", 32)])

    def test_tokenize_bytes_base16(self) -> None:
        self.assertEqual(tokenize("~~16'414243"), [BytesLit("414243", 16)])

    def test_tokenize_hole(self) -> None:
        self.assertEqual(tokenize("()"), [LeftParen(), RightParen()])

    def test_tokenize_hole_with_spaces(self) -> None:
        self.assertEqual(tokenize("( )"), [LeftParen(), RightParen()])

    def test_tokenize_parenthetical_expression(self) -> None:
        self.assertEqual(tokenize("(1+2)"), [LeftParen(), IntLit(1), Operator("+"), IntLit(2), RightParen()])

    def test_tokenize_pipe(self) -> None:
        self.assertEqual(
            tokenize("1 |> f . f = a -> a + 1"),
            [
                IntLit(1),
                Operator("|>"),
                Name("f"),
                Operator("."),
                Name("f"),
                Operator("="),
                Name("a"),
                Operator("->"),
                Name("a"),
                Operator("+"),
                IntLit(1),
            ],
        )

    def test_tokenize_reverse_pipe(self) -> None:
        self.assertEqual(
            tokenize("f <| 1 . f = a -> a + 1"),
            [
                Name("f"),
                Operator("<|"),
                IntLit(1),
                Operator("."),
                Name("f"),
                Operator("="),
                Name("a"),
                Operator("->"),
                Name("a"),
                Operator("+"),
                IntLit(1),
            ],
        )

    def test_tokenize_record_no_fields(self) -> None:
        self.assertEqual(
            tokenize("{ }"),
            [LeftBrace(), RightBrace()],
        )

    def test_tokenize_record_no_fields_no_spaces(self) -> None:
        self.assertEqual(
            tokenize("{}"),
            [LeftBrace(), RightBrace()],
        )

    def test_tokenize_record_one_field(self) -> None:
        self.assertEqual(
            tokenize("{ a = 4 }"),
            [LeftBrace(), Name("a"), Operator("="), IntLit(4), RightBrace()],
        )

    def test_tokenize_record_multiple_fields(self) -> None:
        self.assertEqual(
            tokenize('{ a = 4, b = "z" }'),
            [
                LeftBrace(),
                Name("a"),
                Operator("="),
                IntLit(4),
                Operator(","),
                Name("b"),
                Operator("="),
                StringLit("z"),
                RightBrace(),
            ],
        )

    def test_tokenize_record_access(self) -> None:
        self.assertEqual(
            tokenize("r@a"),
            [Name("r"), Operator("@"), Name("a")],
        )

    def test_tokenize_right_eval(self) -> None:
        self.assertEqual(tokenize("a!b"), [Name("a"), Operator("!"), Name("b")])

    def test_tokenize_match(self) -> None:
        self.assertEqual(
            tokenize("g = | 1 -> 2 | 2 -> 3"),
            [
                Name("g"),
                Operator("="),
                Operator("|"),
                IntLit(1),
                Operator("->"),
                IntLit(2),
                Operator("|"),
                IntLit(2),
                Operator("->"),
                IntLit(3),
            ],
        )

    def test_tokenize_compose(self) -> None:
        self.assertEqual(
            tokenize("f >> g"),
            [Name("f"), Operator(">>"), Name("g")],
        )

    def test_tokenize_compose_reverse(self) -> None:
        self.assertEqual(
            tokenize("f << g"),
            [Name("f"), Operator("<<"), Name("g")],
        )

    def test_first_lineno_is_one(self) -> None:
        l = Lexer("abc")
        self.assertEqual(l.lineno, 1)

    def test_first_colno_is_one(self) -> None:
        l = Lexer("abc")
        self.assertEqual(l.colno, 1)

    def test_first_line_is_empty(self) -> None:
        l = Lexer("abc")
        self.assertEqual(l.line, "")

    def test_read_char_increments_colno(self) -> None:
        l = Lexer("abc")
        l.read_char()
        self.assertEqual(l.colno, 2)
        self.assertEqual(l.lineno, 1)

    def test_read_newline_increments_lineno(self) -> None:
        l = Lexer("ab\nc")
        l.read_char()
        l.read_char()
        l.read_char()
        self.assertEqual(l.lineno, 2)
        self.assertEqual(l.colno, 1)

    def test_read_char_appends_to_line(self) -> None:
        l = Lexer("ab\nc")
        l.read_char()
        l.read_char()
        self.assertEqual(l.line, "ab")
        l.read_char()
        self.assertEqual(l.line, "")

    def test_read_one_sets_lineno(self) -> None:
        l = Lexer("a b \n c d")
        a = l.read_one()
        b = l.read_one()
        c = l.read_one()
        d = l.read_one()
        self.assertEqual(a.lineno, 1)
        self.assertEqual(b.lineno, 1)
        self.assertEqual(c.lineno, 2)
        self.assertEqual(d.lineno, 2)

    def test_tokenize_list_with_only_spread(self) -> None:
        self.assertEqual(tokenize("[ ... ]"), [LeftBracket(), Operator("..."), RightBracket()])

    def test_tokenize_list_with_spread(self) -> None:
        self.assertEqual(
            tokenize("[ 1 , ... ]"),
            [
                LeftBracket(),
                IntLit(1),
                Operator(","),
                Operator("..."),
                RightBracket(),
            ],
        )

    def test_tokenize_list_with_spread_no_spaces(self) -> None:
        self.assertEqual(
            tokenize("[ 1,... ]"),
            [
                LeftBracket(),
                IntLit(1),
                Operator(","),
                Operator("..."),
                RightBracket(),
            ],
        )

    def test_tokenize_list_with_named_spread(self) -> None:
        self.assertEqual(
            tokenize("[1,...rest]"),
            [
                LeftBracket(),
                IntLit(1),
                Operator(","),
                Operator("..."),
                Name("rest"),
                RightBracket(),
            ],
        )

    def test_tokenize_record_with_only_spread(self) -> None:
        self.assertEqual(
            tokenize("{ ... }"),
            [
                LeftBrace(),
                Operator("..."),
                RightBrace(),
            ],
        )

    def test_tokenize_record_with_spread(self) -> None:
        self.assertEqual(
            tokenize("{ x = 1, ...}"),
            [
                LeftBrace(),
                Name("x"),
                Operator("="),
                IntLit(1),
                Operator(","),
                Operator("..."),
                RightBrace(),
            ],
        )

    def test_tokenize_record_with_spread_no_spaces(self) -> None:
        self.assertEqual(
            tokenize("{x=1,...}"),
            [
                LeftBrace(),
                Name("x"),
                Operator("="),
                IntLit(1),
                Operator(","),
                Operator("..."),
                RightBrace(),
            ],
        )

    def test_tokenize_symbol_with_space(self) -> None:
        self.assertEqual(tokenize("# abc"), [SymbolToken("abc")])

    def test_tokenize_symbol_with_no_space(self) -> None:
        self.assertEqual(tokenize("#abc"), [SymbolToken("abc")])

    def test_tokenize_symbol_non_name_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "expected name"):
            tokenize("#1")

    def test_tokenize_symbol_eof_raises_unexpected_eof_error(self) -> None:
        with self.assertRaisesRegex(UnexpectedEOFError, "while reading symbol"):
            tokenize("#")


class ParserTests(unittest.TestCase):
    def test_parse_with_empty_tokens_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedEOFError) as ctx:
            parse([])
        self.assertEqual(ctx.exception.args[0], "unexpected end of input")

    def test_parse_digit_returns_int(self) -> None:
        self.assertEqual(parse([IntLit(1)]), Int(1))

    def test_parse_digits_returns_int(self) -> None:
        self.assertEqual(parse([IntLit(123)]), Int(123))

    def test_parse_negative_int_returns_binary_sub_int(self) -> None:
        self.assertEqual(parse([Operator("-"), IntLit(123)]), Binop(BinopKind.SUB, Int(0), Int(123)))

    def test_parse_negative_var_returns_binary_sub_int(self) -> None:
        self.assertEqual(parse([Operator("-"), Name("x")]), Binop(BinopKind.SUB, Int(0), Var("x")))

    def test_parse_negative_int_binds_tighter_than_plus(self) -> None:
        self.assertEqual(
            parse([Operator("-"), Name("l"), Operator("+"), Name("r")]),
            Binop(BinopKind.ADD, Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_negative_int_binds_tighter_than_mul(self) -> None:
        self.assertEqual(
            parse([Operator("-"), Name("l"), Operator("*"), Name("r")]),
            Binop(BinopKind.MUL, Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_negative_int_binds_tighter_than_index(self) -> None:
        self.assertEqual(
            parse([Operator("-"), Name("l"), Operator("@"), Name("r")]),
            Access(Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_negative_int_binds_tighter_than_apply(self) -> None:
        self.assertEqual(
            parse([Operator("-"), Name("l"), Name("r")]),
            Apply(Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_decimal_returns_float(self) -> None:
        self.assertEqual(parse([FloatLit(3.14)]), Float(3.14))

    def test_parse_negative_float_returns_binary_sub_float(self) -> None:
        self.assertEqual(parse([Operator("-"), FloatLit(3.14)]), Binop(BinopKind.SUB, Int(0), Float(3.14)))

    def test_parse_var_returns_var(self) -> None:
        self.assertEqual(parse([Name("abc_123")]), Var("abc_123"))

    def test_parse_sha_var_returns_var(self) -> None:
        self.assertEqual(parse([Name("$sha1'abc")]), Var("$sha1'abc"))

    def test_parse_sha_var_without_quote_returns_var(self) -> None:
        self.assertEqual(parse([Name("$sha1abc")]), Var("$sha1abc"))

    def test_parse_dollar_returns_var(self) -> None:
        self.assertEqual(parse([Name("$")]), Var("$"))

    def test_parse_dollar_dollar_returns_var(self) -> None:
        self.assertEqual(parse([Name("$$")]), Var("$$"))

    @unittest.skip("TODO: make this fail to parse")
    def test_parse_sha_var_without_dollar_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token"):
            parse([Name("sha1'abc")])

    def test_parse_dollar_dollar_var_returns_var(self) -> None:
        self.assertEqual(parse([Name("$$bills")]), Var("$$bills"))

    def test_parse_bytes_returns_bytes(self) -> None:
        self.assertEqual(parse([BytesLit("QUJD", 64)]), Bytes(b"ABC"))

    def test_parse_binary_add_returns_binop(self) -> None:
        self.assertEqual(parse([IntLit(1), Operator("+"), IntLit(2)]), Binop(BinopKind.ADD, Int(1), Int(2)))

    def test_parse_binary_sub_returns_binop(self) -> None:
        self.assertEqual(parse([IntLit(1), Operator("-"), IntLit(2)]), Binop(BinopKind.SUB, Int(1), Int(2)))

    def test_parse_binary_add_right_returns_binop(self) -> None:
        self.assertEqual(
            parse([IntLit(1), Operator("+"), IntLit(2), Operator("+"), IntLit(3)]),
            Binop(BinopKind.ADD, Int(1), Binop(BinopKind.ADD, Int(2), Int(3))),
        )

    def test_mul_binds_tighter_than_add_right(self) -> None:
        self.assertEqual(
            parse([IntLit(1), Operator("+"), IntLit(2), Operator("*"), IntLit(3)]),
            Binop(BinopKind.ADD, Int(1), Binop(BinopKind.MUL, Int(2), Int(3))),
        )

    def test_mul_binds_tighter_than_add_left(self) -> None:
        self.assertEqual(
            parse([IntLit(1), Operator("*"), IntLit(2), Operator("+"), IntLit(3)]),
            Binop(BinopKind.ADD, Binop(BinopKind.MUL, Int(1), Int(2)), Int(3)),
        )

    def test_exp_binds_tighter_than_mul_right(self) -> None:
        self.assertEqual(
            parse([IntLit(5), Operator("*"), IntLit(2), Operator("^"), IntLit(3)]),
            Binop(BinopKind.MUL, Int(5), Binop(BinopKind.EXP, Int(2), Int(3))),
        )

    def test_list_access_binds_tighter_than_append(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("+<"), Name("ls"), Operator("@"), IntLit(0)]),
            Binop(BinopKind.LIST_APPEND, Var("a"), Access(Var("ls"), Int(0))),
        )

    def test_parse_binary_str_concat_returns_binop(self) -> None:
        self.assertEqual(
            parse([StringLit("abc"), Operator("++"), StringLit("def")]),
            Binop(BinopKind.STRING_CONCAT, String("abc"), String("def")),
        )

    def test_parse_binary_list_cons_returns_binop(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator(">+"), Name("b")]),
            Binop(BinopKind.LIST_CONS, Var("a"), Var("b")),
        )

    def test_parse_binary_list_append_returns_binop(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("+<"), Name("b")]),
            Binop(BinopKind.LIST_APPEND, Var("a"), Var("b")),
        )

    def test_parse_binary_op_returns_binop(self) -> None:
        ops = ["+", "-", "*", "/", "^", "%", "==", "/=", "<", ">", "<=", ">=", "&&", "||", "++", ">+", "+<"]
        for op in ops:
            with self.subTest(op=op):
                kind = BinopKind.from_str(op)
                self.assertEqual(parse([Name("a"), Operator(op), Name("b")]), Binop(kind, Var("a"), Var("b")))

    def test_parse_empty_list(self) -> None:
        self.assertEqual(
            parse([LeftBracket(), RightBracket()]),
            List([]),
        )

    def test_parse_list_of_ints_returns_list(self) -> None:
        self.assertEqual(
            parse([LeftBracket(), IntLit(1), Operator(","), IntLit(2), RightBracket()]),
            List([Int(1), Int(2)]),
        )

    def test_parse_list_with_only_comma_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token Operator(lineno=-1, value=',')")):
            parse([LeftBracket(), Operator(","), RightBracket()])

    def test_parse_list_with_two_commas_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token Operator(lineno=-1, value=',')")):
            parse([LeftBracket(), Operator(","), Operator(","), RightBracket()])

    def test_parse_list_with_trailing_comma_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token RightBracket(lineno=-1)")):
            parse([LeftBracket(), IntLit(1), Operator(","), RightBracket()])

    def test_parse_assign(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("="), IntLit(1)]),
            Assign(Var("a"), Int(1)),
        )

    def test_parse_function_one_arg_returns_function(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("->"), Name("a"), Operator("+"), IntLit(1)]),
            Function(Var("a"), Binop(BinopKind.ADD, Var("a"), Int(1))),
        )

    def test_parse_function_two_args_returns_functions(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("->"), Name("b"), Operator("->"), Name("a"), Operator("+"), Name("b")]),
            Function(Var("a"), Function(Var("b"), Binop(BinopKind.ADD, Var("a"), Var("b")))),
        )

    def test_parse_assign_function(self) -> None:
        self.assertEqual(
            parse([Name("id"), Operator("="), Name("x"), Operator("->"), Name("x")]),
            Assign(Var("id"), Function(Var("x"), Var("x"))),
        )

    def test_parse_function_application_one_arg(self) -> None:
        self.assertEqual(parse([Name("f"), Name("a")]), Apply(Var("f"), Var("a")))

    def test_parse_function_application_two_args(self) -> None:
        self.assertEqual(parse([Name("f"), Name("a"), Name("b")]), Apply(Apply(Var("f"), Var("a")), Var("b")))

    def test_parse_where(self) -> None:
        self.assertEqual(parse([Name("a"), Operator("."), Name("b")]), Where(Var("a"), Var("b")))

    def test_parse_nested_where(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("."), Name("b"), Operator("."), Name("c")]),
            Where(Where(Var("a"), Var("b")), Var("c")),
        )

    def test_parse_assert(self) -> None:
        self.assertEqual(parse([Name("a"), Operator("?"), Name("b")]), Assert(Var("a"), Var("b")))

    def test_parse_nested_assert(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("?"), Name("b"), Operator("?"), Name("c")]),
            Assert(Assert(Var("a"), Var("b")), Var("c")),
        )

    def test_parse_mixed_assert_where(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("?"), Name("b"), Operator("."), Name("c")]),
            Where(Assert(Var("a"), Var("b")), Var("c")),
        )

    def test_parse_hastype(self) -> None:
        self.assertEqual(parse([Name("a"), Operator(":"), Name("b")]), Binop(BinopKind.HASTYPE, Var("a"), Var("b")))

    def test_parse_hole(self) -> None:
        self.assertEqual(parse([LeftParen(), RightParen()]), Hole())

    def test_parse_parenthesized_expression(self) -> None:
        self.assertEqual(
            parse([LeftParen(), IntLit(1), Operator("+"), IntLit(2), RightParen()]),
            Binop(BinopKind.ADD, Int(1), Int(2)),
        )

    def test_parse_parenthesized_add_mul(self) -> None:
        self.assertEqual(
            parse([LeftParen(), IntLit(1), Operator("+"), IntLit(2), RightParen(), Operator("*"), IntLit(3)]),
            Binop(BinopKind.MUL, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3)),
        )

    def test_parse_pipe(self) -> None:
        self.assertEqual(
            parse([IntLit(1), Operator("|>"), Name("f")]),
            Apply(Var("f"), Int(1)),
        )

    def test_parse_nested_pipe(self) -> None:
        self.assertEqual(
            parse([IntLit(1), Operator("|>"), Name("f"), Operator("|>"), Name("g")]),
            Apply(Var("g"), Apply(Var("f"), Int(1))),
        )

    def test_parse_reverse_pipe(self) -> None:
        self.assertEqual(
            parse([Name("f"), Operator("<|"), IntLit(1)]),
            Apply(Var("f"), Int(1)),
        )

    def test_parse_nested_reverse_pipe(self) -> None:
        self.assertEqual(
            parse([Name("g"), Operator("<|"), Name("f"), Operator("<|"), IntLit(1)]),
            Apply(Var("g"), Apply(Var("f"), Int(1))),
        )

    def test_parse_empty_record(self) -> None:
        self.assertEqual(parse([LeftBrace(), RightBrace()]), Record({}))

    def test_parse_record_single_field(self) -> None:
        self.assertEqual(parse([LeftBrace(), Name("a"), Operator("="), IntLit(4), RightBrace()]), Record({"a": Int(4)}))

    def test_parse_record_with_expression(self) -> None:
        self.assertEqual(
            parse([LeftBrace(), Name("a"), Operator("="), IntLit(1), Operator("+"), IntLit(2), RightBrace()]),
            Record({"a": Binop(BinopKind.ADD, Int(1), Int(2))}),
        )

    def test_parse_record_multiple_fields(self) -> None:
        self.assertEqual(
            parse(
                [
                    LeftBrace(),
                    Name("a"),
                    Operator("="),
                    IntLit(4),
                    Operator(","),
                    Name("b"),
                    Operator("="),
                    StringLit("z"),
                    RightBrace(),
                ]
            ),
            Record({"a": Int(4), "b": String("z")}),
        )

    def test_non_variable_in_assignment_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse([IntLit(3), Operator("="), IntLit(4)])
        self.assertEqual(ctx.exception.args[0], "expected variable in assignment Int(value=3)")

    def test_non_assign_in_record_constructor_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse([LeftBrace(), IntLit(1), Operator(","), IntLit(2), RightBrace()])
        self.assertEqual(ctx.exception.args[0], "failed to parse variable assignment in record constructor")

    def test_parse_right_eval_returns_binop(self) -> None:
        self.assertEqual(parse([Name("a"), Operator("!"), Name("b")]), Binop(BinopKind.RIGHT_EVAL, Var("a"), Var("b")))

    def test_parse_right_eval_with_defs_returns_binop(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("!"), Name("b"), Operator("."), Name("c")]),
            Binop(BinopKind.RIGHT_EVAL, Var("a"), Where(Var("b"), Var("c"))),
        )

    def test_parse_match_no_cases_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse([Operator("|")])
        self.assertEqual(ctx.exception.args[0], "unexpected end of input")

    def test_parse_match_one_case(self) -> None:
        self.assertEqual(
            parse([Operator("|"), IntLit(1), Operator("->"), IntLit(2)]),
            MatchFunction([MatchCase(Int(1), Int(2))]),
        )

    def test_parse_match_two_cases(self) -> None:
        self.assertEqual(
            parse(
                [
                    Operator("|"),
                    IntLit(1),
                    Operator("->"),
                    IntLit(2),
                    Operator("|"),
                    IntLit(2),
                    Operator("->"),
                    IntLit(3),
                ]
            ),
            MatchFunction(
                [
                    MatchCase(Int(1), Int(2)),
                    MatchCase(Int(2), Int(3)),
                ]
            ),
        )

    def test_parse_compose(self) -> None:
        self.assertEqual(parse([Name("f"), Operator(">>"), Name("g")]), Compose(Var("f"), Var("g")))

    def test_parse_compose_reverse(self) -> None:
        self.assertEqual(parse([Name("f"), Operator("<<"), Name("g")]), Compose(Var("g"), Var("f")))

    def test_parse_double_compose(self) -> None:
        self.assertEqual(
            parse([Name("f"), Operator("<<"), Name("g"), Operator("<<"), Name("h")]),
            Compose(Compose(Var("h"), Var("g")), Var("f")),
        )

    def test_boolean_and_binds_tighter_than_or(self) -> None:
        self.assertEqual(
            parse([Name("x"), Operator("||"), Name("y"), Operator("&&"), Name("z")]),
            Binop(BinopKind.BOOL_OR, Var("x"), Binop(BinopKind.BOOL_AND, Var("y"), Var("z"))),
        )

    def test_parse_list_spread(self) -> None:
        self.assertEqual(
            parse([LeftBracket(), IntLit(1), Operator(","), Operator("..."), RightBracket()]),
            List([Int(1), Spread()]),
        )

    @unittest.skip("TODO(max): Raise if ...x is used with non-name")
    def test_parse_list_with_non_name_expr_after_spread_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token IntLit(lineno=-1, value=1)")):
            parse([LeftBracket(), IntLit(1), Operator(","), Operator("..."), IntLit(2), RightBracket()])

    def test_parse_list_with_named_spread(self) -> None:
        self.assertEqual(
            parse(
                [
                    LeftBracket(),
                    IntLit(1),
                    Operator(","),
                    Operator("..."),
                    Name("rest"),
                    RightBracket(),
                ]
            ),
            List([Int(1), Spread("rest")]),
        )

    def test_parse_list_spread_beginning_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of list match")):
            parse([LeftBracket(), Operator("..."), Operator(","), IntLit(1), RightBracket()])

    def test_parse_list_named_spread_beginning_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of list match")):
            parse([LeftBracket(), Operator("..."), Name("rest"), Operator(","), IntLit(1), RightBracket()])

    def test_parse_list_spread_middle_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of list match")):
            parse([LeftBracket(), IntLit(1), Operator(","), Operator("..."), Operator(","), IntLit(1), RightBracket()])

    def test_parse_list_named_spread_middle_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of list match")):
            parse(
                [
                    LeftBracket(),
                    IntLit(1),
                    Operator(","),
                    Operator("..."),
                    Name("rest"),
                    Operator(","),
                    IntLit(1),
                    RightBracket(),
                ]
            )

    def test_parse_record_spread(self) -> None:
        self.assertEqual(
            parse([LeftBrace(), Name("x"), Operator("="), IntLit(1), Operator(","), Operator("..."), RightBrace()]),
            Record({"x": Int(1), "...": Spread()}),
        )

    def test_parse_record_spread_beginning_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of record match")):
            parse([LeftBrace(), Operator("..."), Operator(","), Name("x"), Operator("="), IntLit(1), RightBrace()])

    def test_parse_record_spread_middle_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of record match")):
            parse(
                [
                    LeftBrace(),
                    Name("x"),
                    Operator("="),
                    IntLit(1),
                    Operator(","),
                    Operator("..."),
                    Operator(","),
                    Name("y"),
                    Operator("="),
                    IntLit(2),
                    RightBrace(),
                ]
            )

    def test_parse_record_with_only_comma_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token Operator(lineno=-1, value=',')")):
            parse([LeftBrace(), Operator(","), RightBrace()])

    def test_parse_record_with_two_commas_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token Operator(lineno=-1, value=',')")):
            parse([LeftBrace(), Operator(","), Operator(","), RightBrace()])

    def test_parse_record_with_trailing_comma_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token RightBrace(lineno=-1)")):
            parse([LeftBrace(), Name("x"), Operator("="), IntLit(1), Operator(","), RightBrace()])

    def test_parse_symbol_returns_symbol(self) -> None:
        self.assertEqual(parse([SymbolToken("abc")]), Symbol("abc"))


class MatchTests(unittest.TestCase):
    def test_match_with_equal_ints_returns_empty_dict(self) -> None:
        self.assertEqual(match(Int(1), pattern=Int(1)), {})

    def test_match_with_inequal_ints_returns_none(self) -> None:
        self.assertEqual(match(Int(2), pattern=Int(1)), None)

    def test_match_int_with_non_int_returns_none(self) -> None:
        self.assertEqual(match(String("abc"), pattern=Int(1)), None)

    def test_match_with_equal_floats_raises_match_error(self) -> None:
        with self.assertRaisesRegex(MatchError, re.escape("pattern matching is not supported for Floats")):
            match(Float(1), pattern=Float(1))

    def test_match_with_inequal_floats_raises_match_error(self) -> None:
        with self.assertRaisesRegex(MatchError, re.escape("pattern matching is not supported for Floats")):
            match(Float(2), pattern=Float(1))

    def test_match_float_with_non_float_raises_match_error(self) -> None:
        with self.assertRaisesRegex(MatchError, re.escape("pattern matching is not supported for Floats")):
            match(String("abc"), pattern=Float(1))

    def test_match_with_equal_strings_returns_empty_dict(self) -> None:
        self.assertEqual(match(String("a"), pattern=String("a")), {})

    def test_match_with_inequal_strings_returns_none(self) -> None:
        self.assertEqual(match(String("b"), pattern=String("a")), None)

    def test_match_string_with_non_string_returns_none(self) -> None:
        self.assertEqual(match(Int(1), pattern=String("abc")), None)

    def test_match_var_returns_dict_with_var_name(self) -> None:
        self.assertEqual(match(String("abc"), pattern=Var("a")), {"a": String("abc")})

    def test_match_record_with_non_record_returns_none(self) -> None:
        self.assertEqual(
            match(
                Int(2),
                pattern=Record({"x": Var("x"), "y": Var("y")}),
            ),
            None,
        )

    def test_match_record_with_more_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Var("x"), "y": Var("y"), "z": Var("z")}),
            ),
            None,
        )

    def test_match_record_with_fewer_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Var("x")}),
            ),
            None,
        )

    def test_match_record_with_vars_returns_dict_with_keys(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Var("x"), "y": Var("y")}),
            ),
            {"x": Int(1), "y": Int(2)},
        )

    def test_match_record_with_matching_const_returns_dict_with_other_keys(self) -> None:
        # TODO(max): Should this be the case? I feel like we should return all
        # the keys.
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Int(1), "y": Var("y")}),
            ),
            {"y": Int(2)},
        )

    def test_match_record_with_non_matching_const_returns_none(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Int(3), "y": Var("y")}),
            ),
            None,
        )

    def test_match_list_with_non_list_returns_none(self) -> None:
        self.assertEqual(
            match(
                Int(2),
                pattern=List([Var("x"), Var("y")]),
            ),
            None,
        )

    def test_match_list_with_more_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("x"), Var("y"), Var("z")]),
            ),
            None,
        )

    def test_match_list_with_fewer_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("x")]),
            ),
            None,
        )

    def test_match_list_with_vars_returns_dict_with_keys(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("x"), Var("y")]),
            ),
            {"x": Int(1), "y": Int(2)},
        )

    def test_match_list_with_matching_const_returns_dict_with_other_keys(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Int(1), Var("y")]),
            ),
            {"y": Int(2)},
        )

    def test_match_list_with_non_matching_const_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Int(3), Var("y")]),
            ),
            None,
        )

    def test_parse_right_pipe(self) -> None:
        text = "3 + 4 |> $$quote"
        ast = parse(tokenize(text))
        self.assertEqual(ast, Apply(Var("$$quote"), Binop(BinopKind.ADD, Int(3), Int(4))))

    def test_parse_left_pipe(self) -> None:
        text = "$$quote <| 3 + 4"
        ast = parse(tokenize(text))
        self.assertEqual(ast, Apply(Var("$$quote"), Binop(BinopKind.ADD, Int(3), Int(4))))

    def test_parse_match_with_left_apply(self) -> None:
        text = """| a -> b <| c
                  | d -> e"""
        tokens = tokenize(text)
        self.assertEqual(
            tokens,
            [
                Operator("|"),
                Name("a"),
                Operator("->"),
                Name("b"),
                Operator("<|"),
                Name("c"),
                Operator("|"),
                Name("d"),
                Operator("->"),
                Name("e"),
            ],
        )
        ast = parse(tokens)
        self.assertEqual(
            ast, MatchFunction([MatchCase(Var("a"), Apply(Var("b"), Var("c"))), MatchCase(Var("d"), Var("e"))])
        )

    def test_parse_match_with_right_apply(self) -> None:
        text = """
| 1 -> 19
| a -> a |> (x -> x + 1)
"""
        tokens = tokenize(text)
        ast = parse(tokens)
        self.assertEqual(
            ast,
            MatchFunction(
                [
                    MatchCase(Int(1), Int(19)),
                    MatchCase(
                        Var("a"),
                        Apply(
                            Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(1))),
                            Var("a"),
                        ),
                    ),
                ]
            ),
        )

    def test_match_list_with_spread_returns_empty_dict(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2), Int(3), Int(4), Int(5)]),
                pattern=List([Int(1), Spread()]),
            ),
            {},
        )

    def test_match_list_with_named_spread_returns_name_bound_to_rest(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2), Int(3), Int(4)]),
                pattern=List([Var("a"), Int(2), Spread("rest")]),
            ),
            {"a": Int(1), "rest": List([Int(3), Int(4)])},
        )

    def test_match_list_with_named_spread_returns_name_bound_to_empty_rest(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("a"), Int(2), Spread("rest")]),
            ),
            {"a": Int(1), "rest": List([])},
        )

    def test_match_list_with_mismatched_spread_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2), Int(3), Int(4), Int(5)]),
                pattern=List([Int(1), Int(6), Spread()]),
            ),
            None,
        )

    def test_match_record_with_constant_and_spread_returns_empty_dict(self) -> None:
        self.assertEqual(
            match(
                Record({"a": Int(1), "b": Int(2), "c": Int(3)}),
                pattern=Record({"a": Int(1), "...": Spread()}),
            ),
            {},
        )

    def test_match_record_with_var_and_spread_returns_match(self) -> None:
        self.assertEqual(
            match(
                Record({"a": Int(1), "b": Int(2), "c": Int(3)}),
                pattern=Record({"a": Var("x"), "...": Spread()}),
            ),
            {"x": Int(1)},
        )

    def test_match_record_with_mismatched_spread_returns_none(self) -> None:
        self.assertEqual(
            match(
                Record({"a": Int(1), "b": Int(2), "c": Int(3)}),
                pattern=Record({"d": Var("x"), "...": Spread()}),
            ),
            None,
        )

    def test_match_symbol_with_equal_symbol_returns_empty_dict(self) -> None:
        self.assertEqual(match(Symbol("abc"), pattern=Symbol("abc")), {})

    def test_match_symbol_with_inequal_symbol_returns_none(self) -> None:
        self.assertEqual(match(Symbol("def"), pattern=Symbol("abc")), None)

    def test_match_symbol_with_different_type_returns_none(self) -> None:
        self.assertEqual(match(Int(123), pattern=Symbol("abc")), None)


class EvalTests(unittest.TestCase):
    def test_eval_int_returns_int(self) -> None:
        exp = Int(5)
        self.assertEqual(eval_exp({}, exp), Int(5))

    def test_eval_float_returns_float(self) -> None:
        exp = Float(3.14)
        self.assertEqual(eval_exp({}, exp), Float(3.14))

    def test_eval_str_returns_str(self) -> None:
        exp = String("xyz")
        self.assertEqual(eval_exp({}, exp), String("xyz"))

    def test_eval_bytes_returns_bytes(self) -> None:
        exp = Bytes(b"xyz")
        self.assertEqual(eval_exp({}, exp), Bytes(b"xyz"))

    def test_eval_with_non_existent_var_raises_name_error(self) -> None:
        exp = Var("no")
        with self.assertRaises(NameError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "name 'no' is not defined")

    def test_eval_with_bound_var_returns_value(self) -> None:
        exp = Var("yes")
        env = {"yes": Int(123)}
        self.assertEqual(eval_exp(env, exp), Int(123))

    def test_eval_with_binop_add_returns_sum(self) -> None:
        exp = Binop(BinopKind.ADD, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(3))

    def test_eval_with_nested_binop(self) -> None:
        exp = Binop(BinopKind.ADD, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(6))

    def test_eval_with_binop_add_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.ADD, Int(1), String("hello"))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected Int or Float, got String")

    def test_eval_with_binop_sub(self) -> None:
        exp = Binop(BinopKind.SUB, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(-1))

    def test_eval_with_binop_mul(self) -> None:
        exp = Binop(BinopKind.MUL, Int(2), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(6))

    def test_eval_with_binop_div(self) -> None:
        exp = Binop(BinopKind.DIV, Int(3), Int(10))
        self.assertEqual(eval_exp({}, exp), Float(0.3))

    def test_eval_with_binop_floor_div(self) -> None:
        exp = Binop(BinopKind.FLOOR_DIV, Int(2), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(0))

    def test_eval_with_binop_exp(self) -> None:
        exp = Binop(BinopKind.EXP, Int(2), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(8))

    def test_eval_with_binop_mod(self) -> None:
        exp = Binop(BinopKind.MOD, Int(10), Int(4))
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_eval_with_binop_equal_with_equal_returns_true(self) -> None:
        exp = Binop(BinopKind.EQUAL, Int(1), Int(1))
        self.assertEqual(eval_exp({}, exp), Symbol("true"))

    def test_eval_with_binop_equal_with_inequal_returns_false(self) -> None:
        exp = Binop(BinopKind.EQUAL, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Symbol("false"))

    def test_eval_with_binop_not_equal_with_equal_returns_false(self) -> None:
        exp = Binop(BinopKind.NOT_EQUAL, Int(1), Int(1))
        self.assertEqual(eval_exp({}, exp), Symbol("false"))

    def test_eval_with_binop_not_equal_with_inequal_returns_true(self) -> None:
        exp = Binop(BinopKind.NOT_EQUAL, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Symbol("true"))

    def test_eval_with_binop_concat_with_strings_returns_string(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, String("hello"), String(" world"))
        self.assertEqual(eval_exp({}, exp), String("hello world"))

    def test_eval_with_binop_concat_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, Int(123), String(" world"))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected String, got Int")

    def test_eval_with_binop_concat_with_string_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, String(" world"), Int(123))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected String, got Int")

    def test_eval_with_binop_cons_with_int_list_returns_list(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, Int(1), List([Int(2), Int(3)]))
        self.assertEqual(eval_exp({}, exp), List([Int(1), Int(2), Int(3)]))

    def test_eval_with_binop_cons_with_list_list_returns_nested_list(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, List([]), List([]))
        self.assertEqual(eval_exp({}, exp), List([List([])]))

    def test_eval_with_binop_cons_with_list_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, List([]), Int(123))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected List, got Int")

    def test_eval_with_list_append(self) -> None:
        exp = Binop(BinopKind.LIST_APPEND, List([Int(1), Int(2)]), Int(3))
        self.assertEqual(eval_exp({}, exp), List([Int(1), Int(2), Int(3)]))

    def test_eval_with_list_evaluates_elements(self) -> None:
        exp = List(
            [
                Binop(BinopKind.ADD, Int(1), Int(2)),
                Binop(BinopKind.ADD, Int(3), Int(4)),
            ]
        )
        self.assertEqual(eval_exp({}, exp), List([Int(3), Int(7)]))

    def test_eval_with_function_returns_closure_with_improved_env(self) -> None:
        exp = Function(Var("x"), Var("x"))
        self.assertEqual(eval_exp({"a": Int(1), "b": Int(2)}, exp), Closure({}, exp))

    def test_eval_with_match_function_returns_closure_with_improved_env(self) -> None:
        exp = MatchFunction([])
        self.assertEqual(eval_exp({"a": Int(1), "b": Int(2)}, exp), Closure({}, exp))

    def test_eval_assign_returns_env_object(self) -> None:
        exp = Assign(Var("a"), Int(1))
        env: Env = {}
        result = eval_exp(env, exp)
        self.assertEqual(result, EnvObject({"a": Int(1)}))

    def test_eval_assign_function_returns_closure_without_function_in_env(self) -> None:
        exp = Assign(Var("a"), Function(Var("x"), Var("x")))
        result = eval_exp({}, exp)
        assert isinstance(result, EnvObject)
        closure = result.env["a"]
        self.assertIsInstance(closure, Closure)
        self.assertEqual(closure, Closure({}, Function(Var("x"), Var("x"))))

    def test_eval_assign_function_returns_closure_with_function_in_env(self) -> None:
        exp = Assign(Var("a"), Function(Var("x"), Var("a")))
        result = eval_exp({}, exp)
        assert isinstance(result, EnvObject)
        closure = result.env["a"]
        self.assertIsInstance(closure, Closure)
        self.assertEqual(closure, Closure({"a": closure}, Function(Var("x"), Var("a"))))

    def test_eval_assign_does_not_modify_env(self) -> None:
        exp = Assign(Var("a"), Int(1))
        env: Env = {}
        eval_exp(env, exp)
        self.assertEqual(env, {})

    def test_eval_where_evaluates_in_order(self) -> None:
        exp = Where(Binop(BinopKind.ADD, Var("a"), Int(2)), Assign(Var("a"), Int(1)))
        env: Env = {}
        self.assertEqual(eval_exp(env, exp), Int(3))
        self.assertEqual(env, {})

    def test_eval_nested_where(self) -> None:
        exp = Where(
            Where(
                Binop(BinopKind.ADD, Var("a"), Var("b")),
                Assign(Var("a"), Int(1)),
            ),
            Assign(Var("b"), Int(2)),
        )
        env: Env = {}
        self.assertEqual(eval_exp(env, exp), Int(3))
        self.assertEqual(env, {})

    def test_eval_assert_with_truthy_cond_returns_value(self) -> None:
        exp = Assert(Int(123), Symbol("true"))
        self.assertEqual(eval_exp({}, exp), Int(123))

    def test_eval_assert_with_falsey_cond_raises_assertion_error(self) -> None:
        exp = Assert(Int(123), Symbol("false"))
        with self.assertRaisesRegex(AssertionError, re.escape("condition #false failed")):
            eval_exp({}, exp)

    def test_eval_nested_assert(self) -> None:
        exp = Assert(Assert(Int(123), Symbol("true")), Symbol("true"))
        self.assertEqual(eval_exp({}, exp), Int(123))

    def test_eval_hole(self) -> None:
        exp = Hole()
        self.assertEqual(eval_exp({}, exp), Hole())

    def test_eval_function_application_one_arg(self) -> None:
        exp = Apply(Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(1))), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(3))

    def test_eval_function_application_two_args(self) -> None:
        exp = Apply(
            Apply(Function(Var("a"), Function(Var("b"), Binop(BinopKind.ADD, Var("a"), Var("b")))), Int(3)),
            Int(2),
        )
        self.assertEqual(eval_exp({}, exp), Int(5))

    def test_eval_function_returns_closure_with_captured_env(self) -> None:
        exp = Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Var("y")))
        res = eval_exp({"y": Int(5)}, exp)
        self.assertIsInstance(res, Closure)
        assert isinstance(res, Closure)  # for mypy
        self.assertEqual(res.env, {"y": Int(5)})

    def test_eval_function_capture_env(self) -> None:
        exp = Apply(Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Var("y"))), Int(2))
        self.assertEqual(eval_exp({"y": Int(5)}, exp), Int(7))

    def test_eval_non_function_raises_type_error(self) -> None:
        exp = Apply(Int(3), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("attempted to apply a non-closure of type Int")):
            eval_exp({}, exp)

    def test_eval_access_from_invalid_object_raises_type_error(self) -> None:
        exp = Access(Int(4), String("x"))
        with self.assertRaisesRegex(TypeError, re.escape("attempted to access from type Int")):
            eval_exp({}, exp)

    def test_eval_record_evaluates_value_expressions(self) -> None:
        exp = Record({"a": Binop(BinopKind.ADD, Int(1), Int(2))})
        self.assertEqual(eval_exp({}, exp), Record({"a": Int(3)}))

    def test_eval_record_access_with_invalid_accessor_raises_type_error(self) -> None:
        exp = Access(Record({"a": Int(4)}), Int(0))
        with self.assertRaisesRegex(
            TypeError, re.escape("cannot access record field using Int, expected a field name")
        ):
            eval_exp({}, exp)

    def test_eval_record_access_with_unknown_accessor_raises_name_error(self) -> None:
        exp = Access(Record({"a": Int(4)}), Var("b"))
        with self.assertRaisesRegex(NameError, re.escape("no assignment to b found in record")):
            eval_exp({}, exp)

    def test_eval_record_access(self) -> None:
        exp = Access(Record({"a": Int(4)}), Var("a"))
        self.assertEqual(eval_exp({}, exp), Int(4))

    def test_eval_list_access_with_invalid_accessor_raises_type_error(self) -> None:
        exp = Access(List([Int(4)]), String("hello"))
        with self.assertRaisesRegex(TypeError, re.escape("cannot index into list using type String, expected integer")):
            eval_exp({}, exp)

    def test_eval_list_access_with_out_of_bounds_accessor_raises_value_error(self) -> None:
        exp = Access(List([Int(1), Int(2), Int(3)]), Int(4))
        with self.assertRaisesRegex(ValueError, re.escape("index 4 out of bounds for list")):
            eval_exp({}, exp)

    def test_eval_list_access(self) -> None:
        exp = Access(List([String("a"), String("b"), String("c")]), Int(2))
        self.assertEqual(eval_exp({}, exp), String("c"))

    def test_right_eval_evaluates_right_hand_side(self) -> None:
        exp = Binop(BinopKind.RIGHT_EVAL, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_match_no_cases_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([]), Int(1))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval_exp({}, exp)

    def test_match_int_with_equal_int_matches(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=Int(1), body=Int(2))]), Int(1))
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_match_int_with_inequal_int_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=Int(1), body=Int(2))]), Int(3))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval_exp({}, exp)

    def test_match_string_with_equal_string_matches(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=String("a"), body=String("b"))]), String("a"))
        self.assertEqual(eval_exp({}, exp), String("b"))

    def test_match_string_with_inequal_string_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=String("a"), body=String("b"))]), String("c"))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval_exp({}, exp)

    def test_match_falls_through_to_next(self) -> None:
        exp = Apply(
            MatchFunction([MatchCase(pattern=Int(3), body=Int(4)), MatchCase(pattern=Int(1), body=Int(2))]), Int(1)
        )
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_eval_compose(self) -> None:
        exp = Compose(
            Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(3))),
            Function(Var("x"), Binop(BinopKind.MUL, Var("x"), Int(2))),
        )
        env = {"a": Int(1)}
        expected = Closure(
            {},
            Function(
                Var("x"),
                Apply(
                    Closure({}, Function(Var("x"), Binop(BinopKind.MUL, Var("x"), Int(2)))),
                    Apply(Closure({}, Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(3)))), Var("x")),
                ),
            ),
        )
        self.assertEqual(eval_exp(env, exp), expected)

    def test_eval_compose_apply(self) -> None:
        exp = Apply(
            Compose(
                Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(3))),
                Function(Var("x"), Binop(BinopKind.MUL, Var("x"), Int(2))),
            ),
            Int(4),
        )
        self.assertEqual(
            eval_exp({}, exp),
            Int(14),
        )

    def test_eval_native_function_returns_function(self) -> None:
        exp = NativeFunction("times2", lambda x: Int(x.value * 2))  # type: ignore [attr-defined]
        self.assertIs(eval_exp({}, exp), exp)

    def test_eval_apply_native_function_calls_function(self) -> None:
        exp = Apply(NativeFunction("times2", lambda x: Int(x.value * 2)), Int(3))  # type: ignore [attr-defined]
        self.assertEqual(eval_exp({}, exp), Int(6))

    def test_eval_apply_quote_returns_ast(self) -> None:
        ast = Binop(BinopKind.ADD, Int(1), Int(2))
        exp = Apply(Var("$$quote"), ast)
        self.assertIs(eval_exp({}, exp), ast)

    def test_eval_apply_closure_with_match_function_has_access_to_closure_vars(self) -> None:
        ast = Apply(Closure({"x": Int(1)}, MatchFunction([MatchCase(Var("y"), Var("x"))])), Int(2))
        self.assertEqual(eval_exp({}, ast), Int(1))

    def test_eval_less_returns_bool(self) -> None:
        ast = Binop(BinopKind.LESS, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), Symbol("true"))

    def test_eval_less_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.LESS, String("xyz"), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int or Float, got String")):
            eval_exp({}, ast)

    def test_eval_less_equal_returns_bool(self) -> None:
        ast = Binop(BinopKind.LESS_EQUAL, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), Symbol("true"))

    def test_eval_less_equal_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.LESS_EQUAL, String("xyz"), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int or Float, got String")):
            eval_exp({}, ast)

    def test_eval_greater_returns_bool(self) -> None:
        ast = Binop(BinopKind.GREATER, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), Symbol("false"))

    def test_eval_greater_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.GREATER, String("xyz"), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int or Float, got String")):
            eval_exp({}, ast)

    def test_eval_greater_equal_returns_bool(self) -> None:
        ast = Binop(BinopKind.GREATER_EQUAL, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), Symbol("false"))

    def test_eval_greater_equal_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.GREATER_EQUAL, String("xyz"), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int or Float, got String")):
            eval_exp({}, ast)

    def test_boolean_and_evaluates_args(self) -> None:
        ast = Binop(BinopKind.BOOL_AND, Symbol("true"), Var("a"))
        self.assertEqual(eval_exp({"a": Symbol("false")}, ast), Symbol("false"))

        ast = Binop(BinopKind.BOOL_AND, Var("a"), Symbol("false"))
        self.assertEqual(eval_exp({"a": Symbol("true")}, ast), Symbol("false"))

    def test_boolean_or_evaluates_args(self) -> None:
        ast = Binop(BinopKind.BOOL_OR, Symbol("false"), Var("a"))
        self.assertEqual(eval_exp({"a": Symbol("true")}, ast), Symbol("true"))

        ast = Binop(BinopKind.BOOL_OR, Var("a"), Symbol("true"))
        self.assertEqual(eval_exp({"a": Symbol("false")}, ast), Symbol("true"))

    def test_boolean_and_short_circuit(self) -> None:
        def raise_func(message: Object) -> Object:
            if not isinstance(message, String):
                raise TypeError(f"raise_func expected String, but got {type(message).__name__}")
            raise RuntimeError(message)

        error = NativeFunction("error", raise_func)
        apply = Apply(Var("error"), String("expected failure"))
        ast = Binop(BinopKind.BOOL_AND, Symbol("false"), apply)
        self.assertEqual(eval_exp({"error": error}, ast), Symbol("false"))

    def test_boolean_or_short_circuit(self) -> None:
        def raise_func(message: Object) -> Object:
            if not isinstance(message, String):
                raise TypeError(f"raise_func expected String, but got {type(message).__name__}")
            raise RuntimeError(message)

        error = NativeFunction("error", raise_func)
        apply = Apply(Var("error"), String("expected failure"))
        ast = Binop(BinopKind.BOOL_OR, Symbol("true"), apply)
        self.assertEqual(eval_exp({"error": error}, ast), Symbol("true"))

    def test_boolean_and_on_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.BOOL_AND, Int(1), Int(2))
        with self.assertRaisesRegex(TypeError, re.escape("expected #true or #false, got Int")):
            eval_exp({}, exp)

    def test_boolean_or_on_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.BOOL_OR, Int(1), Int(2))
        with self.assertRaisesRegex(TypeError, re.escape("expected #true or #false, got Int")):
            eval_exp({}, exp)

    def test_eval_record_with_spread_fails(self) -> None:
        exp = Record({"x": Spread()})
        with self.assertRaisesRegex(RuntimeError, "cannot evaluate a spread"):
            eval_exp({}, exp)

    def test_eval_symbol_returns_symbol(self) -> None:
        self.assertEqual(eval_exp({}, Symbol("abc")), Symbol("abc"))

    def test_eval_float_and_float_addition_returns_float(self) -> None:
        self.assertEqual(eval_exp({}, Binop(BinopKind.ADD, Float(1.0), Float(2.0))), Float(3.0))

    def test_eval_int_and_float_addition_returns_float(self) -> None:
        self.assertEqual(eval_exp({}, Binop(BinopKind.ADD, Int(1), Float(2.0))), Float(3.0))

    def test_eval_int_and_float_division_returns_float(self) -> None:
        self.assertEqual(eval_exp({}, Binop(BinopKind.DIV, Int(1), Float(2.0))), Float(0.5))

    def test_eval_float_and_int_division_returns_float(self) -> None:
        self.assertEqual(eval_exp({}, Binop(BinopKind.DIV, Float(1.0), Int(2))), Float(0.5))

    def test_eval_int_and_int_division_returns_float(self) -> None:
        self.assertEqual(eval_exp({}, Binop(BinopKind.DIV, Int(1), Int(2))), Float(0.5))


class EndToEndTestsBase(unittest.TestCase):
    def _run(self, text: str, env: Env | None = None) -> Object:
        tokens = tokenize(text)
        ast = parse(tokens)
        if env is None:
            env = boot_env()
        return eval_exp(env, ast)


class EndToEndTests(EndToEndTestsBase):
    def test_int_returns_int(self) -> None:
        self.assertEqual(self._run("1"), Int(1))

    def test_float_returns_float(self) -> None:
        self.assertEqual(self._run("3.14"), Float(3.14))

    def test_bytes_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~QUJD"), Bytes(b"ABC"))

    def test_bytes_base85_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~85'K|(_"), Bytes(b"ABC"))

    def test_bytes_base64_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~64'QUJD"), Bytes(b"ABC"))

    def test_bytes_base32_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~32'IFBEG==="), Bytes(b"ABC"))

    def test_bytes_base16_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~16'414243"), Bytes(b"ABC"))

    def test_int_add_returns_int(self) -> None:
        self.assertEqual(self._run("1 + 2"), Int(3))

    def test_int_sub_returns_int(self) -> None:
        self.assertEqual(self._run("1 - 2"), Int(-1))

    def test_string_concat_returns_string(self) -> None:
        self.assertEqual(self._run('"abc" ++ "def"'), String("abcdef"))

    def test_list_cons_returns_list(self) -> None:
        self.assertEqual(self._run("1 >+ [2,3]"), List([Int(1), Int(2), Int(3)]))

    def test_list_cons_nested_returns_list(self) -> None:
        self.assertEqual(self._run("1 >+ 2 >+ [3,4]"), List([Int(1), Int(2), Int(3), Int(4)]))

    def test_list_append_returns_list(self) -> None:
        self.assertEqual(self._run("[1,2] +< 3"), List([Int(1), Int(2), Int(3)]))

    def test_list_append_nested_returns_list(self) -> None:
        self.assertEqual(self._run("[1,2] +< 3 +< 4"), List([Int(1), Int(2), Int(3), Int(4)]))

    def test_empty_list(self) -> None:
        self.assertEqual(self._run("[ ]"), List([]))

    def test_empty_list_with_no_spaces(self) -> None:
        self.assertEqual(self._run("[]"), List([]))

    def test_list_of_ints(self) -> None:
        self.assertEqual(self._run("[ 1 , 2 ]"), List([Int(1), Int(2)]))

    def test_list_of_exprs(self) -> None:
        self.assertEqual(
            self._run("[ 1 + 2 , 3 + 4 ]"),
            List([Int(3), Int(7)]),
        )

    def test_where(self) -> None:
        self.assertEqual(self._run("a + 2 . a = 1"), Int(3))

    def test_nested_where(self) -> None:
        self.assertEqual(self._run("a + b . a = 1 . b = 2"), Int(3))

    def test_assert_with_truthy_cond_returns_value(self) -> None:
        self.assertEqual(self._run("a + 1 ? a == 1 . a = 1"), Int(2))

    def test_assert_with_falsey_cond_raises_assertion_error(self) -> None:
        with self.assertRaisesRegex(AssertionError, "condition a == 2 failed"):
            self._run("a + 1 ? a == 2 . a = 1")

    def test_nested_assert(self) -> None:
        self.assertEqual(self._run("a + b ? a == 1 ? b == 2 . a = 1 . b = 2"), Int(3))

    def test_hole(self) -> None:
        self.assertEqual(self._run("()"), Hole())

    def test_bindings_behave_like_letstar(self) -> None:
        with self.assertRaises(NameError) as ctx:
            self._run("b . a = 1 . b = a")
        self.assertEqual(ctx.exception.args[0], "name 'a' is not defined")

    def test_function_application_two_args(self) -> None:
        self.assertEqual(self._run("(a -> b -> a + b) 3 2"), Int(5))

    def test_function_create_list_correct_order(self) -> None:
        self.assertEqual(self._run("(a -> b -> [a, b]) 3 2"), List([Int(3), Int(2)]))

    def test_create_record(self) -> None:
        self.assertEqual(self._run("{a = 1 + 3}"), Record({"a": Int(4)}))

    def test_access_record(self) -> None:
        self.assertEqual(self._run('rec@b . rec = { a = 1, b = "x" }'), String("x"))

    def test_access_list(self) -> None:
        self.assertEqual(self._run("xs@1 . xs = [1, 2, 3]"), Int(2))

    def test_access_list_var(self) -> None:
        self.assertEqual(self._run("xs@y . y = 2 . xs = [1, 2, 3]"), Int(3))

    def test_access_list_expr(self) -> None:
        self.assertEqual(self._run("xs@(1+1) . xs = [1, 2, 3]"), Int(3))

    def test_functions_eval_arguments(self) -> None:
        self.assertEqual(self._run("(x -> x) c . c = 1"), Int(1))

    def test_non_var_function_arg_raises_parse_error(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self._run("1 -> a")
        self.assertEqual(ctx.exception.args[0], "expected variable in function definition 1")

    def test_compose(self) -> None:
        self.assertEqual(self._run("((a -> a + 1) >> (b -> b * 2)) 3"), Int(8))

    def test_compose_does_not_expose_internal_x(self) -> None:
        with self.assertRaisesRegex(NameError, "name 'x' is not defined"):
            self._run("f 3 . f = ((y -> x) >> (z -> x))")

    def test_double_compose(self) -> None:
        self.assertEqual(self._run("((a -> a + 1) >> (x -> x) >> (b -> b * 2)) 3"), Int(8))

    def test_reverse_compose(self) -> None:
        self.assertEqual(self._run("((a -> a + 1) << (b -> b * 2)) 3"), Int(7))

    def test_simple_int_match(self) -> None:
        self.assertEqual(
            self._run(
                """
                inc 2
                . inc =
                  | 1 -> 2
                  | 2 -> 3
                  """
            ),
            Int(3),
        )

    def test_match_var_binds_var(self) -> None:
        self.assertEqual(
            self._run(
                """
                id 3
                . id =
                  | x -> x
                """
            ),
            Int(3),
        )

    def test_match_var_binds_first_arm(self) -> None:
        self.assertEqual(
            self._run(
                """
                id 3
                . id =
                  | x -> x
                  | y -> y * 2
                """
            ),
            Int(3),
        )

    def test_match_function_can_close_over_variables(self) -> None:
        self.assertEqual(
            self._run(
                """
        f 1 2
        . f = a ->
          | b -> a + b
        """
            ),
            Int(3),
        )

    def test_match_record_binds_var(self) -> None:
        self.assertEqual(
            self._run(
                """
                get_x rec
                . rec = { x = 3 }
                . get_x =
                  | { x = x } -> x
                """
            ),
            Int(3),
        )

    def test_match_record_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult rec
                . rec = { x = 3, y = 4 }
                . mult =
                  | { x = x, y = y } -> x * y
                """
            ),
            Int(12),
        )

    def test_match_record_with_extra_fields_does_not_match(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                mult rec
                . rec = { x = 3 }
                . mult =
                  | { x = x, y = y } -> x * y
                """
            )

    def test_match_record_with_constant(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult rec
                . rec = { x = 4, y = 5 }
                . mult =
                  | { x = 3, y = y } -> 1
                  | { x = 4, y = y } -> 2
                """
            ),
            Int(2),
        )

    def test_match_record_with_non_record_fails(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                get_x 3
                . get_x =
                  | { x = x } -> x
                """
            )

    def test_match_record_doubly_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                get_x rec
                . rec = { a = 3, b = 3 }
                . get_x =
                  | { a = x, b = x } -> x
                """
            ),
            Int(3),
        )

    def test_match_list_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult xs
                . xs = [1, 2, 3, 4, 5]
                . mult =
                  | [1, x, 3, y, 5] -> x * y
                """
            ),
            Int(8),
        )

    def test_match_list_incorrect_length_does_not_match(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                mult xs
                . xs = [1, 2, 3]
                . mult =
                  | [1, 2] -> 1
                  | [1, 2, 3, 4] -> 1
                  | [1, 3] -> 1
                """
            )

    def test_match_list_with_constant(self) -> None:
        self.assertEqual(
            self._run(
                """
                middle xs
                . xs = [4, 5, 6]
                . middle =
                  | [1, x, 3] -> x
                  | [4, x, 6] -> x
                  | [7, x, 9] -> x
                """
            ),
            Int(5),
        )

    def test_match_list_with_non_list_fails(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                get_x 3
                . get_x =
                  | [2, x] -> x
                """
            )

    def test_match_list_doubly_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult xs
                . xs = [1, 2, 3, 2, 1]
                . mult =
                  | [1, x, 3, x, 1] -> x
                """
            ),
            Int(2),
        )

    def test_pipe(self) -> None:
        self.assertEqual(self._run("1 |> (a -> a + 2)"), Int(3))

    def test_pipe_nested(self) -> None:
        self.assertEqual(self._run("1 |> (a -> a + 2) |> (b -> b * 2)"), Int(6))

    def test_reverse_pipe(self) -> None:
        self.assertEqual(self._run("(a -> a + 2) <| 1"), Int(3))

    def test_reverse_pipe_nested(self) -> None:
        self.assertEqual(self._run("(b -> b * 2) <| (a -> a + 2) <| 1"), Int(6))

    def test_function_can_reference_itself(self) -> None:
        result = self._run(
            """
    f 1
    . f = n -> f
    """,
            {},
        )
        self.assertEqual(result, Closure({"f": result}, Function(Var("n"), Var("f"))))

    def test_function_can_call_itself(self) -> None:
        with self.assertRaises(RecursionError):
            self._run(
                """
        f 1
        . f = n -> f n
        """
            )

    def test_match_function_can_call_itself(self) -> None:
        self.assertEqual(
            self._run(
                """
        fac 5
        . fac =
          | 0 -> 1
          | 1 -> 1
          | n -> n * fac (n - 1)
        """
            ),
            Int(120),
        )

    def test_list_access_binds_tighter_than_append(self) -> None:
        self.assertEqual(self._run("[1, 2, 3] +< xs@0 . xs = [4]"), List([Int(1), Int(2), Int(3), Int(4)]))

    def test_exponentiation(self) -> None:
        self.assertEqual(self._run("6 ^ 2"), Int(36))

    def test_modulus(self) -> None:
        self.assertEqual(self._run("11 % 3"), Int(2))

    def test_exp_binds_tighter_than_mul(self) -> None:
        self.assertEqual(self._run("5 * 2 ^ 3"), Int(40))

    def test_symbol_true_returns_true(self) -> None:
        self.assertEqual(self._run("# true", {}), Symbol("true"))

    def test_symbol_false_returns_false(self) -> None:
        self.assertEqual(self._run("#false", {}), Symbol("false"))

    def test_boolean_and_binds_tighter_than_or(self) -> None:
        self.assertEqual(self._run("#true || #true && boom", {}), Symbol("true"))

    def test_compare_binds_tighter_than_boolean_and(self) -> None:
        self.assertEqual(self._run("1 < 2 && 2 < 1"), Symbol("false"))

    def test_match_list_spread(self) -> None:
        self.assertEqual(
            self._run(
                """
        f [2, 4, 6]
        . f =
          | [] -> 0
          | [x, ...] -> x
          | c -> 1
        """
            ),
            Int(2),
        )

    def test_match_list_named_spread(self) -> None:
        self.assertEqual(
            self._run(
                """
        tail [1,2,3]
        . tail =
          | [first, ...rest] -> rest
        """
            ),
            List([Int(2), Int(3)]),
        )

    def test_match_record_spread(self) -> None:
        self.assertEqual(
            self._run(
                """
        f {x = 4, y = 5}
        . f =
          | {} -> 0
          | {x = a, ...} -> a
          | c -> 1
        """
            ),
            Int(4),
        )

    def test_match_expr_as_boolean_symbols(self) -> None:
        self.assertEqual(
            self._run(
                """
        say (1 < 2)
        . say =
          | #false -> "oh no"
          | #true -> "omg"
        """
            ),
            String("omg"),
        )


class ClosureOptimizeTests(unittest.TestCase):
    def test_int(self) -> None:
        self.assertEqual(free_in(Int(1)), set())

    def test_float(self) -> None:
        self.assertEqual(free_in(Float(1.0)), set())

    def test_string(self) -> None:
        self.assertEqual(free_in(String("x")), set())

    def test_bytes(self) -> None:
        self.assertEqual(free_in(Bytes(b"x")), set())

    def test_hole(self) -> None:
        self.assertEqual(free_in(Hole()), set())

    def test_spread(self) -> None:
        self.assertEqual(free_in(Spread()), set())

    def test_spread_name(self) -> None:
        # TODO(max): Should this be assumed to always be in a place where it
        # defines a name, and therefore never have free variables?
        self.assertEqual(free_in(Spread("x")), {"x"})

    def test_nativefunction(self) -> None:
        self.assertEqual(free_in(NativeFunction("id", lambda x: x)), set())

    def test_symbol(self) -> None:
        self.assertEqual(free_in(Symbol("x")), set())

    def test_var(self) -> None:
        self.assertEqual(free_in(Var("x")), {"x"})

    def test_binop(self) -> None:
        self.assertEqual(free_in(Binop(BinopKind.ADD, Var("x"), Var("y"))), {"x", "y"})

    def test_empty_list(self) -> None:
        self.assertEqual(free_in(List([])), set())

    def test_list(self) -> None:
        self.assertEqual(free_in(List([Var("x"), Var("y")])), {"x", "y"})

    def test_empty_record(self) -> None:
        self.assertEqual(free_in(Record({})), set())

    def test_record(self) -> None:
        self.assertEqual(free_in(Record({"x": Var("x"), "y": Var("y")})), {"x", "y"})

    def test_function(self) -> None:
        exp = parse(tokenize("x -> x + y"))
        self.assertEqual(free_in(exp), {"y"})

    def test_nested_function(self) -> None:
        exp = parse(tokenize("x -> y -> x + y + z"))
        self.assertEqual(free_in(exp), {"z"})

    def test_match_function(self) -> None:
        exp = parse(tokenize("| 1 -> x | 2 -> y | x -> 3 | z -> 4"))
        self.assertEqual(free_in(exp), {"x", "y"})

    def test_match_case_int(self) -> None:
        exp = MatchCase(Int(1), Var("x"))
        self.assertEqual(free_in(exp), {"x"})

    def test_match_case_var(self) -> None:
        exp = MatchCase(Var("x"), Binop(BinopKind.ADD, Var("x"), Var("y")))
        self.assertEqual(free_in(exp), {"y"})

    def test_match_case_list(self) -> None:
        exp = MatchCase(List([Var("x")]), Binop(BinopKind.ADD, Var("x"), Var("y")))
        self.assertEqual(free_in(exp), {"y"})

    def test_match_case_list_spread(self) -> None:
        exp = MatchCase(List([Spread()]), Binop(BinopKind.ADD, Var("xs"), Var("y")))
        self.assertEqual(free_in(exp), {"xs", "y"})

    def test_match_case_list_spread_name(self) -> None:
        exp = MatchCase(List([Spread("xs")]), Binop(BinopKind.ADD, Var("xs"), Var("y")))
        self.assertEqual(free_in(exp), {"y"})

    def test_match_case_record(self) -> None:
        exp = MatchCase(
            Record({"x": Int(1), "y": Var("y"), "a": Var("z")}),
            Binop(BinopKind.ADD, Binop(BinopKind.ADD, Var("x"), Var("y")), Var("z")),
        )
        self.assertEqual(free_in(exp), {"x"})

    def test_match_case_record_spread(self) -> None:
        exp = MatchCase(Record({"...": Spread()}), Binop(BinopKind.ADD, Var("x"), Var("y")))
        self.assertEqual(free_in(exp), {"x", "y"})

    def test_match_case_record_spread_name(self) -> None:
        exp = MatchCase(Record({"...": Spread("x")}), Binop(BinopKind.ADD, Var("x"), Var("y")))
        self.assertEqual(free_in(exp), {"y"})

    def test_apply(self) -> None:
        self.assertEqual(free_in(Apply(Var("x"), Var("y"))), {"x", "y"})

    def test_access(self) -> None:
        self.assertEqual(free_in(Access(Var("x"), Var("y"))), {"x", "y"})

    def test_where(self) -> None:
        exp = parse(tokenize("x . x = 1"))
        self.assertEqual(free_in(exp), set())

    def test_where_same_name(self) -> None:
        exp = parse(tokenize("x . x = x+y"))
        self.assertEqual(free_in(exp), {"x", "y"})

    def test_assign(self) -> None:
        exp = Assign(Var("x"), Int(1))
        self.assertEqual(free_in(exp), set())

    def test_assign_same_name(self) -> None:
        exp = Assign(Var("x"), Var("x"))
        self.assertEqual(free_in(exp), {"x"})

    def test_closure(self) -> None:
        # TODO(max): Should x be considered free in the closure if it's in the
        # env?
        exp = Closure({"x": Int(1)}, Function(Var("_"), List([Var("x"), Var("y")])))
        self.assertEqual(free_in(exp), {"x", "y"})


class StdLibTests(EndToEndTestsBase):
    def test_stdlib_add(self) -> None:
        self.assertEqual(self._run("$$add 3 4", STDLIB), Int(7))

    def test_stdlib_quote(self) -> None:
        self.assertEqual(self._run("$$quote (3 + 4)"), Binop(BinopKind.ADD, Int(3), Int(4)))

    def test_stdlib_quote_pipe(self) -> None:
        self.assertEqual(self._run("3 + 4 |> $$quote"), Binop(BinopKind.ADD, Int(3), Int(4)))

    def test_stdlib_quote_reverse_pipe(self) -> None:
        self.assertEqual(self._run("$$quote <| 3 + 4"), Binop(BinopKind.ADD, Int(3), Int(4)))

    def test_stdlib_serialize(self) -> None:
        self.assertEqual(self._run("$$serialize 3", STDLIB), Bytes(value=b"d4:type3:Int5:valuei3ee"))

    def test_stdlib_serialize_expr(self) -> None:
        self.assertEqual(
            self._run("(1+2) |> $$quote |> $$serialize", STDLIB),
            Bytes(value=b"d4:leftd4:type3:Int5:valuei1ee2:op3:ADD5:rightd4:type3:Int5:valuei2ee4:type5:Binope"),
        )

    def test_stdlib_listlength_empty_list_returns_zero(self) -> None:
        self.assertEqual(self._run("$$listlength []", STDLIB), Int(0))

    def test_stdlib_listlength_returns_length(self) -> None:
        self.assertEqual(self._run("$$listlength [1,2,3]", STDLIB), Int(3))

    def test_stdlib_listlength_with_non_list_raises_type_error(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            self._run("$$listlength 1", STDLIB)
        self.assertEqual(ctx.exception.args[0], "listlength expected List, but got Int")


class PreludeTests(EndToEndTestsBase):
    def test_id_returns_input(self) -> None:
        self.assertEqual(self._run("id 123"), Int(123))

    def test_filter_returns_matching(self) -> None:
        self.assertEqual(
            self._run(
                """
        filter (x -> x < 4) [2, 6, 3, 7, 1, 8]
        """
            ),
            List([Int(2), Int(3), Int(1)]),
        )

    def test_filter_with_function_returning_non_bool_raises_match_error(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
        filter (x -> #no) [1]
        """
            )

    def test_quicksort(self) -> None:
        self.assertEqual(
            self._run(
                """
        quicksort [2, 6, 3, 7, 1, 8]
        """
            ),
            List([Int(1), Int(2), Int(3), Int(6), Int(7), Int(8)]),
        )

    def test_quicksort_with_empty_list(self) -> None:
        self.assertEqual(
            self._run(
                """
        quicksort []
        """
            ),
            List([]),
        )

    def test_quicksort_with_non_int_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        quicksort ["a", "c", "b"]
        """
            )

    def test_concat(self) -> None:
        self.assertEqual(
            self._run(
                """
        concat [1, 2, 3] [4, 5, 6]
        """
            ),
            List([Int(1), Int(2), Int(3), Int(4), Int(5), Int(6)]),
        )

    def test_concat_with_first_list_empty(self) -> None:
        self.assertEqual(
            self._run(
                """
        concat [] [4, 5, 6]
        """
            ),
            List([Int(4), Int(5), Int(6)]),
        )

    def test_concat_with_second_list_empty(self) -> None:
        self.assertEqual(
            self._run(
                """
        concat [1, 2, 3] []
        """
            ),
            List([Int(1), Int(2), Int(3)]),
        )

    def test_concat_with_both_lists_empty(self) -> None:
        self.assertEqual(
            self._run(
                """
        concat [] []
        """
            ),
            List([]),
        )

    def test_map(self) -> None:
        self.assertEqual(
            self._run(
                """
        map (x -> x * 2) [3, 1, 2]
        """
            ),
            List([Int(6), Int(2), Int(4)]),
        )

    def test_map_with_non_function_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        map 4 [3, 1, 2]
        """
            )

    def test_map_with_non_list_raises_match_error(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
        map (x -> x * 2) 3
        """
            )

    def test_range(self) -> None:
        self.assertEqual(
            self._run(
                """
        range 3
        """
            ),
            List([Int(0), Int(1), Int(2)]),
        )

    def test_range_with_non_int_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        range "a"
        """
            )

    def test_foldr(self) -> None:
        self.assertEqual(
            self._run(
                """
        foldr (x -> a -> a + x) 0 [1, 2, 3]
        """
            ),
            Int(6),
        )

    def test_foldr_on_empty_list_returns_empty_list(self) -> None:
        self.assertEqual(
            self._run(
                """
        foldr (x -> a -> a + x) 0 []
        """
            ),
            Int(0),
        )

    def test_take(self) -> None:
        self.assertEqual(
            self._run(
                """
        take 3 [1, 2, 3, 4, 5]
        """
            ),
            List([Int(1), Int(2), Int(3)]),
        )

    def test_take_n_more_than_list_length_returns_full_list(self) -> None:
        self.assertEqual(
            self._run(
                """
        take 5 [1, 2, 3]
        """
            ),
            List([Int(1), Int(2), Int(3)]),
        )

    def test_take_with_non_int_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        take "a" [1, 2, 3]
        """
            )

    def test_all_returns_true(self) -> None:
        self.assertEqual(
            self._run(
                """
        all (x -> x < 5) [1, 2, 3, 4]
        """
            ),
            Symbol("true"),
        )

    def test_all_returns_false(self) -> None:
        self.assertEqual(
            self._run(
                """
        all (x -> x < 5) [2, 4, 6]
        """
            ),
            Symbol("false"),
        )

    def test_all_with_empty_list_returns_true(self) -> None:
        self.assertEqual(
            self._run(
                """
        all (x -> x == 5) []
        """
            ),
            Symbol("true"),
        )

    def test_all_with_non_bool_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        all (x -> x) [1, 2, 3]
        """
            )

    def test_all_short_circuits(self) -> None:
        self.assertEqual(
            self._run(
                """
        all (x -> x > 1) [1, "a", "b"]
        """
            ),
            Symbol("false"),
        )

    def test_any_returns_true(self) -> None:
        self.assertEqual(
            self._run(
                """
        any (x -> x < 4) [1, 3, 5]
        """
            ),
            Symbol("true"),
        )

    def test_any_returns_false(self) -> None:
        self.assertEqual(
            self._run(
                """
        any (x -> x < 3) [4, 5, 6]
        """
            ),
            Symbol("false"),
        )

    def test_any_with_empty_list_returns_false(self) -> None:
        self.assertEqual(
            self._run(
                """
        any (x -> x == 5) []
        """
            ),
            Symbol("false"),
        )

    def test_any_with_non_bool_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        any (x -> x) [1, 2, 3]
        """
            )

    def test_any_short_circuits(self) -> None:
        self.assertEqual(
            self._run(
                """
        any (x -> x > 1) [2, "a", "b"]
        """
            ),
            Symbol("true"),
        )


class BencodeTests(unittest.TestCase):
    def test_bencode_int(self) -> None:
        self.assertEqual(bencode(123), b"i123e")

    def test_bencode_negative_int(self) -> None:
        self.assertEqual(bencode(-123), b"i-123e")

    def test_bencode_bytes(self) -> None:
        self.assertEqual(bencode(b"abc"), b"3:abc")

    def test_bencode_empty_list(self) -> None:
        self.assertEqual(bencode([]), b"le")

    def test_bencode_list_of_ints(self) -> None:
        self.assertEqual(bencode([1, 2, 3]), b"li1ei2ei3ee")

    def test_bencode_list_of_lists(self) -> None:
        self.assertEqual(bencode([[1, 2], [3, 4]]), b"lli1ei2eeli3ei4eee")

    def test_bencode_dict_sorts_keys(self) -> None:
        d = {}
        d[b"b"] = 1
        d[b"a"] = 2
        # It's sorted by insertion order (guaranteed Python 3.6+)
        self.assertEqual([*d], [b"b", b"a"])
        # It's sorted lexicographically
        self.assertEqual(bencode(d), b"d1:ai2e1:bi1ee")


class BdecodeTests(unittest.TestCase):
    def test_bdecode_int(self) -> None:
        self.assertEqual(bdecode("i123e"), 123)

    def test_bdecode_negative_int(self) -> None:
        self.assertEqual(bdecode("i-123e"), -123)

    def test_bdecode_bytes(self) -> None:
        self.assertEqual(bdecode("3:abc"), "abc")

    def test_bdecode_empty_list(self) -> None:
        self.assertEqual(bdecode("le"), [])

    def test_bdecode_list_of_ints(self) -> None:
        self.assertEqual(bdecode("li1ei2ei3ee"), [1, 2, 3])

    def test_bdecode_list_of_lists(self) -> None:
        self.assertEqual(bdecode("lli1ei2eeli3ei4eee"), [[1, 2], [3, 4]])

    def test_bdecode_dict_sorts_keys(self) -> None:
        self.assertEqual(bdecode("d1:ai2e1:bi1ee"), {"b": 1, "a": 2})


class ObjectSerializeTests(unittest.TestCase):
    def test_serialize_int(self) -> None:
        obj = Int(123)
        self.assertEqual(obj.serialize(), {"type": "Int", "value": 123})

    def test_serialize_negative_int(self) -> None:
        obj = Int(-123)
        self.assertEqual(obj.serialize(), {"type": "Int", "value": -123})

    def test_serialize_float_raises_not_implemented_error(self) -> None:
        obj = Float(3.14)
        with self.assertRaisesRegex(NotImplementedError, re.escape("serialization for Float is not supported")):
            obj.serialize()

    def test_serialize_str(self) -> None:
        obj = String("abc")
        self.assertEqual(obj.serialize(), {"type": "String", "value": "abc"})

    def test_serialize_bytes(self) -> None:
        obj = Bytes(b"abc")
        self.assertEqual(obj.serialize(), {"type": "Bytes", "value": "~~YWJj"})

    def test_serialize_var(self) -> None:
        obj = Var("abc")
        self.assertEqual(obj.serialize(), {"type": "Var", "name": "abc"})

    def test_serialize_symbol(self) -> None:
        obj = Symbol("true")
        self.assertEqual(obj.serialize(), {"type": "Symbol", "value": "true"})

    def test_serialize_binary_add(self) -> None:
        obj = Binop(BinopKind.ADD, Int(123), Int(456))
        self.assertEqual(
            obj.serialize(),
            {
                "left": {"type": "Int", "value": 123},
                "op": "ADD",
                "right": {"type": "Int", "value": 456},
                "type": "Binop",
            },
        )

    def test_serialize_list(self) -> None:
        obj = List([Int(1), Int(2)])
        self.assertEqual(
            obj.serialize(),
            {"type": "List", "items": [{"type": "Int", "value": 1}, {"type": "Int", "value": 2}]},
        )

    def test_serialize_assign(self) -> None:
        obj = Assign(Var("x"), Int(2))
        self.assertEqual(
            obj.serialize(),
            {"type": "Assign", "name": {"name": "x", "type": "Var"}, "value": {"type": "Int", "value": 2}},
        )

    def test_serialize_record(self) -> None:
        obj = Record({"x": Int(1)})
        self.assertEqual(obj.serialize(), {"data": {"x": {"type": "Int", "value": 1}}, "type": "Record"})

    def test_serialize_env_object(self) -> None:
        obj = EnvObject({"x": Int(1)})
        self.assertEqual(
            obj.serialize(),
            {"env": {"x": {"type": "Int", "value": 1}}, "type": "EnvObject"},
        )

    def test_serialize_function(self) -> None:
        obj = Function(Var("x"), Var("x"))
        self.assertEqual(
            obj.serialize(),
            {
                "arg": {"name": "x", "type": "Var"},
                "body": {"name": "x", "type": "Var"},
                "type": "Function",
            },
        )

    def test_serialize_apply(self) -> None:
        obj = Apply(Var("f"), Var("x"))
        self.assertEqual(
            obj.serialize(),
            {
                "func": {"name": "f", "type": "Var"},
                "arg": {"name": "x", "type": "Var"},
                "type": "Apply",
            },
        )

    def test_serialize_closure(self) -> None:
        obj = Closure({"a": Int(123)}, Function(Var("x"), Var("x")))
        self.assertEqual(
            obj.serialize(),
            {
                "env": {"a": {"type": "Int", "value": 123}},
                "func": {
                    "arg": {"name": "x", "type": "Var"},
                    "body": {"name": "x", "type": "Var"},
                    "type": "Function",
                },
                "type": "Closure",
            },
        )


class ObjectDeserializeTests(unittest.TestCase):
    def test_deserialize_int(self) -> None:
        msg = {"type": "Int", "value": 123}
        self.assertEqual(Object.deserialize(msg), Int(123))

    def test_deserialize_negative_int(self) -> None:
        msg = {"type": "Int", "value": -123}
        self.assertEqual(Object.deserialize(msg), Int(-123))

    def test_deserialize_str(self) -> None:
        msg = {"type": "String", "value": "abc"}
        self.assertEqual(Object.deserialize(msg), String("abc"))

    def test_deserialize_bytes(self) -> None:
        msg = {"type": "Bytes", "value": b"abc"}
        self.assertEqual(Object.deserialize(msg), Bytes(b"abc"))

    def test_deserialize_var(self) -> None:
        msg = {"type": "Var", "name": "abc"}
        self.assertEqual(Object.deserialize(msg), Var("abc"))

    def test_deserialize_symbol(self) -> None:
        msg = {"type": "Symbol", "value": "abc"}
        self.assertEqual(Object.deserialize(msg), Symbol("abc"))

    def test_deserialize_binary_add(self) -> None:
        msg = {
            "left": {"type": "Int", "value": 123},
            "op": "ADD",
            "right": {"type": "Int", "value": 456},
            "type": "Binop",
        }
        obj = Binop(BinopKind.ADD, Int(123), Int(456))
        self.assertEqual(Object.deserialize(msg), obj)

    def test_deserialize_env_object(self) -> None:
        obj = EnvObject({"x": Int(1)})
        msg = {"env": {"x": {"type": "Int", "value": 1}}, "type": "EnvObject"}
        self.assertEqual(Object.deserialize(msg), obj)

    def test_deserialize_function(self) -> None:
        obj = Function(Var("x"), Var("x"))
        msg = {
            "arg": {"name": "x", "type": "Var"},
            "body": {"name": "x", "type": "Var"},
            "type": "Function",
        }
        self.assertEqual(Object.deserialize(msg), obj)

    def test_deserialize_closure(self) -> None:
        obj = Closure({"a": Int(123)}, Function(Var("x"), Var("x")))
        msg = {
            "env": {"a": {"type": "Int", "value": 123}},
            "func": {
                "arg": {"name": "x", "type": "Var"},
                "body": {"name": "x", "type": "Var"},
                "type": "Function",
            },
            "type": "Closure",
        }
        self.assertEqual(Object.deserialize(msg), obj)

    def test_deserialize_native_function_relocation_returns_native_function_from_stdlib(self) -> None:
        obj = STDLIB["$$fetch"]
        msg = {"type": "NativeFunctionRelocation", "name": "$$fetch"}
        result = Object.deserialize(msg)
        self.assertEqual(result, obj)
        self.assertIs(result, obj)

    def test_deserialize_apply(self) -> None:
        obj = Apply(Var("f"), Var("x"))
        msg = {
            "func": {"name": "f", "type": "Var"},
            "arg": {"name": "x", "type": "Var"},
            "type": "Apply",
        }
        self.assertEqual(Object.deserialize(msg), obj)

    def test_deserialize_record(self) -> None:
        obj = Record({"x": Int(1), "y": Int(2)})
        msg = {
            "data": {"x": {"type": "Int", "value": 1}, "y": {"type": "Int", "value": 2}},
            "type": "Record",
        }
        self.assertEqual(Object.deserialize(msg), obj)


class SerializeTests(unittest.TestCase):
    def test_serialize_int(self) -> None:
        obj = Int(3)
        self.assertEqual(serialize(obj), b"d4:type3:Int5:valuei3ee")

    def test_serialize_str(self) -> None:
        obj = String("abc")
        self.assertEqual(serialize(obj), b"d4:type6:String5:value3:abce")

    def test_serialize_bytes(self) -> None:
        obj = Bytes(b"abc")
        self.assertEqual(serialize(obj), b"d4:type5:Bytes5:value6:~~YWJje")

    def test_serialize_var(self) -> None:
        obj = Var("abc")
        self.assertEqual(serialize(obj), b"d4:name3:abc4:type3:Vare")

    def test_serialize_symbol(self) -> None:
        obj = Symbol("abcd")
        self.assertEqual(serialize(obj), b"d4:type6:Symbol5:value4:abcde")

    def test_serialize_function(self) -> None:
        obj = Function(Var("x"), Binop(BinopKind.ADD, Int(1), Var("x")))
        self.assertEqual(
            serialize(obj),
            b"d3:argd4:name1:x4:type3:Vare4:bodyd4:leftd4:type3:Int5:valuei1ee2:op3:ADD5:rightd4:name1:x4:type3:Vare4:type5:Binope4:type8:Functione",
        )


class ScrapMonadTests(unittest.TestCase):
    def test_create_copies_env(self) -> None:
        env = {"a": Int(123)}
        result = ScrapMonad(env)
        self.assertEqual(result.env, env)
        self.assertIsNot(result.env, env)

    def test_bind_returns_new_monad(self) -> None:
        env = {"a": Int(123)}
        orig = ScrapMonad(env)
        result, next_monad = orig.bind(Assign(Var("b"), Int(456)))
        self.assertEqual(orig.env, {"a": Int(123)})
        self.assertEqual(next_monad.env, {"a": Int(123), "b": Int(456)})


class PrettyPrintTests(unittest.TestCase):
    def test_pretty_print_int(self) -> None:
        obj = Int(1)
        self.assertEqual(str(obj), "1")

    def test_pretty_print_float(self) -> None:
        obj = Float(3.14)
        self.assertEqual(str(obj), "3.14")

    def test_pretty_print_string(self) -> None:
        obj = String("hello")
        self.assertEqual(str(obj), '"hello"')

    def test_pretty_print_bytes(self) -> None:
        obj = Bytes(b"abc")
        self.assertEqual(str(obj), "~~YWJj")

    def test_pretty_print_var(self) -> None:
        obj = Var("ref")
        self.assertEqual(str(obj), "ref")

    def test_pretty_print_hole(self) -> None:
        obj = Hole()
        self.assertEqual(str(obj), "()")

    def test_pretty_print_spread(self) -> None:
        obj = Spread()
        self.assertEqual(str(obj), "...")

    def test_pretty_print_named_spread(self) -> None:
        obj = Spread("rest")
        self.assertEqual(str(obj), "...rest")

    def test_pretty_print_binop(self) -> None:
        obj = Binop(BinopKind.ADD, Int(1), Int(2))
        self.assertEqual(str(obj), "1 + 2")

    def test_pretty_print_int_list(self) -> None:
        obj = List([Int(1), Int(2), Int(3)])
        self.assertEqual(str(obj), "[1, 2, 3]")

    def test_pretty_print_str_list(self) -> None:
        obj = List([String("1"), String("2"), String("3")])
        self.assertEqual(str(obj), '["1", "2", "3"]')

    def test_pretty_print_assign(self) -> None:
        obj = Assign(Var("x"), Int(3))
        self.assertEqual(str(obj), "x = 3")

    def test_pretty_print_function(self) -> None:
        obj = Function(Var("x"), Binop(BinopKind.ADD, Int(1), Var("x")))
        self.assertEqual(
            str(obj),
            "Function(arg=Var(name='x'), body=Binop(op=<BinopKind.ADD: 1>, left=Int(value=1), right=Var(name='x')))",
        )

    def test_pretty_print_apply(self) -> None:
        obj = Apply(Var("x"), Var("y"))
        self.assertEqual(str(obj), "Apply(func=Var(name='x'), arg=Var(name='y'))")

    def test_pretty_print_compose(self) -> None:
        obj = Compose(
            Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(3))),
            Function(Var("x"), Binop(BinopKind.MUL, Var("x"), Int(2))),
        )
        self.assertEqual(
            str(obj),
            "Compose(inner=Function(arg=Var(name='x'), body=Binop(op=<BinopKind.ADD: 1>, "
            "left=Var(name='x'), right=Int(value=3))), outer=Function(arg=Var(name='x'), "
            "body=Binop(op=<BinopKind.MUL: 3>, left=Var(name='x'), right=Int(value=2))))",
        )

    def test_pretty_print_where(self) -> None:
        obj = Where(
            Binop(BinopKind.ADD, Var("a"), Var("b")),
            Assign(Var("a"), Int(1)),
        )
        self.assertEqual(
            str(obj),
            "Where(body=Binop(op=<BinopKind.ADD: 1>, left=Var(name='a'), "
            "right=Var(name='b')), binding=Assign(name=Var(name='a'), value=Int(value=1)))",
        )

    def test_pretty_print_assert(self) -> None:
        obj = Assert(Int(123), Symbol("true"))
        self.assertEqual(str(obj), "Assert(value=Int(value=123), cond=Symbol(value='true'))")

    def test_pretty_print_envobject(self) -> None:
        obj = EnvObject({"x": Int(1)})
        self.assertEqual(str(obj), "EnvObject(keys=dict_keys(['x']))")

    def test_pretty_print_matchcase(self) -> None:
        obj = MatchCase(pattern=Int(1), body=Int(2))
        self.assertEqual(str(obj), "MatchCase(pattern=Int(value=1), body=Int(value=2))")

    def test_pretty_print_matchfunction(self) -> None:
        obj = MatchFunction([MatchCase(Var("y"), Var("x"))])
        self.assertEqual(str(obj), "MatchFunction(cases=[MatchCase(pattern=Var(name='y'), body=Var(name='x'))])")

    def test_pretty_print_relocation(self) -> None:
        obj = Relocation("relocate")
        self.assertEqual(str(obj), "Relocation(name='relocate')")

    def test_pretty_print_nativefunction(self) -> None:
        obj = NativeFunction("times2", lambda x: Int(x.value * 2))  # type: ignore [attr-defined]
        self.assertEqual(str(obj), "NativeFunction(name=times2)")

    def test_pretty_print_closure(self) -> None:
        obj = Closure({"a": Int(123)}, Function(Var("x"), Var("x")))
        self.assertEqual(
            str(obj), "Closure(env={'a': Int(value=123)}, func=Function(arg=Var(name='x'), body=Var(name='x')))"
        )

    def test_pretty_print_record(self) -> None:
        obj = Record({"a": Int(1), "b": Int(2)})
        self.assertEqual(str(obj), "{a = 1, b = 2}")

    def test_pretty_print_access(self) -> None:
        obj = Access(Record({"a": Int(4)}), Var("a"))
        self.assertEqual(str(obj), "Access(obj=Record(data={'a': Int(value=4)}), at=Var(name='a'))")

    def test_pretty_print_symbol(self) -> None:
        obj = Symbol("x")
        self.assertEqual(str(obj), "#x")

