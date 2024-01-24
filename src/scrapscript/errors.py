class ParseError(SyntaxError):
    pass


# TODO(max): Replace with EOFError?
class UnexpectedEOFError(ParseError):
    pass
