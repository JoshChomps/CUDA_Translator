"""Lexer / Tokenizer for NVIDIA PTX (Parallel Thread Execution) Assembly."""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Iterator, Optional


class TokenType(Enum):
    DIRECTIVE = auto()   # .version, .target, .entry, .param, .reg, .shared
    IDENTIFIER = auto()  # %r1, %tid.x, my_kernel, %p0
    PREDICATE = auto()   # @%p1, @!%p1
    NUMBER = auto()      # 42, 0x1f, 3.1415
    STRING = auto()      # "string"
    COMMA = auto()       # ,
    SEMICOLON = auto()   # ;
    COLON = auto()       # :
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    PLUS = auto()        # +
    MINUS = auto()       # -
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"Token({self.type.name}, '{self.value}', L{self.line}:{self.column})"


class PTXLexer:
    """Tokenizer converting raw PTX text into structured tokens."""

    TOKEN_SPEC = [
        ("COMMENT_LINE", r"//[^\n]*"),
        ("COMMENT_BLOCK", r"/\*[\s\S]*?\*/"),
        ("PREDICATE", r"@!?%[a-zA-Z0-9_]+"),
        ("DIRECTIVE", r"\.[a-zA-Z0-9_]+"),
        ("HEX_INT", r"0[xX][0-9a-fA-F]+"),
        ("FLOAT", r"[+-]?[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?"),
        ("INT", r"[+-]?[0-9]+"),
        ("IDENTIFIER", r"[%$]?[a-zA-Z_][a-zA-Z0-9_.]*|<[0-9]+>"),
        ("STRING", r'"[^"\\]*(?:\\.[^"\\]*)*"'),
        ("COMMA", r","),
        ("SEMICOLON", r";"),
        ("COLON", r":"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("WHITESPACE", r"[ \t\r\n]+"),
    ]

    def __init__(self, source: str) -> None:
        self.source = source
        regex_parts = [f"(?P<{name}>{pattern})" for name, pattern in self.TOKEN_SPEC]
        self.master_pattern = re.compile("|".join(regex_parts))

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        line = 1
        line_start = 0

        for match in self.master_pattern.finditer(self.source):
            kind = match.lastgroup
            val = match.group()
            col = match.start() - line_start + 1

            if kind in {"COMMENT_LINE", "COMMENT_BLOCK"}:
                newlines = val.count("\n")
                if newlines > 0:
                    line += newlines
                    line_start = match.end() - len(val.split("\n")[-1])
                continue
            elif kind == "WHITESPACE":
                newlines = val.count("\n")
                if newlines > 0:
                    line += newlines
                    line_start = match.end() - len(val.split("\n")[-1])
                continue
            elif kind == "DIRECTIVE":
                tokens.append(Token(TokenType.DIRECTIVE, val, line, col))
            elif kind == "PREDICATE":
                tokens.append(Token(TokenType.PREDICATE, val, line, col))
            elif kind in {"HEX_INT", "INT", "FLOAT"}:
                tokens.append(Token(TokenType.NUMBER, val, line, col))
            elif kind == "IDENTIFIER":
                tokens.append(Token(TokenType.IDENTIFIER, val, line, col))
            elif kind == "STRING":
                tokens.append(Token(TokenType.STRING, val, line, col))
            elif kind == "COMMA":
                tokens.append(Token(TokenType.COMMA, val, line, col))
            elif kind == "SEMICOLON":
                tokens.append(Token(TokenType.SEMICOLON, val, line, col))
            elif kind == "COLON":
                tokens.append(Token(TokenType.COLON, val, line, col))
            elif kind == "LPAREN":
                tokens.append(Token(TokenType.LPAREN, val, line, col))
            elif kind == "RPAREN":
                tokens.append(Token(TokenType.RPAREN, val, line, col))
            elif kind == "LBRACKET":
                tokens.append(Token(TokenType.LBRACKET, val, line, col))
            elif kind == "RBRACKET":
                tokens.append(Token(TokenType.RBRACKET, val, line, col))
            elif kind == "LBRACE":
                tokens.append(Token(TokenType.LBRACE, val, line, col))
            elif kind == "RBRACE":
                tokens.append(Token(TokenType.RBRACE, val, line, col))
            elif kind == "PLUS":
                tokens.append(Token(TokenType.PLUS, val, line, col))
            elif kind == "MINUS":
                tokens.append(Token(TokenType.MINUS, val, line, col))

        tokens.append(Token(TokenType.EOF, "", line, 1))
        return tokens
