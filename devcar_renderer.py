"""
DevCAR Renderer
----------------
Standalone Parser + Renderer for a subset of the DevCAR DSL used by the
launcher. This module intentionally duplicates a minimal, robust subset
of the language so it can be executed and tested without importing the
giant `laucheride.py` script.

Features:
- Lexer for tokens used by the small DSL
- Parser producing a small AST
- HTMLGenerator visitor producing safe HTML with Tailwind-like classes
- Command-line helper and programmatic render_file() function

This file intentionally aims to be self-contained and well-documented
so unit tests can validate rendering even in CI environments where
PyQt is not installed.
"""

from __future__ import annotations

import re
import json
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, List, Optional, Dict

__all__ = [
    "render_file",
    "render_string",
    "DevCarLexer",
    "Parser",
]


class TokenType(Enum):
    # Structural
    IDENT = auto()
    NUMBER = auto()
    STRING = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    EQUALS = auto()
    DOT = auto()
    NEWLINE = auto()
    EOF = auto()

    # Keywords
    LAYOUT = auto()
    CREATE = auto()
    TABLE = auto()
    QUICKPICK = auto()
    ICON = auto()
    ITEM = auto()
    LIST = auto()
    INPUT = auto()
    CHECKBOX = auto()
    TITLE = auto()
    FORM = auto()
    COMPONENT = auto()
    NAVBAR = auto()
    CHART = auto()
    INCLUDE_COMPONENT = auto()
    HTML_BLOCK = auto()
    IF = auto()
    ELSE = auto()
    ENDIF = auto()
    FOR = auto()
    EACH = auto()
    IN = auto()
    ENDFOR = auto()


@dataclass
class Token:
    type: TokenType
    lexeme: str
    literal: Any
    line: int


class DevCarLexer:
    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.start = 0
        self.current = 0
        self.line = 1

    def lex(self) -> List[Token]:
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        self.tokens.append(Token(TokenType.EOF, "", None, self.line))
        return self.tokens

    def is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def advance(self) -> str:
        ch = self.source[self.current]
        self.current += 1
        return ch

    def peek(self) -> str:
        if self.is_at_end():
            return "\0"
        return self.source[self.current]

    def add_token(self, ttype: TokenType, literal: Any = None):
        text = self.source[self.start:self.current]
        self.tokens.append(Token(ttype, text, literal, self.line))

    def scan_token(self):
        c = self.advance()
        if c == "\n":
            self.add_token(TokenType.NEWLINE)
            self.line += 1
            return
        if c.isspace():
            return
        if c == '[':
            self.add_token(TokenType.LBRACKET)
            return
        if c == ']':
            self.add_token(TokenType.RBRACKET)
            return
        if c == ',':
            self.add_token(TokenType.COMMA)
            return
        if c == '=':
            self.add_token(TokenType.EQUALS)
            return
        if c == '.':
            self.add_token(TokenType.DOT)
            return
        if c == '"' or c == "'":
            quote = c
            start = self.current
            while not self.is_at_end() and self.peek() != quote:
                if self.peek() == '\n':
                    self.line += 1
                self.advance()
            if self.is_at_end():
                value = self.source[start:self.current]
            else:
                value = self.source[start:self.current]
                self.advance()
            self.add_token(TokenType.STRING, value)
            return
        if c.isdigit():
            start = self.current - 1
            while self.peek().isdigit():
                self.advance()
            if self.peek() == '.' and self.source[self.current+1].isdigit():
                self.advance()
                while self.peek().isdigit():
                    self.advance()
            value = float(self.source[start:self.current])
            self.add_token(TokenType.NUMBER, value)
            return
        if c.isalpha():
            start = self.current - 1
            while self.peek().isalnum() or self.peek() in ['_', '-']:
                self.advance()
            word = self.source[start:self.current]
            upper = word.upper()
            # Map keywords
            kw_map = {
                'LAYOUT': TokenType.LAYOUT,
                'CREATE': TokenType.CREATE,
                'TABLE': TokenType.TABLE,
                'QUICKPICK': TokenType.QUICKPICK,
                'ICON': TokenType.ICON,
                'ITEM': TokenType.ITEM,
                'LIST': TokenType.LIST,
                'INPUT': TokenType.INPUT,
                'CHECKBOX': TokenType.CHECKBOX,
                'TITLE': TokenType.TITLE,
                'FORM': TokenType.FORM,
                'COMPONENT': TokenType.COMPONENT,
                'NAVBAR': TokenType.NAVBAR,
                'CHART': TokenType.CHART,
                'INCLUDE_COMPONENT': TokenType.INCLUDE_COMPONENT,
                'HTML_BLOCK': TokenType.HTML_BLOCK,
                'IF': TokenType.IF,
                'ELSE': TokenType.ELSE,
                'ENDIF': TokenType.ENDIF,
                'FOR': TokenType.FOR,
                'EACH': TokenType.EACH,
                'IN': TokenType.IN,
                'ENDFOR': TokenType.ENDFOR,
            }
            if upper in kw_map:
                self.add_token(kw_map[upper])
            else:
                self.add_token(TokenType.IDENT, word)
            return


# --- AST Nodes ---
class ASTNode:
    pass

@dataclass
class Literal(ASTNode):
    value: Any

@dataclass
class Identifier(ASTNode):
    name: str

@dataclass
class TitleNode(ASTNode):
    content: str

@dataclass
class InputNode(ASTNode):
    name: str
    type: str
    placeholder: Optional[str]

@dataclass
class ButtonNode(ASTNode):
    label: str
    style: str

@dataclass
class TableNode(ASTNode):
    title: str
    columns: List[str]

@dataclass
class ItemListNode(ASTNode):
    items: List[str]

@dataclass
class FormNode(ASTNode):
    name: str
    body: List[ASTNode]


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF

    def peek(self) -> Token:
        return self.tokens[self.current]

    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.tokens[self.current - 1]

    def match(self, *types) -> bool:
        if self.peek().type in types:
            self.advance()
            return True
        return False

    def consume(self, ttype: TokenType, msg: str = 'Expected token') -> Token:
        if self.peek().type == ttype:
            return self.advance()
        raise SyntaxError(f"{msg} at line {self.peek().line}")

    def parse(self) -> List[ASTNode]:
        nodes: List[ASTNode] = []
        while not self.is_at_end():
            t = self.peek()
            if t.type == TokenType.NEWLINE:
                self.advance(); continue
            if t.type == TokenType.TITLE:
                nodes.append(self.title())
                continue
            if t.type == TokenType.INPUT:
                nodes.append(self.input())
                continue
            if t.type == TokenType.QUICKPICK:
                nodes.append(self.button())
                continue
            if t.type == TokenType.CREATE:
                nodes.append(self.create_table())
                continue
            if t.type == TokenType.ITEM:
                # Expect 'LIST' after ITEM
                self.advance()
                if self.peek().type == TokenType.LIST:
                    self.advance()
                nodes.append(self.item_list())
                continue
            if t.type == TokenType.FORM:
                nodes.append(self.form())
                continue
            # fallback: consume and skip
            self.advance()
        return nodes

    def title(self) -> TitleNode:
        self.consume(TokenType.TITLE)
        self.consume(TokenType.LBRACKET)
        tok = self.consume(TokenType.STRING, "Expect title string")
        self.consume(TokenType.RBRACKET)
        return TitleNode(tok.literal)

    def input(self) -> InputNode:
        self.consume(TokenType.INPUT)
        self.consume(TokenType.LBRACKET)
        name_tok = self.consume(TokenType.IDENT)
        typ = 'text'
        placeholder = None
        if self.match(TokenType.COMMA):
            # simple property parse: expect IDENT=STRING pairs
            while not self.match(TokenType.RBRACKET):
                prop = self.consume(TokenType.IDENT).literal
                self.consume(TokenType.EQUALS)
                val_tok = self.consume(TokenType.IDENT) if self.peek().type == TokenType.IDENT else self.consume(TokenType.STRING)
                if prop.lower() == 'type':
                    typ = val_tok.literal
                if prop.lower() == 'placeholder':
                    placeholder = val_tok.literal
                if self.peek().type == TokenType.COMMA:
                    self.advance()
            return InputNode(name_tok.literal, typ, placeholder)
        self.consume(TokenType.RBRACKET)
        return InputNode(name_tok.literal, typ, placeholder)

    def button(self) -> ButtonNode:
        self.consume(TokenType.QUICKPICK)
        self.consume(TokenType.LBRACKET)
        label_tok = self.consume(TokenType.IDENT) if self.peek().type == TokenType.IDENT else self.consume(TokenType.STRING)
        style = 'primary'
        if self.match(TokenType.COMMA):
            style_tok = self.consume(TokenType.IDENT)
            style = style_tok.literal
        self.consume(TokenType.RBRACKET)
        return ButtonNode(label_tok.literal, style)

    def create_table(self) -> TableNode:
        # expect: CREATE TABLE[Title, Col1, Col2, ...]
        self.consume(TokenType.CREATE)
        if self.peek().type == TokenType.TABLE:
            self.advance()
        self.consume(TokenType.LBRACKET)
        title_tok = self.consume(TokenType.IDENT) if self.peek().type == TokenType.IDENT else self.consume(TokenType.STRING)
        cols: List[str] = []
        while not self.match(TokenType.RBRACKET):
            if self.match(TokenType.COMMA):
                tok = self.consume(TokenType.IDENT) if self.peek().type == TokenType.IDENT else self.consume(TokenType.STRING)
                cols.append(tok.literal)
            else:
                tok = self.consume(TokenType.IDENT) if self.peek().type == TokenType.IDENT else self.consume(TokenType.STRING)
                cols.append(tok.literal)
                if self.peek().type == TokenType.COMMA:
                    self.advance()
        return TableNode(title_tok.literal, cols)

    def item_list(self) -> ItemListNode:
        self.consume(TokenType.LBRACKET)
        items: List[str] = []
        # may be empty
        while not self.match(TokenType.RBRACKET):
            if self.peek().type in (TokenType.IDENT, TokenType.STRING):
                tok = self.advance()
                items.append(tok.literal)
            if self.peek().type == TokenType.COMMA:
                self.advance()
        return ItemListNode(items)

    def form(self) -> FormNode:
        self.consume(TokenType.FORM)
        self.consume(TokenType.LBRACKET)
        name_tok = self.consume(TokenType.IDENT) if self.peek().type == TokenType.IDENT else self.consume(TokenType.STRING)
        self.consume(TokenType.RBRACKET)
        body: List[ASTNode] = []
        # naive: consume until ENDIF or EOF or NEWLINE markers - for simplicity we'll read until next FORM or EOF
        while not self.is_at_end() and self.peek().type not in (TokenType.FORM, TokenType.EOF):
            if self.peek().type == TokenType.TITLE:
                body.append(self.title()); continue
            if self.peek().type == TokenType.INPUT:
                body.append(self.input()); continue
            if self.peek().type == TokenType.QUICKPICK:
                body.append(self.button()); continue
            # consume and ignore unknowns
            self.advance()
        return FormNode(name_tok.literal, body)


class HTMLEscaper:
    @staticmethod
    def escape(text: str) -> str:
        return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


class HTMLGenerator:
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        self.esc = HTMLEscaper()

    def generate(self, nodes: List[ASTNode]) -> str:
        body = []
        for n in nodes:
            body.append(self.visit(n))
        return self._wrap(''.join(body))

    def visit(self, node: ASTNode) -> str:
        if isinstance(node, TitleNode):
            return f'<h2 class="text-xl font-bold mb-2">{self.esc.escape(node.content)}</h2>'
        if isinstance(node, InputNode):
            return (f'<div class="mb-3"><label class="block text-sm font-medium">{self.esc.escape(node.name)}</label>'
                    f'<input type="{self.esc.escape(node.type)}" name="{self.esc.escape(node.name)}" placeholder="{self.esc.escape(node.placeholder or "")}" class="p-2 border rounded w-full"/></div>')
        if isinstance(node, ButtonNode):
            classes = 'bg-indigo-600 text-white' if node.style == 'primary' else 'bg-gray-300 text-black'
            return f'<button class="px-4 py-2 rounded {classes}" onclick="alert(\'{self.esc.escape(node.label)}\')">{self.esc.escape(node.label)}</button>'
        if isinstance(node, TableNode):
            cols = ''.join(f'<th class="px-4 py-2">{self.esc.escape(c)}</th>' for c in node.columns)
            sample = ''.join('<td class="px-4 py-2">Sample</td>' for _ in node.columns)
            return f'<div class="mb-4"><h3 class="font-bold">{self.esc.escape(node.title)}</h3><table class="table"><thead><tr>{cols}</tr></thead><tbody><tr>{sample}</tr></tbody></table></div>'
        if isinstance(node, ItemListNode):
            items = ''.join(f'<li>{self.esc.escape(it)}</li>' for it in node.items)
            return f'<ul class="list-disc pl-4">{items}</ul>'
        if isinstance(node, FormNode):
            inner = ''.join(self.visit(child) for child in node.body)
            return f'<form id="{self.esc.escape(node.name)}" class="bg-white p-4 rounded">{inner}<div class="mt-3"><button class="px-3 py-2 bg-green-500 text-white rounded">Submit</button></div></form>'
        return ''

    def _wrap(self, body_html: str) -> str:
        # Minimal but self-contained HTML with local fallback styles
        css = """
        body { font-family: Inter, Arial, sans-serif; }
        .container { max-width: 900px; margin: 24px auto; }
        """
        return f"""<!doctype html>
        <html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
        <style>{css}</style>
        </head><body><div class='container'>{body_html}</div></body></html>"""


def render_string(source: str) -> str:
    lexer = DevCarLexer(source)
    tokens = lexer.lex()
    parser = Parser(tokens)
    ast = parser.parse()
    gen = HTMLGenerator()
    return gen.generate(ast)


def render_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        src = f.read()
    return render_string(src)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('file', nargs='?', help='DevCAR source to render')
    args = p.parse_args()
    if args.file:
        print(render_file(args.file))
    else:
        print(render_string("TITLE[\"Hello DevCAR\"]\nQUICKPICK[\"Run\", primary]"))
