package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// === Lexer ===
const (
	EOF       = ""
	ILLEGAL   = "ILLEGAL"
	IDENT     = "IDENT"
	INT       = "INT"
	STRING    = "STRING"
	ASSIGN    = "="
	PLUS      = "+"
	MINUS     = "-"
	BANG      = "!"
	ASTERISK  = "*"
	SLASH     = "/"
	LT        = "<"
	GT        = ">"
	COMMA     = ","
	SEMICOLON = ";"
	LPAREN    = "("
	RPAREN    = ")"
	LBRACE    = "{"
	RBRACE    = "}"
	TRUE      = "true"
	FALSE     = "false"
	FUNCTION  = "fn"
	LET       = "let"
	IF        = "if"
	ELSE      = "else"
	RETURN    = "return"
)

var keywords = map[string]string{
	"fn":     FUNCTION,
	"let":    LET,
	"true":   TRUE,
	"false":  FALSE,
	"if":     IF,
	"else":   ELSE,
	"return": RETURN,
}

type Token struct { Type, Literal string }

type Lexer struct { input string; pos, readPos int; ch byte }

func NewLexer(in string) *Lexer { l := &Lexer{input:in}; l.readChar(); return l }
func (l *Lexer) readChar() {
	if l.readPos >= len(l.input) { l.ch = 0 } else { l.ch = l.input[l.readPos] }
	l.pos = l.readPos; l.readPos++
}
func (l *Lexer) NextToken() Token {
	l.skipWhitespace()
	var tok Token
	switch l.ch {
	case '=':
		tok = newToken(ASSIGN, l.ch)
	case '+': tok = newToken(PLUS, l.ch)
	case '-': tok = newToken(MINUS, l.ch)
	case '!': tok = newToken(BANG, l.ch)
	case '/': tok = newToken(SLASH, l.ch)
	case '*': tok = newToken(ASTERISK, l.ch)
	case '<': tok = newToken(LT, l.ch)
	case '>': tok = newToken(GT, l.ch)
	case ';': tok = newToken(SEMICOLON, l.ch)
	case ',': tok = newToken(COMMA, l.ch)
	case '(':
		tok = newToken(LPAREN, l.ch)
	case ')': tok = newToken(RPAREN, l.ch)
	case '{': tok = newToken(LBRACE, l.ch)
	case '}': tok = newToken(RBRACE, l.ch)
	case '"': tok.Type = STRING; tok.Literal = l.readString()
	case 0:
		tok.Literal = ""; tok.Type = EOF
	default:
		if isLetter(l.ch) {
			lit := l.readIdentifier(); typ := IDENT
			if kw, ok := keywords[lit]; ok { typ = kw }
			tok = Token{typ, lit}; return tok
		} else if isDigit(l.ch) {
			lit := l.readNumber(); tok = Token{INT, lit}; return tok
		} else {
			tok = newToken(ILLEGAL, l.ch)
		}
	}
	l.readChar(); return tok
}
func newToken(t string, ch byte) Token { return Token{t, string(ch)} }
func (l *Lexer) skipWhitespace() {
	for l.ch==' '||l.ch=='\t'||l.ch=='\n'||l.ch=='\r' { l.readChar() }
}
func isLetter(ch byte) bool { return 'a'<=ch&&ch<='z' || 'A'<=ch&&ch<='Z' || ch=='_' }
func (l *Lexer) readIdentifier() string { pos:=l.pos; for isLetter(l.ch){l.readChar()}; return l.input[pos:l.pos] }
func isDigit(ch byte) bool { return '0'<=ch && ch<='9' }
func (l *Lexer) readNumber() string { pos:=l.pos; for isDigit(l.ch){l.readChar()}; return l.input[pos:l.pos] }
func (l *Lexer) readString() string {
	l.readChar(); pos:=l.pos
	for l.ch!='"' && l.ch!=0 { l.readChar() }
	s := l.input[pos:l.pos]; l.readChar(); return s
}

// === AST Nodes ===
type Node interface{ TokenLiteral() string; String() string }
type Statement interface{ Node; statementNode() }
type Expression interface{ Node; expressionNode() }

type Program struct{ Statements []Statement }
func (p *Program) TokenLiteral() string { if len(p.Statements)>0 { return p.Statements[0].TokenLiteral() } return "" }
func (p *Program) String() string { var out string; for _, s:=range p.Statements { out+=s.String() }; return out }

// identifiers, integers, booleans, prefix, infix, if, fn, call, return, let

type Identifier struct{ Token Token; Value string }
func (i *Identifier) expressionNode(){ } func (i *Identifier) TokenLiteral()string{ return i.Token.Literal }
func (i *Identifier) String() string{ return i.Value }

type LetStatement struct{ Token Token; Name *Identifier; Value Expression }
func (ls *LetStatement) statementNode(){} func (ls *LetStatement) TokenLiteral()string{ return ls.Token.Literal }
func (ls *LetStatement) String() string{ return fmt.Sprintf("%s %s = %s;", ls.TokenLiteral(), ls.Name.String(), ls.Value.String()) }

type ReturnStatement struct{ Token Token; ReturnValue Expression }
func (rs *ReturnStatement) statementNode(){} func (rs *ReturnStatement) TokenLiteral()string{ return rs.Token.Literal }
func (rs *ReturnStatement) String() string{ return fmt.Sprintf("%s %s;", rs.TokenLiteral(), rs.ReturnValue.String()) }

type ExpressionStatement struct{ Token Token; Expression Expression }
func (es *ExpressionStatement) statementNode(){} func (es *ExpressionStatement) TokenLiteral()string{ return es.Token.Literal }
func (es *ExpressionStatement) String() string{ if es.Expression!=nil { return es.Expression.String() }; return "" }

type IntegerLiteral struct{ Token Token; Value int64 }
func (il *IntegerLiteral) expressionNode(){} func (il *IntegerLiteral) TokenLiteral()string{ return il.Token.Literal }
func (il *IntegerLiteral) String() string{ return il.Token.Literal }

type StringLiteral struct{ Token Token; Value string }
func (sl *StringLiteral) expressionNode(){} func (sl *StringLiteral) TokenLiteral()string{ return sl.Token.Literal }
func (sl *StringLiteral) String() string{ return sl.Token.Literal }

type Boolean struct{ Token Token; Value bool }
func (b *Boolean) expressionNode(){} func (b *Boolean) TokenLiteral()string{ return b.Token.Literal }
func (b *Boolean) String() string{ return b.Token.Literal }

type PrefixExpression struct{ Token Token; Operator string; Right Expression }
func (pe *PrefixExpression) expressionNode(){} func (pe *PrefixExpression) TokenLiteral()string{ return pe.Token.Literal }
func (pe *PrefixExpression) String() string{ return fmt.Sprintf("(%s%s)", pe.Operator, pe.Right.String()) }

type InfixExpression struct{ Token Token; Left Expression; Operator string; Right Expression }
func (ie *InfixExpression) expressionNode(){} func (ie *InfixExpression) TokenLiteral()string{ return ie.Token.Literal }
func (ie *InfixExpression) String() string{ return fmt.Sprintf("(%s %s %s)", ie.Left.String(), ie.Operator, ie.Right.String()) }

type BlockStatement struct{ Token Token; Statements []Statement }
func (bs *BlockStatement) statementNode(){} func (bs *BlockStatement) TokenLiteral()string{ return bs.Token.Literal }
func (bs *BlockStatement) String() string{ var out string; for _, s:=range bs.Statements{ out+=s.String() }; return out }

type IfExpression struct{ Token Token; Condition Expression; Consequence, Alternative *BlockStatement }
func (ie *IfExpression) expressionNode(){} func (ie *IfExpression) TokenLiteral()string{ return ie.Token.Literal }
func (ie *IfExpression) String() string{
	out := fmt.Sprintf("if%s %s", ie.Condition.String(), ie.Consequence.String())
	if ie.Alternative!=nil { out += fmt.Sprintf(" else %s", ie.Alternative.String()) }
	return out
}

type FunctionLiteral struct{ Token Token; Parameters []*Identifier; Body *BlockStatement }
func (fl *FunctionLiteral) expressionNode(){} func (fl *FunctionLiteral) TokenLiteral()string{ return fl.Token.Literal }
func (fl *FunctionLiteral) String() string{ var params []string; for _, p := range fl.Parameters { params = append(params, p.String()) }
	return fmt.Sprintf("%s(%s) %s", fl.TokenLiteral(), strings.Join(params, ","), fl.Body.String()) }

type CallExpression struct{ Token Token; Function Expression; Arguments []Expression }
func (ce *CallExpression) expressionNode(){} func (ce *CallExpression) TokenLiteral()string{ return ce.Token.Literal }
func (ce *CallExpression) String() string{ var args []string; for _, a := range ce.Arguments{ args = append(args, a.String()) }
	return fmt.Sprintf("%s(%s)", ce.Function.String(), strings.Join(args, ",")) }

// === Parser ===
type Parser struct{ l *Lexer; curTok, peekTok Token; errors []string }
func NewParser(l *Lexer) *Parser { p := &Parser{l:l}; p.nextToken(); p.nextToken(); return p }
func (p *Parser) nextToken() { p.curTok=p.peekTok; p.peekTok=p.l.NextToken() }
func (p *Parser) ParseProgram() *Program {
	prog := &Program{}
	for p.curTok.Type!=EOF {
		stmt := p.parseStatement()
		if stmt!=nil { prog.Statements = append(prog.Statements, stmt) }
		p.nextToken()
	}
	return prog
}
func (p *Parser) parseStatement() Statement {
	switch p.curTok.Type {
	case LET: return p.parseLetStatement()
	case RETURN: return p.parseReturnStatement()
	default: return p.parseExpressionStatement()
	}
}
func (p *Parser) parseLetStatement() *LetStatement {
	stmt := &LetStatement{Token:p.curTok}
	if !p.expectPeek(IDENT) { return nil }
	stmt.Name = &Identifier{Token:p.curTok, Value:p.curTok.Literal}
	if !p.expectPeek(ASSIGN) { return nil }
	p.nextToken(); stmt.Value = p.parseExpression(LOWEST)
	if p.peekTok.Type==SEMICOLON { p.nextToken() }
	return stmt
}
func (p *Parser) parseReturnStatement() *ReturnStatement {
	rs := &ReturnStatement{Token:p.curTok}
	p.nextToken(); rs.ReturnValue = p.parseExpression(LOWEST)
	if p.peekTok.Type==SEMICOLON { p.nextToken() }
	return rs
}
func (p *Parser) parseExpressionStatement() *ExpressionStatement {
	expr := &ExpressionStatement{Token:p.curTok}
	expr.Expression = p.parseExpression(LOWEST)
	if p.peekTok.Type==SEMICOLON { p.nextToken() }
	return expr
}
// Pratt parsing setup omitted for brevity, but supports prefix/infix
// ... (implementation continues with precedence, parsePrefix, parseInfix, parseIf, parseFunction, parseCall)
// === Evaluator ===
type ObjectType string
const (
	INTEGER_OBJ = "INTEGER"
	BOOLEAN_OBJ = "BOOLEAN"
	STRING_OBJ  = "STRING"
	NULL_OBJ    = "NULL"
	RETURN_OBJ  = "RETURN_VALUE"
	ERROR_OBJ   = "ERROR"
	FUNCTION_OBJ= "FUNCTION"
)

type Object interface{ Type() ObjectType; Inspect() string }

type Integer struct{ Value int64 }
func (i *Integer) Type() ObjectType { return INTEGER_OBJ }
func (i *Integer) Inspect() string { return fmt.Sprint(i.Value) }

type BooleanObj struct{ Value bool }
func (b *BooleanObj) Type() ObjectType { return BOOLEAN_OBJ }
func (b *BooleanObj) Inspect() string { return fmt.Sprint(b.Value) }

type StringObj struct{ Value string }
func (s *StringObj) Type() ObjectType { return STRING_OBJ }
func (s *StringObj) Inspect() string { return s.Value }

type Null struct{} 
func (n *Null) Type() ObjectType { return NULL_OBJ }
func (n *Null) Inspect() string { return "null" }

type ReturnValue struct{ Value Object }
func (rv *ReturnValue) Type() ObjectType { return RETURN_OBJ }
func (rv *ReturnValue) Inspect() string { return rv.Value.Inspect() }

type ErrorObj struct{ Message string }
func (e *ErrorObj) Type() ObjectType { return ERROR_OBJ }
func (e *ErrorObj) Inspect() string { return "ERROR: " + e.Message }

type Environment struct{ store map[string]Object; outer *Environment }
func NewEnv() *Environment { return &Environment{store:make(map[string]Object)} }
func (e *Environment) Get(name string) (Object, bool) {
	obj, ok := e.store[name]
	if !ok && e.outer!=nil { obj, ok = e.outer.Get(name) }
	return obj, ok
}
func (e *Environment) Set(name string, val Object) Object { e.store[name] = val; return val }

// eval functions omitted for brevity: evalProgram, evalBlock, evalExpressions, evalPrefix, evalInfix, evalIf, evalIdentifier, applyFunction
// Builtins: len, print

func Eval(node Node, env *Environment) Object {
	switch n := node.(type) {
	case *Program:
		return evalProgram(n, env)
	case *ExpressionStatement:
		return Eval(n.Expression, env)
	case *IntegerLiteral:
		return &Integer{Value: n.Value}
	case *Boolean:
		return nativeBoolToBooleanObject(n.Value)
	case *StringLiteral:
		return &StringObj{Value: n.Value}
	// ... other node types
	}
	return nil
}

func main() {
	reader := bufio.NewReader(os.Stdin)
	env := NewEnv()
	fmt.Println("Welcome to GoScript REPL. Type 'exit' to quit.")
	for {
		fmt.Print("> ")
		line, err := reader.ReadString('\n')
		if err == io.EOF { break }
		if strings.TrimSpace(line) == "exit" { break }
		lexer := NewLexer(line)
		parser := NewParser(lexer)
		program := parser.ParseProgram()
		result := Eval(program, env)
		if result != nil {
			fmt.Println(result.Inspect())
		}
	}
}

