package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode"
)

// Token types
type TokenType int

const (
	// Literals
	NUMBER TokenType = iota
	STRING
	IDENTIFIER
	BOOLEAN

	// Operators
	PLUS
	MINUS
	MULTIPLY
	DIVIDE
	MODULO
	ASSIGN
	EQUAL
	NOT_EQUAL
	LESS
	GREATER
	LESS_EQUAL
	GREATER_EQUAL

	// Delimiters
	LPAREN
	RPAREN
	LBRACE
	RBRACE
	LBRACKET
	RBRACKET
	COMMA
	DOT
	SEMICOLON
	NEWLINE

	// Keywords
	IF
	ELSE
	WHILE
	FOR
	FUNCTION
	RETURN
	LET
	TRUE
	FALSE
	PRINT
	INPUT
	TYPE
	AND
	OR
	NOT

	// Special
	EOF
	ILLEGAL
)

// Token represents a single token
type Token struct {
	Type    TokenType
	Literal string
	Line    int
	Column  int
}

// Lexer for tokenizing input
type Lexer struct {
	input        string
	position     int
	readPosition int
	ch           byte
	line         int
	column       int
}

func NewLexer(input string) *Lexer {
	l := &Lexer{
		input:  input,
		line:   1,
		column: 0,
	}
	l.readChar()
	return l
}

func (l *Lexer) readChar() {
	if l.readPosition >= len(l.input) {
		l.ch = 0
	} else {
		l.ch = l.input[l.readPosition]
	}
	l.position = l.readPosition
	l.readPosition++
	if l.ch == '\n' {
		l.line++
		l.column = 0
	} else {
		l.column++
	}
}

func (l *Lexer) peekChar() byte {
	if l.readPosition >= len(l.input) {
		return 0
	}
	return l.input[l.readPosition]
}

func (l *Lexer) skipWhitespace() {
	for l.ch == ' ' || l.ch == '\t' || l.ch == '\r' {
		l.readChar()
	}
}

func (l *Lexer) readString() string {
	position := l.position + 1
	for {
		l.readChar()
		if l.ch == '"' || l.ch == 0 {
			break
		}
	}
	return l.input[position:l.position]
}

func (l *Lexer) readNumber() string {
	position := l.position
	hasDecimal := false
	
	for unicode.IsDigit(rune(l.ch)) || (l.ch == '.' && !hasDecimal) {
		if l.ch == '.' {
			hasDecimal = true
		}
		l.readChar()
	}
	return l.input[position:l.position]
}

func (l *Lexer) readIdentifier() string {
	position := l.position
	for unicode.IsLetter(rune(l.ch)) || unicode.IsDigit(rune(l.ch)) || l.ch == '_' {
		l.readChar()
	}
	return l.input[position:l.position]
}

func lookupIdent(ident string) TokenType {
	keywords := map[string]TokenType{
		"if":       IF,
		"else":     ELSE,
		"while":    WHILE,
		"for":      FOR,
		"function": FUNCTION,
		"return":   RETURN,
		"let":      LET,
		"true":     TRUE,
		"false":    FALSE,
		"print":    PRINT,
		"input":    INPUT,
		"type":     TYPE,
		"and":      AND,
		"or":       OR,
		"not":      NOT,
	}
	
	if tok, ok := keywords[ident]; ok {
		return tok
	}
	return IDENTIFIER
}

func (l *Lexer) NextToken() Token {
	var tok Token
	
	l.skipWhitespace()
	
	tok.Line = l.line
	tok.Column = l.column
	
	switch l.ch {
	case '+':
		tok = Token{Type: PLUS, Literal: string(l.ch)}
	case '-':
		tok = Token{Type: MINUS, Literal: string(l.ch)}
	case '*':
		tok = Token{Type: MULTIPLY, Literal: string(l.ch)}
	case '/':
		tok = Token{Type: DIVIDE, Literal: string(l.ch)}
	case '%':
		tok = Token{Type: MODULO, Literal: string(l.ch)}
	case '=':
		if l.peekChar() == '=' {
			ch := l.ch
			l.readChar()
			tok = Token{Type: EQUAL, Literal: string(ch) + string(l.ch)}
		} else {
			tok = Token{Type: ASSIGN, Literal: string(l.ch)}
		}
	case '!':
		if l.peekChar() == '=' {
			ch := l.ch
			l.readChar()
			tok = Token{Type: NOT_EQUAL, Literal: string(ch) + string(l.ch)}
		} else {
			tok = Token{Type: NOT, Literal: string(l.ch)}
		}
	case '<':
		if l.peekChar() == '=' {
			ch := l.ch
			l.readChar()
			tok = Token{Type: LESS_EQUAL, Literal: string(ch) + string(l.ch)}
		} else {
			tok = Token{Type: LESS, Literal: string(l.ch)}
		}
	case '>':
		if l.peekChar() == '=' {
			ch := l.ch
			l.readChar()
			tok = Token{Type: GREATER_EQUAL, Literal: string(ch) + string(l.ch)}
		} else {
			tok = Token{Type: GREATER, Literal: string(l.ch)}
		}
	case '(':
		tok = Token{Type: LPAREN, Literal: string(l.ch)}
	case ')':
		tok = Token{Type: RPAREN, Literal: string(l.ch)}
	case '{':
		tok = Token{Type: LBRACE, Literal: string(l.ch)}
	case '}':
		tok = Token{Type: RBRACE, Literal: string(l.ch)}
	case '[':
		tok = Token{Type: LBRACKET, Literal: string(l.ch)}
	case ']':
		tok = Token{Type: RBRACKET, Literal: string(l.ch)}
	case ',':
		tok = Token{Type: COMMA, Literal: string(l.ch)}
	case '.':
		if unicode.IsDigit(rune(l.peekChar())) {
			tok.Type = NUMBER
			tok.Literal = l.readNumber()
			return tok
		}
		tok = Token{Type: DOT, Literal: string(l.ch)}
	case ';':
		tok = Token{Type: SEMICOLON, Literal: string(l.ch)}
	case '\n':
		tok = Token{Type: NEWLINE, Literal: string(l.ch)}
	case '"':
		tok.Type = STRING
		tok.Literal = l.readString()
	case 0:
		tok.Literal = ""
		tok.Type = EOF
		return tok
	default:
		if unicode.IsLetter(rune(l.ch)) || l.ch == '_' {
			tok.Literal = l.readIdentifier()
			tok.Type = lookupIdent(tok.Literal)
			return tok
		} else if unicode.IsDigit(rune(l.ch)) {
			tok.Type = NUMBER
			tok.Literal = l.readNumber()
			return tok
		} else {
			tok = Token{Type: ILLEGAL, Literal: string(l.ch)}
		}
	}
	
	l.readChar()
	return tok
}

// AST Node interfaces
type Node interface {
	String() string
}

type Statement interface {
	Node
	statementNode()
}

type Expression interface {
	Node
	expressionNode()
}

// Statements
type LetStatement struct {
	Name  string
	Value Expression
}

func (ls *LetStatement) statementNode() {}
func (ls *LetStatement) String() string {
	return fmt.Sprintf("let %s = %s", ls.Name, ls.Value.String())
}

type ExpressionStatement struct {
	Expression Expression
}

func (es *ExpressionStatement) statementNode() {}
func (es *ExpressionStatement) String() string {
	return es.Expression.String()
}

type BlockStatement struct {
	Statements []Statement
}

func (bs *BlockStatement) statementNode() {}
func (bs *BlockStatement) String() string {
	var out strings.Builder
	for _, stmt := range bs.Statements {
		out.WriteString(stmt.String())
	}
	return out.String()
}

type IfStatement struct {
	Condition Expression
	ThenBranch Statement
	ElseBranch Statement
}

func (is *IfStatement) statementNode() {}
func (is *IfStatement) String() string {
	return fmt.Sprintf("if (%s) %s else %s", is.Condition.String(), is.ThenBranch.String(), is.ElseBranch.String())
}

type WhileStatement struct {
	Condition Expression
	Body      Statement
}

func (ws *WhileStatement) statementNode() {}
func (ws *WhileStatement) String() string {
	return fmt.Sprintf("while (%s) %s", ws.Condition.String(), ws.Body.String())
}

type FunctionStatement struct {
	Name       string
	Parameters []string
	Body       *BlockStatement
}

func (fs *FunctionStatement) statementNode() {}
func (fs *FunctionStatement) String() string {
	return fmt.Sprintf("function %s(%s) %s", fs.Name, strings.Join(fs.Parameters, ", "), fs.Body.String())
}

type ReturnStatement struct {
	Value Expression
}

func (rs *ReturnStatement) statementNode() {}
func (rs *ReturnStatement) String() string {
	if rs.Value != nil {
		return fmt.Sprintf("return %s", rs.Value.String())
	}
	return "return"
}

type PrintStatement struct {
	Value Expression
}

func (ps *PrintStatement) statementNode() {}
func (ps *PrintStatement) String() string {
	return fmt.Sprintf("print %s", ps.Value.String())
}

// Expressions
type Identifier struct {
	Value string
}

func (i *Identifier) expressionNode() {}
func (i *Identifier) String() string { return i.Value }

type NumberLiteral struct {
	Value float64
}

func (nl *NumberLiteral) expressionNode() {}
func (nl *NumberLiteral) String() string { return fmt.Sprintf("%g", nl.Value) }

type StringLiteral struct {
	Value string
}

func (sl *StringLiteral) expressionNode() {}
func (sl *StringLiteral) String() string { return fmt.Sprintf("\"%s\"", sl.Value) }

type BooleanLiteral struct {
	Value bool
}

func (bl *BooleanLiteral) expressionNode() {}
func (bl *BooleanLiteral) String() string { return fmt.Sprintf("%t", bl.Value) }

type InfixExpression struct {
	Left     Expression
	Operator string
	Right    Expression
}

func (ie *InfixExpression) expressionNode() {}
func (ie *InfixExpression) String() string {
	return fmt.Sprintf("(%s %s %s)", ie.Left.String(), ie.Operator, ie.Right.String())
}

type PrefixExpression struct {
	Operator string
	Right    Expression
}

func (pe *PrefixExpression) expressionNode() {}
func (pe *PrefixExpression) String() string {
	return fmt.Sprintf("(%s%s)", pe.Operator, pe.Right.String())
}

type CallExpression struct {
	Function  Expression
	Arguments []Expression
}

func (ce *CallExpression) expressionNode() {}
func (ce *CallExpression) String() string {
	args := make([]string, len(ce.Arguments))
	for i, arg := range ce.Arguments {
		args[i] = arg.String()
	}
	return fmt.Sprintf("%s(%s)", ce.Function.String(), strings.Join(args, ", "))
}

type ArrayLiteral struct {
	Elements []Expression
}

func (al *ArrayLiteral) expressionNode() {}
func (al *ArrayLiteral) String() string {
	elements := make([]string, len(al.Elements))
	for i, elem := range al.Elements {
		elements[i] = elem.String()
	}
	return fmt.Sprintf("[%s]", strings.Join(elements, ", "))
}

type IndexExpression struct {
	Left  Expression
	Index Expression
}

func (ie *IndexExpression) expressionNode() {}
func (ie *IndexExpression) String() string {
	return fmt.Sprintf("(%s[%s])", ie.Left.String(), ie.Index.String())
}

type InputExpression struct{}

func (ie *InputExpression) expressionNode() {}
func (ie *InputExpression) String() string { return "input()" }

type TypeExpression struct {
	Value Expression
}

func (te *TypeExpression) expressionNode() {}
func (te *TypeExpression) String() string {
	return fmt.Sprintf("type(%s)", te.Value.String())
}

// Parser
type Parser struct {
	lexer *Lexer
	
	curToken  Token
	peekToken Token
	
	errors []string
}

func NewParser(lexer *Lexer) *Parser {
	p := &Parser{lexer: lexer}
	
	// Read two tokens, so curToken and peekToken are both set
	p.nextToken()
	p.nextToken()
	
	return p
}

func (p *Parser) nextToken() {
	p.curToken = p.peekToken
	p.peekToken = p.lexer.NextToken()
}

func (p *Parser) Errors() []string {
	return p.errors
}

func (p *Parser) addError(msg string) {
	p.errors = append(p.errors, fmt.Sprintf("Line %d: %s", p.curToken.Line, msg))
}

func (p *Parser) expectPeek(t TokenType) bool {
	if p.peekToken.Type == t {
		p.nextToken()
		return true
	}
	p.addError(fmt.Sprintf("expected next token to be %v, got %v instead", t, p.peekToken.Type))
	return false
}

func (p *Parser) ParseProgram() []Statement {
	statements := []Statement{}
	
	for p.curToken.Type != EOF {
		if p.curToken.Type == NEWLINE {
			p.nextToken()
			continue
		}
		
		stmt := p.parseStatement()
		if stmt != nil {
			statements = append(statements, stmt)
		}
		p.nextToken()
	}
	
	return statements
}

func (p *Parser) parseStatement() Statement {
	switch p.curToken.Type {
	case LET:
		return p.parseLetStatement()
	case IF:
		return p.parseIfStatement()
	case WHILE:
		return p.parseWhileStatement()
	case FUNCTION:
		return p.parseFunctionStatement()
	case RETURN:
		return p.parseReturnStatement()
	case PRINT:
		return p.parsePrintStatement()
	case LBRACE:
		return p.parseBlockStatement()
	default:
		return p.parseExpressionStatement()
	}
}

func (p *Parser) parseLetStatement() *LetStatement {
	stmt := &LetStatement{}
	
	if !p.expectPeek(IDENTIFIER) {
		return nil
	}
	
	stmt.Name = p.curToken.Literal
	
	if !p.expectPeek(ASSIGN) {
		return nil
	}
	
	p.nextToken()
	stmt.Value = p.parseExpression(LOWEST)
	
	if p.peekToken.Type == SEMICOLON {
		p.nextToken()
	}
	
	return stmt
}

func (p *Parser) parseIfStatement() *IfStatement {
	stmt := &IfStatement{}
	
	if !p.expectPeek(LPAREN) {
		return nil
	}
	
	p.nextToken()
	stmt.Condition = p.parseExpression(LOWEST)
	
	if !p.expectPeek(RPAREN) {
		return nil
	}
	
	p.nextToken()
	stmt.ThenBranch = p.parseStatement()
	
	if p.peekToken.Type == ELSE {
		p.nextToken()
		p.nextToken()
		stmt.ElseBranch = p.parseStatement()
	}
	
	return stmt
}

func (p *Parser) parseWhileStatement() *WhileStatement {
	stmt := &WhileStatement{}
	
	if !p.expectPeek(LPAREN) {
		return nil
	}
	
	p.nextToken()
	stmt.Condition = p.parseExpression(LOWEST)
	
	if !p.expectPeek(RPAREN) {
		return nil
	}
	
	p.nextToken()
	stmt.Body = p.parseStatement()
	
	return stmt
}

func (p *Parser) parseFunctionStatement() *FunctionStatement {
	stmt := &FunctionStatement{}
	
	if !p.expectPeek(IDENTIFIER) {
		return nil
	}
	
	stmt.Name = p.curToken.Literal
	
	if !p.expectPeek(LPAREN) {
		return nil
	}
	
	stmt.Parameters = p.parseFunctionParameters()
	
	if !p.expectPeek(LBRACE) {
		return nil
	}
	
	stmt.Body = p.parseBlockStatement()
	
	return stmt
}

func (p *Parser) parseFunctionParameters() []string {
	identifiers := []string{}
	
	if p.peekToken.Type == RPAREN {
		p.nextToken()
		return identifiers
	}
	
	p.nextToken()
	identifiers = append(identifiers, p.curToken.Literal)
	
	for p.peekToken.Type == COMMA {
		p.nextToken()
		p.nextToken()
		identifiers = append(identifiers, p.curToken.Literal)
	}
	
	if !p.expectPeek(RPAREN) {
		return nil
	}
	
	return identifiers
}

func (p *Parser) parseReturnStatement() *ReturnStatement {
	stmt := &ReturnStatement{}
	
	p.nextToken()
	
	if p.curToken.Type == NEWLINE || p.curToken.Type == SEMICOLON {
		return stmt
	}
	
	stmt.Value = p.parseExpression(LOWEST)
	
	if p.peekToken.Type == SEMICOLON {
		p.nextToken()
	}
	
	return stmt
}

func (p *Parser) parsePrintStatement() *PrintStatement {
	stmt := &PrintStatement{}
	
	p.nextToken()
	stmt.Value = p.parseExpression(LOWEST)
	
	if p.peekToken.Type == SEMICOLON {
		p.nextToken()
	}
	
	return stmt
}

func (p *Parser) parseBlockStatement() *BlockStatement {
	block := &BlockStatement{}
	block.Statements = []Statement{}
	
	p.nextToken()
	
	for p.curToken.Type != RBRACE && p.curToken.Type != EOF {
		if p.curToken.Type == NEWLINE {
			p.nextToken()
			continue
		}
		
		stmt := p.parseStatement()
		if stmt != nil {
			block.Statements = append(block.Statements, stmt)
		}
		p.nextToken()
	}
	
	return block
}

func (p *Parser) parseExpressionStatement() *ExpressionStatement {
	stmt := &ExpressionStatement{}
	stmt.Expression = p.parseExpression(LOWEST)
	
	if p.peekToken.Type == SEMICOLON {
		p.nextToken()
	}
	
	return stmt
}

// Expression parsing with precedence
type Precedence int

const (
	_ Precedence = iota
	LOWEST
	LOGICAL    // and, or
	EQUALS     // ==, !=
	LESSGREATER // > or <
	SUM        // +
	PRODUCT    // *
	PREFIX     // -X or !X
	CALL       // myFunction(X)
	INDEX      // array[index]
)

var precedences = map[TokenType]Precedence{
	AND:           LOGICAL,
	OR:            LOGICAL,
	EQUAL:        EQUALS,
	NOT_EQUAL:    EQUALS,
	LESS:         LESSGREATER,
	GREATER:      LESSGREATER,
	LESS_EQUAL:   LESSGREATER,
	GREATER_EQUAL: LESSGREATER,
	PLUS:         SUM,
	MINUS:        SUM,
	DIVIDE:       PRODUCT,
	MULTIPLY:     PRODUCT,
	MODULO:       PRODUCT,
	LPAREN:       CALL,
	LBRACKET:     INDEX,
}

func (p *Parser) peekPrecedence() Precedence {
	if p, ok := precedences[p.peekToken.Type]; ok {
		return p
	}
	return LOWEST
}

func (p *Parser) curPrecedence() Precedence {
	if p, ok := precedences[p.curToken.Type]; ok {
		return p
	}
	return LOWEST
}

func (p *Parser) parseExpression(precedence Precedence) Expression {
	var left Expression
	
	switch p.curToken.Type {
	case IDENTIFIER:
		left = &Identifier{Value: p.curToken.Literal}
	case NUMBER:
		val, err := strconv.ParseFloat(p.curToken.Literal, 64)
		if err != nil {
			p.addError(fmt.Sprintf("could not parse %q as float", p.curToken.Literal))
			return nil
		}
		left = &NumberLiteral{Value: val}
	case STRING:
		left = &StringLiteral{Value: p.curToken.Literal}
	case TRUE:
		left = &BooleanLiteral{Value: true}
	case FALSE:
		left = &BooleanLiteral{Value: false}
	case MINUS, NOT:
		operator := p.curToken.Literal
		p.nextToken()
		right := p.parseExpression(PREFIX)
		left = &PrefixExpression{Operator: operator, Right: right}
	case LPAREN:
		p.nextToken()
		left = p.parseExpression(LOWEST)
		if !p.expectPeek(RPAREN) {
			return nil
		}
	case LBRACKET:
		left = p.parseArrayLiteral()
	case INPUT:
		left = &InputExpression{}
	case TYPE:
		if !p.expectPeek(LPAREN) {
			return nil
		}
		p.nextToken()
		expr := p.parseExpression(LOWEST)
		if !p.expectPeek(RPAREN) {
			return nil
		}
		left = &TypeExpression{Value: expr}
	default:
		p.addError(fmt.Sprintf("no prefix parse function for %v found", p.curToken.Type))
		return nil
	}
	
	for p.peekToken.Type != SEMICOLON && precedence < p.peekPrecedence() {
		switch p.peekToken.Type {
		case PLUS, MINUS, DIVIDE, MULTIPLY, MODULO, EQUAL, NOT_EQUAL, LESS, GREATER, LESS_EQUAL, GREATER_EQUAL, AND, OR:
			p.nextToken()
			left = p.parseInfixExpression(left)
		case LPAREN:
			p.nextToken()
			left = p.parseCallExpression(left)
		case LBRACKET:
			p.nextToken()
			left = p.parseIndexExpression(left)
		default:
			return left
		}
	}
	
	return left
}

func (p *Parser) parseInfixExpression(left Expression) Expression {
	expression := &InfixExpression{
		Left:     left,
		Operator: p.curToken.Literal,
	}
	
	precedence := p.curPrecedence()
	p.nextToken()
	expression.Right = p.parseExpression(precedence)
	
	return expression
}

func (p *Parser) parseCallExpression(fn Expression) Expression {
	exp := &CallExpression{Function: fn}
	exp.Arguments = p.parseExpressionList(RPAREN)
	return exp
}

func (p *Parser) parseArrayLiteral() Expression {
	array := &ArrayLiteral{}
	array.Elements = p.parseExpressionList(RBRACKET)
	return array
}

func (p *Parser) parseIndexExpression(left Expression) Expression {
	exp := &IndexExpression{Left: left}
	
	p.nextToken()
	exp.Index = p.parseExpression(LOWEST)
	
	if !p.expectPeek(RBRACKET) {
		return nil
	}
	
	return exp
}

func (p *Parser) parseExpressionList(end TokenType) []Expression {
	args := []Expression{}
	
	if p.peekToken.Type == end {
		p.nextToken()
		return args
	}
	
	p.nextToken()
	args = append(args, p.parseExpression(LOWEST))
	
	for p.peekToken.Type == COMMA {
		p.nextToken()
		p.nextToken()
		args = append(args, p.parseExpression(LOWEST))
	}
	
	if !p.expectPeek(end) {
		return nil
	}
	
	return args
}

// Value types for the interpreter
type ValueType string

const (
	NUMBER_VAL   ValueType = "NUMBER"
	STRING_VAL   ValueType = "STRING"
	BOOLEAN_VAL  ValueType = "BOOLEAN"
	ARRAY_VAL    ValueType = "ARRAY"
	FUNCTION_VAL ValueType = "FUNCTION"
	RETURN_VAL   ValueType = "RETURN"
	NULL_VAL     ValueType = "NULL"
)

type Value interface {
	Type() ValueType
	String() string
}

type NumberValue struct {
	Value float64
}

func (nv *NumberValue) Type() ValueType { return NUMBER_VAL }
func (nv *NumberValue) String() string  { return fmt.Sprintf("%g", nv.Value) }

type StringValue struct {
	Value string
}

func (sv *StringValue) Type() ValueType { return STRING_VAL }
func (sv *StringValue) String() string  { return sv.Value }

type BooleanValue struct {
	Value bool
}

func (bv *BooleanValue) Type() ValueType { return BOOLEAN_VAL }
func (bv *BooleanValue) String() string  { return fmt.Sprintf("%t", bv.Value) }

type ArrayValue struct {
	Elements []Value
}

func (av *ArrayValue) Type() ValueType { return ARRAY_VAL }
func (av *ArrayValue) String() string {
	var elements []string
	for _, elem := range av.Elements {
		elements = append(elements, elem.String())
	}
	return fmt.Sprintf("[%s]", strings.Join(elements, ", "))
}

type FunctionValue struct {
	Parameters   []string
	Body         *BlockStatement
	Environment  *Environment
}

func (fv *FunctionValue) Type() ValueType { return FUNCTION_VAL }
func (fv *FunctionValue) String() string {
	return fmt.Sprintf("function(%s) { ... }", strings.Join(fv.Parameters, ", "))
}

type ReturnValue struct {
	Value Value
}

func (rv *ReturnValue) Type() ValueType { return RETURN_VAL }
func (rv *ReturnValue) String() string  { return rv.Value.String() }

type NullValue struct{}

func (nv *NullValue) Type() ValueType { return NULL_VAL }
func (nv *NullValue) String() string  { return "null" }

// Environment for variable storage
type Environment struct {
	store map[string]Value
	outer *Environment
}

func NewEnvironment() *Environment {
	return &Environment{
		store: make(map[string]Value),
		outer: nil,
	}
}

func NewEnclosedEnvironment(outer *Environment) *Environment {
	env := NewEnvironment()
	env.outer = outer
	return env
}

func (e *Environment) Get(name string) (Value, bool) {
	value, ok := e.store[name]
	if !ok && e.outer != nil {
		value, ok = e.outer.Get(name)
	}
	return value, ok
}

func (e *Environment) Set(name string, val Value) Value {
	e.store[name] = val
	return val
}

// Interpreter
type Interpreter struct {
	errors []string
}

func NewInterpreter() *Interpreter {
	return &Interpreter{}
}

func (i *Interpreter) Errors() []string {
	return i.errors
}

func (i *Interpreter) addError(msg string) {
	i.errors = append(i.errors, msg)
}

func (i *Interpreter) Eval(node Node, env *Environment) Value {
	switch node := node.(type) {
	
	// Statements
	case *LetStatement:
		val := i.Eval(node.Value, env)
		if i.isError(val) {
			return val
		}
		env.Set(node.Name, val)
		return &NullValue{}
		
	case *ExpressionStatement:
		return i.Eval(node.Expression, env)
		
	case *BlockStatement:
		return i.evalBlockStatement(node, env)
		
	case *IfStatement:
		return i.evalIfStatement(node, env)
		
	case *WhileStatement:
		return i.evalWhileStatement(node, env)
		
	case *FunctionStatement:
		fn := &FunctionValue{
			Parameters:  node.Parameters,
			Body:        node.Body,
			Environment: env,
		}
		env.Set(node.Name, fn)
		return &NullValue{}
		
	case *ReturnStatement:
		val := &NullValue{}
		if node.Value != nil {
			val = i.Eval(node.Value, env)
			if i.isError(val) {
				return val
			}
		}
		return &ReturnValue{Value: val}
		
	case *PrintStatement:
		val := i.Eval(node.Value, env)
		if i.isError(val) {
			return val
		}
		fmt.Println(val.String())
		return &NullValue{}
	
	// Expressions
	case *NumberLiteral:
		return &NumberValue{Value: node.Value}
		
	case *StringLiteral:
		return &StringValue{Value: node.Value}
		
	case *BooleanLiteral:
		return &BooleanValue{Value: node.Value}
		
	case *Identifier:
		return i.evalIdentifier(node, env)
		
	case *PrefixExpression:
		right := i.Eval(node.Right, env)
		if i.isError(right) {
			return right
		}
		return i.evalPrefixExpression(node.Operator, right)
		
	case *InfixExpression:
		left := i.Eval(node.Left, env)
		if i.isError(left) {
			return left
		}
		right := i.Eval(node.Right, env)
		if i.isError(right) {
			return right
		}
		return i.evalInfixExpression(node.Operator, left, right)
		
	case *CallExpression:
		function := i.Eval(node.Function, env)
		if i.isError(function) {
			return function
		}
		args := i.evalExpressions(node.Arguments, env)
		if len(args) == 1 && i.isError(args[0]) {
			return args[0]
		}
		return i.applyFunction(function, args)
		
	case *ArrayLiteral:
		elements := i.evalExpressions(node.Elements, env)
		if len(elements) == 1 && i.isError(elements[0]) {
			return elements[0]
		}
		return &ArrayValue{Elements: elements}
		
	case *IndexExpression:
		left := i.Eval(node.Left, env)
		if i.isError(left) {
			return left
		}
		index := i.Eval(node.Index, env)
		if i.isError(index) {
			return index
		}
		return i.evalIndexExpression(left, index)
		
	case *InputExpression:
		reader := bufio.NewReader(os.Stdin)
		text, _ := reader.ReadString('\n')
		text = strings.TrimSpace(text)
		return &StringValue{Value: text}
		
	case *TypeExpression:
		val := i.Eval(node.Value, env)
		if i.isError(val) {
			return val
		}
		return &StringValue{Value: string(val.Type())}
	}
	
	return i.newError("unknown node type: %T", node)
}

func (i *Interpreter) evalBlockStatement(block *BlockStatement, env *Environment) Value {
	var result Value
	
	for _, statement := range block.Statements {
		result = i.Eval(statement, env)
		
		if result != nil {
			rt := result.Type()
			if rt == RETURN_VAL {
				return result
			}
		}
	}
	
	return result
}

func (i *Interpreter) evalIfStatement(ie *IfStatement, env *Environment) Value {
	condition := i.Eval(ie.Condition, env)
	if i.isError(condition) {
		return condition
	}
	
	if i.isTruthy(condition) {
		return i.Eval(ie.ThenBranch, env)
	} else if ie.ElseBranch != nil {
		return i.Eval(ie.ElseBranch, env)
	} else {
		return &NullValue{}
	}
}

func (i *Interpreter) evalWhileStatement(ws *WhileStatement, env *Environment) Value {
	var result Value = &NullValue{}
	
	for {
		condition := i.Eval(ws.Condition, env)
		if i.isError(condition) {
			return condition
		}
		
		if !i.isTruthy(condition) {
			break
		}
		
		result = i.Eval(ws.Body, env)
		if result != nil && result.Type() == RETURN_VAL {
			return result
		}
	}
	
	return result
}

func (i *Interpreter) evalIdentifier(node *Identifier, env *Environment) Value {
	val, ok := env.Get(node.Value)
	if !ok {
		return i.newError("identifier not found: " + node.Value)
	}
	return val
}

func (i *Interpreter) evalPrefixExpression(operator string, right Value) Value {
	switch operator {
	case "!":
		return i.evalBangOperatorExpression(right)
	case "-":
		return i.evalMinusPrefixOperatorExpression(right)
	default:
		return i.newError("unknown operator: %s%s", operator, right.Type())
	}
}

func (i *Interpreter) evalBangOperatorExpression(right Value) Value {
	switch right {
	case &BooleanValue{Value: true}:
		return &BooleanValue{Value: false}
	case &BooleanValue{Value: false}:
		return &BooleanValue{Value: true}
	case &NullValue{}:
		return &BooleanValue{Value: true}
	default:
		return &BooleanValue{Value: false}
	}
}

func (i *Interpreter) evalMinusPrefixOperatorExpression(right Value) Value {
	if right.Type() != NUMBER_VAL {
		return i.newError("unknown operator: -%s", right.Type())
	}
	
	value := right.(*NumberValue).Value
	return &NumberValue{Value: -value}
}

func (i *Interpreter) evalInfixExpression(operator string, left, right Value) Value {
	switch {
	case left.Type() == NUMBER_VAL && right.Type() == NUMBER_VAL:
		return i.evalNumberInfixExpression(operator, left, right)
	case left.Type() == STRING_VAL && right.Type() == STRING_VAL:
		return i.evalStringInfixExpression(operator, left, right)
	case left.Type() == BOOLEAN_VAL && right.Type() == BOOLEAN_VAL:
		return i.evalBooleanInfixExpression(operator, left, right)
	case operator == "==":
		return &BooleanValue{Value: left == right}
	case operator == "!=":
		return &BooleanValue{Value: left != right}
	case operator == "and":
		return &BooleanValue{Value: i.isTruthy(left) && i.isTruthy(right)}
	case operator == "or":
		return &BooleanValue{Value: i.isTruthy(left) || i.isTruthy(right)}
	default:
		return i.newError("unknown operator: %s %s %s", left.Type(), operator, right.Type())
	}
}

func (i *Interpreter) evalNumberInfixExpression(operator string, left, right Value) Value {
	leftVal := left.(*NumberValue).Value
	rightVal := right.(*NumberValue).Value
	
	switch operator {
	case "+":
		return &NumberValue{Value: leftVal + rightVal}
	case "-":
		return &NumberValue{Value: leftVal - rightVal}
	case "*":
		return &NumberValue{Value: leftVal * rightVal}
	case "/":
		if rightVal == 0 {
			return i.newError("division by zero")
		}
		return &NumberValue{Value: leftVal / rightVal}
	case "%":
		if rightVal == 0 {
			return i.newError("modulo by zero")
		}
		return &NumberValue{Value: float64(int(leftVal) % int(rightVal))}
	case "<":
		return &BooleanValue{Value: leftVal < rightVal}
	case ">":
		return &BooleanValue{Value: leftVal > rightVal}
	case "<=":
		return &BooleanValue{Value: leftVal <= rightVal}
	case ">=":
		return &BooleanValue{Value: leftVal >= rightVal}
	case "==":
		return &BooleanValue{Value: leftVal == rightVal}
	case "!=":
		return &BooleanValue{Value: leftVal != rightVal}
	default:
		return i.newError("unknown operator: %s", operator)
	}
}

func (i *Interpreter) evalStringInfixExpression(operator string, left, right Value) Value {
	leftVal := left.(*StringValue).Value
	rightVal := right.(*StringValue).Value
	
	switch operator {
	case "+":
		return &StringValue{Value: leftVal + rightVal}
	case "==":
		return &BooleanValue{Value: leftVal == rightVal}
	case "!=":
		return &BooleanValue{Value: leftVal != rightVal}
	default:
		return i.newError("unknown operator: %s", operator)
	}
}

func (i *Interpreter) evalBooleanInfixExpression(operator string, left, right Value) Value {
	leftVal := left.(*BooleanValue).Value
	rightVal := right.(*BooleanValue).Value
	
	switch operator {
	case "==":
		return &BooleanValue{Value: leftVal == rightVal}
	case "!=":
		return &BooleanValue{Value: leftVal != rightVal}
	case "and":
		return &BooleanValue{Value: leftVal && rightVal}
	case "or":
		return &BooleanValue{Value: leftVal || rightVal}
	default:
		return i.newError("unknown operator: %s", operator)
	}
}

func (i *Interpreter) evalExpressions(exps []Expression, env *Environment) []Value {
	var result []Value
	
	for _, e := range exps {
		evaluated := i.Eval(e, env)
		if i.isError(evaluated) {
			return []Value{evaluated}
		}
		result = append(result, evaluated)
	}
	
	return result
}

func (i *Interpreter) applyFunction(fn Value, args []Value) Value {
	function, ok := fn.(*FunctionValue)
	if !ok {
		return i.newError("not a function: %T", fn)
	}
	
	extendedEnv := i.extendFunctionEnv(function, args)
	evaluated := i.Eval(function.Body, extendedEnv)
	return i.unwrapReturnValue(evaluated)
}

func (i *Interpreter) extendFunctionEnv(fn *FunctionValue, args []Value) *Environment {
	env := NewEnclosedEnvironment(fn.Environment)
	
	for paramIdx, param := range fn.Parameters {
		if paramIdx < len(args) {
			env.Set(param, args[paramIdx])
		} else {
			env.Set(param, &NullValue{})
		}
	}
	
	return env
}

func (i *Interpreter) unwrapReturnValue(obj Value) Value {
	if returnValue, ok := obj.(*ReturnValue); ok {
		return returnValue.Value
	}
	return obj
}

func (i *Interpreter) evalIndexExpression(left, index Value) Value {
	switch {
	case left.Type() == ARRAY_VAL && index.Type() == NUMBER_VAL:
		return i.evalArrayIndexExpression(left, index)
	case left.Type() == STRING_VAL && index.Type() == NUMBER_VAL:
		return i.evalStringIndexExpression(left, index)
	default:
		return i.newError("index operator not supported: %s", left.Type())
	}
}

func (i *Interpreter) evalArrayIndexExpression(array, index Value) Value {
	arrayObject := array.(*ArrayValue)
	idx := int(index.(*NumberValue).Value)
	max := len(arrayObject.Elements) - 1
	
	if idx < 0 || idx > max {
		return &NullValue{}
	}
	
	return arrayObject.Elements[idx]
}

func (i *Interpreter) evalStringIndexExpression(str, index Value) Value {
	stringObject := str.(*StringValue)
	idx := int(index.(*NumberValue).Value)
	max := len(stringObject.Value) - 1
	
	if idx < 0 || idx > max {
		return &NullValue{}
	}
	
	return &StringValue{Value: string(stringObject.Value[idx])}
}

func (i *Interpreter) isTruthy(obj Value) bool {
	switch obj {
	case &NullValue{}:
		return false
	case &BooleanValue{Value: true}:
		return true
	case &BooleanValue{Value: false}:
		return false
	default:
		return true
	}
}

func (i *Interpreter) newError(format string, a ...interface{}) Value {
	err := fmt.Sprintf(format, a...)
	i.addError(err)
	return &StringValue{Value: "ERROR: " + err}
}

func (i *Interpreter) isError(obj Value) bool {
	if obj == nil {
		return false
	}
	if str, ok := obj.(*StringValue); ok {
		return strings.HasPrefix(str.Value, "ERROR:")
	}
	return false
}

// REPL and main execution
func StartREPL() {
	scanner := bufio.NewScanner(os.Stdin)
	env := NewEnvironment()
	interpreter := NewInterpreter()
	
	fmt.Println("Welcome to Flow - A practical interpreted language!")
	fmt.Println("Type your code and press Enter. Type 'exit' to quit.")
	fmt.Print(">> ")
	
	for scanner.Scan() {
		line := scanner.Text()
		
		if line == "exit" {
			fmt.Println("Goodbye!")
			break
		}
		
		if strings.TrimSpace(line) == "" {
			fmt.Print(">> ")
			continue
		}
		
		lexer := NewLexer(line)
		parser := NewParser(lexer)
		statements := parser.ParseProgram()
		
		if len(parser.Errors()) != 0 {
			for _, err := range parser.Errors() {
				fmt.Printf("Parse error: %s\n", err)
			}
		} else {
			for _, stmt := range statements {
				result := interpreter.Eval(stmt, env)
				if result != nil && result.Type() != NULL_VAL {
					if !interpreter.isError(result) {
						fmt.Printf("%s\n", result.String())
					}
				}
			}
			
			if len(interpreter.Errors()) != 0 {
				for _, err := range interpreter.Errors() {
					fmt.Printf("Runtime error: %s\n", err)
				}
				interpreter.errors = []string{} // Clear errors
			}
		}
		
		fmt.Print(">> ")
	}
}

func ExecuteFile(filename string) {
	content, err := os.ReadFile(filename)
	if err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		return
	}
	
	lexer := NewLexer(string(content))
	parser := NewParser(lexer)
	program := parser.ParseProgram()
	
	if len(parser.Errors()) != 0 {
		for _, err := range parser.Errors() {
			fmt.Printf("Parse error: %s\n", err)
		}
		return
	}
	
	env := NewEnvironment()
	interpreter := NewInterpreter()
	
	for _, stmt := range program {
		result := interpreter.Eval(stmt, env)
		if interpreter.isError(result) {
			fmt.Printf("Runtime error: %s\n", result.String())
			return
		}
	}
	
	if len(interpreter.Errors()) != 0 {
		for _, err := range interpreter.Errors() {
			fmt.Printf("Runtime error: %s\n", err)
		}
	}
}

func main() {
	if len(os.Args) > 1 {
		ExecuteFile(os.Args[1])
	} else {
		StartREPL()
	}
}
