package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/user"
	"strconv"
	"strings"
)

// --- token/token.go ---

// TokenType is a string representing the type of a token.
type TokenType string

// Token represents a lexical token with a type and a literal value.
type Token struct {
	Type    TokenType // The type of the token, e.g., IDENT, INT, LET.
	Literal string    // The actual string value of the token, e.g., "x", "10", "let".
}

// Constants for all token types in Aura.
const (
	ILLEGAL TokenType = "ILLEGAL" // Represents an unknown or illegal token.
	EOF     TokenType = "EOF"     // Represents the end of the input.

	// Identifiers + literals
	IDENT  TokenType = "IDENT"  // e.g., add, foobar, x, y
	INT    TokenType = "INT"    // e.g., 1343456
	STRING TokenType = "STRING" // e.g., "hello world" (Not fully implemented in this basic version)

	// Operators
	ASSIGN   TokenType = "="
	PLUS     TokenType = "+"
	MINUS    TokenType = "-"
	ASTERISK TokenType = "*"
	SLASH    TokenType = "/"
	BANG     TokenType = "!" // For negation, e.g., !true (Not fully implemented for booleans yet)

	LT TokenType = "<"
	GT TokenType = ">"

	EQ     TokenType = "==" // Equal
	NOT_EQ TokenType = "!=" // Not equal

	// Delimiters
	COMMA     TokenType = ","
	SEMICOLON TokenType = ";" // Optional semicolon, used to separate statements

	LPAREN   TokenType = "("
	RPAREN   TokenType = ")"
	LBRACE   TokenType = "{" // For blocks (Not fully implemented in this basic version)
	RBRACE   TokenType = "}"
	LBRACKET TokenType = "[" // For arrays/lists (Not implemented)
	RBRACKET TokenType = "]"

	// Keywords
	FUNCTION TokenType = "FUNCTION" // `fn` keyword (Not implemented)
	LET      TokenType = "LET"      // `let` keyword for variable declaration
	TRUE     TokenType = "TRUE"     // `true` boolean literal (Not implemented)
	FALSE    TokenType = "FALSE"    // `false` boolean literal (Not implemented)
	IF       TokenType = "IF"       // `if` keyword (Not implemented)
	ELSE     TokenType = "ELSE"     // `else` keyword (Not implemented)
	RETURN   TokenType = "RETURN"   // `return` keyword (Not implemented)
	PRINT    TokenType = "PRINT"    // `print` keyword for output
)

// keywords maps keyword strings to their corresponding TokenType.
var keywords = map[string]TokenType{
	"fn":     FUNCTION,
	"let":    LET,
	"true":   TRUE,
	"false":  FALSE,
	"if":     IF,
	"else":   ELSE,
	"return": RETURN,
	"print":  PRINT,
}

// LookupIdent checks the keywords map for a given identifier.
// If it's a keyword, it returns the keyword's TokenType. Otherwise, it returns IDENT.
func LookupIdent(ident string) TokenType {
	if tokType, ok := keywords[ident]; ok {
		return tokType
	}
	return IDENT
}

// --- ast/ast.go ---

// Node is the base interface for all AST nodes.
// Every node in the AST must implement this interface.
type Node interface {
	TokenLiteral() string // Returns the literal value of the token this node is associated with.
	String() string       // Returns a string representation of the node, for debugging.
}

// Statement is a type of Node that represents a language statement (e.g., let x = 5;).
// Statements do not produce values.
type Statement interface {
	Node
	statementNode() // A dummy method to differentiate Statement nodes.
}

// Expression is a type of Node that represents a language expression (e.g., 5 + 10).
// Expressions produce values.
type Expression interface {
	Node
	expressionNode() // A dummy method to differentiate Expression nodes.
}

// Program is the root node of every AST produced by the parser.
// It's a collection of statements.
type Program struct {
	Statements []Statement // A slice of statements that make up the program.
}

// TokenLiteral returns the token literal of the first statement in the program.
// If the program has no statements, it returns an empty string.
func (p *Program) TokenLiteral() string {
	if len(p.Statements) > 0 {
		return p.Statements[0].TokenLiteral()
	}
	return ""
}

// String returns a string representation of the entire program.
// It concatenates the string representations of all its statements.
func (p *Program) String() string {
	var out bytes.Buffer
	for _, s := range p.Statements {
		out.WriteString(s.String())
	}
	return out.String()
}

// LetStatement represents a 'let' statement, e.g., `let x = myVar;`.
type LetStatement struct {
	Token Token      // The token.LET token.
	Name  *Identifier // The identifier (variable name) being assigned to.
	Value Expression  // The expression whose value is assigned to the identifier.
}

func (ls *LetStatement) statementNode()       {}
func (ls *LetStatement) TokenLiteral() string { return ls.Token.Literal }
func (ls *LetStatement) String() string {
	var out bytes.Buffer
	out.WriteString(ls.TokenLiteral() + " ")
	out.WriteString(ls.Name.String())
	out.WriteString(" = ")
	if ls.Value != nil {
		out.WriteString(ls.Value.String())
	}
	out.WriteString(";")
	return out.String()
}

// Identifier represents an identifier, e.g., a variable name like `x` or `foobar`.
type Identifier struct {
	Token Token  // The token.IDENT token.
	Value string // The string value of the identifier.
}

func (i *Identifier) expressionNode()      {}
func (i *Identifier) TokenLiteral() string { return i.Token.Literal }
func (i *Identifier) String() string       { return i.Value }

// IntegerLiteral represents an integer literal, e.g., `5`.
type IntegerLiteral struct {
	Token Token // The token.INT token.
	Value int64 // The actual integer value.
}

func (il *IntegerLiteral) expressionNode()      {}
func (il *IntegerLiteral) TokenLiteral() string { return il.Token.Literal }
func (il *IntegerLiteral) String() string       { return il.Token.Literal }

// PrefixExpression represents an expression with a prefix operator, e.g., `-10` or `!true`.
type PrefixExpression struct {
	Token    Token      // The prefix token, e.g., ! or -.
	Operator string     // The operator itself, e.g., "!" or "-".
	Right    Expression // The expression to the right of the operator.
}

func (pe *PrefixExpression) expressionNode()      {}
func (pe *PrefixExpression) TokenLiteral() string { return pe.Token.Literal }
func (pe *PrefixExpression) String() string {
	var out bytes.Buffer
	out.WriteString("(")
	out.WriteString(pe.Operator)
	out.WriteString(pe.Right.String())
	out.WriteString(")")
	return out.String()
}

// InfixExpression represents an expression with an infix operator, e.g., `5 + 5`.
type InfixExpression struct {
	Token    Token      // The operator token, e.g., +.
	Left     Expression // The expression to the left of the operator.
	Operator string     // The operator itself, e.g., "+", "-", "*", "/".
	Right    Expression // The expression to the right of the operator.
}

func (ie *InfixExpression) expressionNode()      {}
func (ie *InfixExpression) TokenLiteral() string { return ie.Token.Literal }
func (ie *InfixExpression) String() string {
	var out bytes.Buffer
	out.WriteString("(")
	out.WriteString(ie.Left.String())
	out.WriteString(" " + ie.Operator + " ")
	out.WriteString(ie.Right.String())
	out.WriteString(")")
	return out.String()
}

// ExpressionStatement represents a statement that consists solely of an expression.
// For example, `x + 10;` as a statement.
type ExpressionStatement struct {
	Token      Token      // The first token of the expression.
	Expression Expression // The expression itself.
}

func (es *ExpressionStatement) statementNode()       {}
func (es *ExpressionStatement) TokenLiteral() string { return es.Token.Literal }
func (es *ExpressionStatement) String() string {
	if es.Expression != nil {
		return es.Expression.String()
	}
	return ""
}

// PrintStatement represents a `print` statement, e.g., `print x + 5;`.
type PrintStatement struct {
	Token    Token      // The token.PRINT token.
	Argument Expression // The expression whose value will be printed.
}

func (ps *PrintStatement) statementNode()       {}
func (ps *PrintStatement) TokenLiteral() string { return ps.Token.Literal }
func (ps *PrintStatement) String() string {
	var out bytes.Buffer
	out.WriteString(ps.TokenLiteral() + " ")
	if ps.Argument != nil {
		out.WriteString(ps.Argument.String())
	}
	out.WriteString(";")
	return out.String()
}

// --- object/object.go ---

// ObjectType_ is a string representing the type of an object.
// Renamed to ObjectType_ to avoid conflict with the TokenType alias if they were ever the same.
// In this specific case, they are different, but it's a good practice for clarity when merging.
// Or, more simply, just ensure the names are distinct if they represent different concepts.
// For this merge, we will use ObjectTypeInternal to be explicit.
type ObjectTypeInternal string

// Constants for different object types.
const (
	INTEGER_OBJ ObjectTypeInternal = "INTEGER"
	BOOLEAN_OBJ ObjectTypeInternal = "BOOLEAN" // Not fully used in this basic version
	NULL_OBJ    ObjectTypeInternal = "NULL"    // Not fully used in this basic version
	ERROR_OBJ   ObjectTypeInternal = "ERROR"
	// More types like STRING_OBJ, FUNCTION_OBJ, ARRAY_OBJ can be added here.
)

// Object is the interface that all Aura objects must implement.
type Object interface {
	Type() ObjectTypeInternal // Returns the type of the object.
	Inspect() string          // Returns a string representation of the object.
}

// Integer represents an integer value.
type Integer struct {
	Value int64
}

func (i *Integer) Type() ObjectTypeInternal { return INTEGER_OBJ }
func (i *Integer) Inspect() string          { return fmt.Sprintf("%d", i.Value) }

// Boolean represents a boolean value (true or false).
type Boolean struct { // Not fully used in this basic version
	Value bool
}

func (b *Boolean) Type() ObjectTypeInternal { return BOOLEAN_OBJ }
func (b *Boolean) Inspect() string          { return fmt.Sprintf("%t", b.Value) }

// Null represents the absence of a value.
type Null struct{} // Not fully used in this basic version

func (n *Null) Type() ObjectTypeInternal { return NULL_OBJ }
func (n *Null) Inspect() string          { return "null" }

// Error represents an error that occurred during evaluation.
type Error struct {
	Message string
}

func (e *Error) Type() ObjectTypeInternal { return ERROR_OBJ }
func (e *Error) Inspect() string          { return "ERROR: " + e.Message }

// Environment stores variable bindings.
type Environment struct {
	store map[string]Object
	outer *Environment // For lexical scoping (closures, nested scopes). Not fully used in this basic version.
}

// NewEnvironment creates a new, empty environment.
func NewEnvironment() *Environment {
	s := make(map[string]Object)
	return &Environment{store: s, outer: nil}
}

// NewEnclosedEnvironment creates a new environment that is enclosed by an outer environment.
// This is used for creating new scopes, like inside functions.
// func NewEnclosedEnvironment(outer *Environment) *Environment { // Keep if needed for future function scope
// 	env := NewEnvironment()
// 	env.outer = outer
// 	return env
// }

// Get retrieves a value from the environment by name.
// It checks the current environment and then outer environments if the name is not found.
func (e *Environment) Get(name string) (Object, bool) {
	obj, ok := e.store[name]
	if !ok && e.outer != nil {
		obj, ok = e.outer.Get(name) // Recursively check outer scope.
	}
	return obj, ok
}

// Set stores a value in the environment with the given name.
func (e *Environment) Set(name string, val Object) Object {
	e.store[name] = val
	return val
}

// --- lexer/lexer.go ---

// Lexer holds the state of the lexical analysis.
type Lexer struct {
	input        string // The input source code string.
	position     int    // Current position in input (points to current char).
	readPosition int    // Current reading position in input (after current char).
	ch           byte   // Current char under examination.
}

// NewLexer creates and returns a new Lexer instance.
func NewLexer(input string) *Lexer {
	l := &Lexer{input: input}
	l.readChar() // Initialize l.ch, l.position, and l.readPosition.
	return l
}

// readChar gives us the next character and advances our position in the input string.
func (l *Lexer) readChar() {
	if l.readPosition >= len(l.input) {
		l.ch = 0 // ASCII code for "NUL" character, signifies EOF or not read yet.
	} else {
		l.ch = l.input[l.readPosition]
	}
	l.position = l.readPosition
	l.readPosition++
}

// NextToken analyzes the current character and returns the corresponding token.
func (l *Lexer) NextToken() Token {
	var tok Token

	l.skipWhitespace() // Ignore whitespace.

	switch l.ch {
	case '=':
		if l.peekChar() == '=' {
			ch := l.ch
			l.readChar()
			literal := string(ch) + string(l.ch)
			tok = Token{Type: EQ, Literal: literal}
		} else {
			tok = newToken(ASSIGN, l.ch)
		}
	case '+':
		tok = newToken(PLUS, l.ch)
	case '-':
		tok = newToken(MINUS, l.ch)
	case '!':
		if l.peekChar() == '=' {
			ch := l.ch
			l.readChar()
			literal := string(ch) + string(l.ch)
			tok = Token{Type: NOT_EQ, Literal: literal}
		} else {
			tok = newToken(BANG, l.ch) // For now, only single '!'
		}
	case '/':
		tok = newToken(SLASH, l.ch)
	case '*':
		tok = newToken(ASTERISK, l.ch)
	case '<':
		tok = newToken(LT, l.ch)
	case '>':
		tok = newToken(GT, l.ch)
	case ';':
		tok = newToken(SEMICOLON, l.ch)
	case '(':
		tok = newToken(LPAREN, l.ch)
	case ')':
		tok = newToken(RPAREN, l.ch)
	case ',':
		tok = newToken(COMMA, l.ch)
	case '{':
		tok = newToken(LBRACE, l.ch)
	case '}':
		tok = newToken(RBRACE, l.ch)
	case 0: // End of input
		tok.Literal = ""
		tok.Type = EOF
	default:
		if isLetter(l.ch) {
			tok.Literal = l.readIdentifier()
			tok.Type = LookupIdent(tok.Literal) // Check if it's a keyword or identifier.
			return tok                          // Early exit because readIdentifier advances pointers.
		} else if isDigit(l.ch) {
			tok.Type = INT
			tok.Literal = l.readNumber()
			return tok // Early exit because readNumber advances pointers.
		} else {
			tok = newToken(ILLEGAL, l.ch) // Unknown character.
		}
	}

	l.readChar() // Advance to the next character.
	return tok
}

// newToken is a helper function to create a new token.
func newToken(tokenType TokenType, ch byte) Token {
	return Token{Type: tokenType, Literal: string(ch)}
}

// readIdentifier reads an identifier (sequence of letters and digits, starting with a letter).
func (l *Lexer) readIdentifier() string {
	position := l.position
	for isLetter(l.ch) || isDigit(l.ch) { // Allow digits in identifiers after the first letter.
		l.readChar()
	}
	return l.input[position:l.position]
}

// isLetter checks if the given character is a letter or underscore.
func isLetter(ch byte) bool {
	return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_'
}

// readNumber reads a sequence of digits to form an integer.
func (l *Lexer) readNumber() string {
	position := l.position
	for isDigit(l.ch) {
		l.readChar()
	}
	return l.input[position:l.position]
}

// isDigit checks if the given character is a digit.
func isDigit(ch byte) bool {
	return '0' <= ch && ch <= '9'
}

// skipWhitespace consumes all whitespace characters (space, tab, newline, carriage return).
func (l *Lexer) skipWhitespace() {
	for l.ch == ' ' || l.ch == '\t' || l.ch == '\n' || l.ch == '\r' {
		l.readChar()
	}
}

// peekChar looks at the next character without consuming the current one.
func (l *Lexer) peekChar() byte {
	if l.readPosition >= len(l.input) {
		return 0
	}
	return l.input[l.readPosition]
}

// --- parser/parser.go ---

// Operator precedence levels.
const (
	_ int = iota
	LOWEST
	EQUALS      // ==
	LESSGREATER // > or <
	SUM         // +
	PRODUCT     // *
	PREFIX      // -X or !X
	CALL        // myFunction(X) (Not implemented)
)

// precedences maps token types to their precedence levels.
var precedences = map[TokenType]int{
	EQ:       EQUALS,
	NOT_EQ:   EQUALS,
	LT:       LESSGREATER,
	GT:       LESSGREATER,
	PLUS:     SUM,
	MINUS:    SUM,
	SLASH:    PRODUCT,
	ASTERISK: PRODUCT,
	// LPAREN: CALL, // For function calls, not implemented yet
}

// Parser holds the state of the parsing process.
type Parser struct {
	l      *Lexer   // Pointer to the lexer instance.
	errors []string // A slice to collect parsing errors.

	curToken  Token // Current token being examined.
	peekToken Token // Next token to be examined.

	// Parsing functions for different expression types.
	prefixParseFns map[TokenType]prefixParseFn
	infixParseFns  map[TokenType]infixParseFn
}

// Type definitions for parsing functions.
type (
	prefixParseFn func() Expression
	infixParseFn  func(Expression) Expression
)

// NewParser creates and returns a new Parser instance.
func NewParser(l *Lexer) *Parser {
	p := &Parser{
		l:      l,
		errors: []string{},
	}

	// Initialize parsing function maps.
	p.prefixParseFns = make(map[TokenType]prefixParseFn)
	p.registerPrefix(IDENT, p.parseIdentifier)
	p.registerPrefix(INT, p.parseIntegerLiteral)
	p.registerPrefix(BANG, p.parsePrefixExpression) // For ! operator
	p.registerPrefix(MINUS, p.parsePrefixExpression) // For - operator
	p.registerPrefix(LPAREN, p.parseGroupedExpression)

	p.infixParseFns = make(map[TokenType]infixParseFn)
	p.registerInfix(PLUS, p.parseInfixExpression)
	p.registerInfix(MINUS, p.parseInfixExpression)
	p.registerInfix(SLASH, p.parseInfixExpression)
	p.registerInfix(ASTERISK, p.parseInfixExpression)
	p.registerInfix(EQ, p.parseInfixExpression)
	p.registerInfix(NOT_EQ, p.parseInfixExpression)
	p.registerInfix(LT, p.parseInfixExpression)
	p.registerInfix(GT, p.parseInfixExpression)

	// Read two tokens, so curToken and peekToken are both set.
	p.nextToken()
	p.nextToken()

	return p
}

// Errors returns the list of parsing errors.
func (p *Parser) Errors() []string {
	return p.errors
}

// nextToken advances the tokens, setting curToken to peekToken and peekToken to the next token from the lexer.
func (p *Parser) nextToken() {
	p.curToken = p.peekToken
	p.peekToken = p.l.NextToken()
}

// ParseProgram is the main entry point for parsing. It creates the root AST Program node.
func (p *Parser) ParseProgram() *Program {
	program := &Program{}
	program.Statements = []Statement{}

	// Iterate through tokens until EOF is reached.
	for p.curToken.Type != EOF {
		stmt := p.parseStatement()
		if stmt != nil {
			program.Statements = append(program.Statements, stmt)
		}
		p.nextToken()
	}
	return program
}

// parseStatement determines the type of statement based on the current token and calls the appropriate parsing method.
func (p *Parser) parseStatement() Statement {
	switch p.curToken.Type {
	case LET:
		return p.parseLetStatement()
	case PRINT:
		return p.parsePrintStatement()
	// case RETURN: // Not implemented yet
	// 	return p.parseReturnStatement()
	default:
		return p.parseExpressionStatement()
	}
}

// parseLetStatement parses a 'let' statement.
// Expected format: let <identifier> = <expression>;
func (p *Parser) parseLetStatement() *LetStatement {
	stmt := &LetStatement{Token: p.curToken}

	if !p.expectPeek(IDENT) { // Expect an identifier after 'let'.
		return nil
	}

	stmt.Name = &Identifier{Token: p.curToken, Value: p.curToken.Literal}

	if !p.expectPeek(ASSIGN) { // Expect an '=' sign.
		return nil
	}

	p.nextToken() // Move past '=' to the start of the expression.

	stmt.Value = p.parseExpression(LOWEST) // Parse the expression.

	if p.peekTokenIs(SEMICOLON) { // Optional semicolon.
		p.nextToken()
	}

	return stmt
}

// parsePrintStatement parses a 'print' statement.
// Expected format: print <expression>;
func (p *Parser) parsePrintStatement() *PrintStatement {
	stmt := &PrintStatement{Token: p.curToken}
	p.nextToken() // Move past 'print' token

	stmt.Argument = p.parseExpression(LOWEST)

	if p.peekTokenIs(SEMICOLON) {
		p.nextToken()
	}
	return stmt
}

// parseExpressionStatement parses a statement that consists of a single expression.
func (p *Parser) parseExpressionStatement() *ExpressionStatement {
	stmt := &ExpressionStatement{Token: p.curToken}
	stmt.Expression = p.parseExpression(LOWEST) // Parse the expression with the lowest precedence.

	// If the next token is a semicolon, consume it (optional).
	if p.peekTokenIs(SEMICOLON) {
		p.nextToken()
	}
	return stmt
}

// parseExpression is the core of the Pratt parser. It handles operator precedence.
func (p *Parser) parseExpression(precedence int) Expression {
	prefix := p.prefixParseFns[p.curToken.Type] // Get the parsing function for the current token type (prefix position).
	if prefix == nil {
		p.noPrefixParseFnError(p.curToken.Type)
		return nil
	}
	leftExp := prefix() // Call the prefix parsing function.

	// While the next token is not a semicolon and has higher precedence, parse as infix.
	for !p.peekTokenIs(SEMICOLON) && precedence < p.peekPrecedence() {
		infix := p.infixParseFns[p.peekToken.Type] // Get infix parsing function for the peekToken.
		if infix == nil {
			return leftExp // No infix operator found, or it's of lower precedence.
		}
		p.nextToken()            // Consume the operator.
		leftExp = infix(leftExp) // Call the infix parsing function with the left expression.
	}
	return leftExp
}

// parseIdentifier parses an identifier.
func (p *Parser) parseIdentifier() Expression {
	return &Identifier{Token: p.curToken, Value: p.curToken.Literal}
}

// parseIntegerLiteral parses an integer literal.
func (p *Parser) parseIntegerLiteral() Expression {
	lit := &IntegerLiteral{Token: p.curToken}
	value, err := strconv.ParseInt(p.curToken.Literal, 0, 64) // Base 0 means interpret from prefix (0x, 0o), or base 10.
	if err != nil {
		msg := fmt.Sprintf("could not parse %q as integer", p.curToken.Literal)
		p.errors = append(p.errors, msg)
		return nil
	}
	lit.Value = value
	return lit
}

// parsePrefixExpression parses expressions like `-5` or `!true`.
func (p *Parser) parsePrefixExpression() Expression {
	expression := &PrefixExpression{
		Token:    p.curToken,
		Operator: p.curToken.Literal,
	}
	p.nextToken()                         // Consume the operator token.
	expression.Right = p.parseExpression(PREFIX) // Parse the operand with PREFIX precedence.
	return expression
}

// parseInfixExpression parses expressions like `5 + 5`.
func (p *Parser) parseInfixExpression(left Expression) Expression {
	expression := &InfixExpression{
		Token:    p.curToken,
		Operator: p.curToken.Literal,
		Left:     left,
	}
	precedence := p.curPrecedence()
	p.nextToken()                                     // Consume the operator.
	expression.Right = p.parseExpression(precedence) // Parse the right operand with the operator's precedence.
	return expression
}

// parseGroupedExpression parses expressions enclosed in parentheses.
func (p *Parser) parseGroupedExpression() Expression {
	p.nextToken() // Consume '('.
	exp := p.parseExpression(LOWEST)
	if !p.expectPeek(RPAREN) { // Expect ')'.
		return nil
	}
	return exp
}

// Helper methods for checking token types and precedence.
func (p *Parser) curTokenIs(t TokenType) bool {
	return p.curToken.Type == t
}

func (p *Parser) peekTokenIs(t TokenType) bool {
	return p.peekToken.Type == t
}

// expectPeek checks if the peekToken is of the expected type. If so, it advances tokens. Otherwise, it records an error.
func (p *Parser) expectPeek(t TokenType) bool {
	if p.peekTokenIs(t) {
		p.nextToken()
		return true
	}
	p.peekError(t)
	return false
}

// peekPrecedence returns the precedence of the peekToken.
func (p *Parser) peekPrecedence() int {
	if prec, ok := precedences[p.peekToken.Type]; ok {
		return prec
	}
	return LOWEST
}

// curPrecedence returns the precedence of the curToken.
func (p *Parser) curPrecedence() int {
	if prec, ok := precedences[p.curToken.Type]; ok {
		return prec
	}
	return LOWEST
}

// Error handling methods.
func (p *Parser) peekError(t TokenType) {
	msg := fmt.Sprintf("expected next token to be %s, got %s instead",
		t, p.peekToken.Type)
	p.errors = append(p.errors, msg)
}

func (p *Parser) noPrefixParseFnError(t TokenType) {
	msg := fmt.Sprintf("no prefix parse function for %s found", t)
	p.errors = append(p.errors, msg)
}

// Registration methods for parsing functions.
func (p *Parser) registerPrefix(tokenType TokenType, fn prefixParseFn) {
	p.prefixParseFns[tokenType] = fn
}

func (p *Parser) registerInfix(tokenType TokenType, fn infixParseFn) {
	p.infixParseFns[tokenType] = fn
}

// --- evaluator/evaluator.go ---

// Predefined global objects (can be extended).
var (
	// NULL_OBJ_INSTANCE  = &Null{}  // Singleton Null object (Uncomment if Null type is used)
	// TRUE_OBJ_INSTANCE  = &Boolean{Value: true}  // Singleton True object (Uncomment if Boolean type is used)
	// FALSE_OBJ_INSTANCE = &Boolean{Value: false} // Singleton False object (Uncomment if Boolean type is used)
)

// Eval is the main evaluation function. It dispatches to specific evaluation functions based on node type.
func Eval(node Node, env *Environment) Object {
	switch node := node.(type) {
	// Statements
	case *Program:
		return evalProgram(node.Statements, env)
	case *ExpressionStatement:
		return Eval(node.Expression, env)
	case *LetStatement:
		val := Eval(node.Value, env)
		if isError(val) {
			return val
		}
		env.Set(node.Name.Value, val) // Store the variable in the environment.
		return nil                     // Let statements don't produce a value themselves.
	case *PrintStatement:
		val := Eval(node.Argument, env)
		if isError(val) {
			return val
		}
		if val != nil { // Print only if there's a non-nil, non-error result.
			fmt.Println(val.Inspect())
		}
		return nil // Print statements don't produce a value.

	// Expressions
	case *IntegerLiteral:
		return &Integer{Value: node.Value}
	case *Identifier:
		return evalIdentifier(node, env)
	case *PrefixExpression:
		right := Eval(node.Right, env)
		if isError(right) {
			return right
		}
		return evalPrefixExpression(node.Operator, right)
	case *InfixExpression:
		left := Eval(node.Left, env)
		if isError(left) {
			return left
		}
		right := Eval(node.Right, env)
		if isError(right) {
			return right
		}
		return evalInfixExpression(node.Operator, left, right)

	// Default case for unhandled node types.
	default:
		return newError("unhandled AST node type: %T", node)
	}
}

// evalProgram evaluates a sequence of statements in a program.
// It stops and returns an error if any statement results in an error.
func evalProgram(stmts []Statement, env *Environment) Object {
	var result Object
	for _, statement := range stmts {
		result = Eval(statement, env)
		if errObj, ok := result.(*Error); ok {
			return errObj // Stop execution on first error.
		}
		// For other return types like ReturnValueObject, handle them here if needed.
	}
	return result // Return the result of the last evaluated statement (often nil for statement lists).
}

// evalIdentifier looks up an identifier (variable) in the environment.
func evalIdentifier(node *Identifier, env *Environment) Object {
	if val, ok := env.Get(node.Value); ok {
		return val
	}
	return newError("identifier not found: " + node.Value)
}

// evalPrefixExpression evaluates prefix expressions like `-5` or `!true`.
func evalPrefixExpression(operator string, right Object) Object {
	switch operator {
	case "!": // Not implemented for current types, placeholder for future boolean logic.
		return newError("unknown operator: %s%s", operator, right.Type())
	case "-":
		return evalMinusPrefixOperatorExpression(right)
	default:
		return newError("unknown operator: %s%s", operator, right.Type())
	}
}

// evalMinusPrefixOperatorExpression handles the negation prefix operator `-`.
func evalMinusPrefixOperatorExpression(right Object) Object {
	if right.Type() != INTEGER_OBJ {
		return newError("unknown operator: -%s", right.Type())
	}
	value := right.(*Integer).Value
	return &Integer{Value: -value}
}

// evalInfixExpression evaluates infix expressions like `5 + 5`.
func evalInfixExpression(operator string, left, right Object) Object {
	// Currently only integer arithmetic is supported.
	if left.Type() == INTEGER_OBJ && right.Type() == INTEGER_OBJ {
		return evalIntegerInfixExpression(operator, left, right)
	}
	// Could add string concatenation here: else if left.Type() == STRING_OBJ && right.Type() == STRING_OBJ
	// Or boolean comparisons: else if operator == "==" or operator == "!="
	return newError("type mismatch: %s %s %s", left.Type(), operator, right.Type())
}

// evalIntegerInfixExpression handles infix operations on integers.
func evalIntegerInfixExpression(operator string, left, right Object) Object {
	leftVal := left.(*Integer).Value
	rightVal := right.(*Integer).Value

	switch operator {
	case "+":
		return &Integer{Value: leftVal + rightVal}
	case "-":
		return &Integer{Value: leftVal - rightVal}
	case "*":
		return &Integer{Value: leftVal * rightVal}
	case "/":
		if rightVal == 0 { // Handle division by zero.
			return newError("division by zero")
		}
		return &Integer{Value: leftVal / rightVal}
	// More operators like <, >, ==, != can be added here for integers.
	// case "<":
	// 	return nativeBoolToBooleanObject(leftVal < rightVal) // Requires nativeBoolToBooleanObject and Boolean type
	// case ">":
	// 	return nativeBoolToBooleanObject(leftVal > rightVal)
	// case "==":
	// 	return nativeBoolToBooleanObject(leftVal == rightVal)
	// case "!=":
	// 	return nativeBoolToBooleanObject(leftVal != rightVal)
	default:
		return newError("unknown operator: %s %s %s", left.Type(), operator, right.Type())
	}
}

// Helper function to create a new error object.
func newError(format string, a ...interface{}) *Error {
	return &Error{Message: fmt.Sprintf(format, a...)}
}

// Helper function to check if an object is an error.
func isError(obj Object) bool {
	if obj != nil {
		return obj.Type() == ERROR_OBJ
	}
	return false
}

/*
// Helper to convert Go bool to Aura Boolean object (Uncomment if Boolean type is used and TRUE/FALSE instances defined)
func nativeBoolToBooleanObject(input bool) *Boolean {
	if input {
		return TRUE_OBJ_INSTANCE
	}
	return FALSE_OBJ_INSTANCE
}
*/

// --- repl/repl.go ---

const PROMPT = "aura>> " // The REPL prompt string.

// StartREPL begins the REPL, reading input, evaluating it, and printing the result.
// Renamed from Start to StartREPL to avoid conflict if main package had another Start func.
func StartREPL(in io.Reader, out io.Writer) {
	scanner := bufio.NewScanner(in)
	env := NewEnvironment() // Create a new environment for the REPL session.

	for {
		fmt.Fprint(out, PROMPT) // Use Fprint to write to the specified output writer.
		scanned := scanner.Scan()
		if !scanned { // Handle EOF (Ctrl+D) or scanner error.
			return
		}

		line := scanner.Text()
		if strings.ToLower(line) == "exit" { // Allow user to type 'exit' to quit.
			fmt.Fprintln(out, "Goodbye!")
			return
		}

		l := NewLexer(line)
		p := NewParser(l)

		program := p.ParseProgram()
		if len(p.Errors()) != 0 {
			printParserErrors(out, p.Errors())
			continue // Skip evaluation if there are parsing errors.
		}

		evaluated := Eval(program, env)
		if evaluated != nil {
			// Only print if Eval returned something (e.g., expression result, not for let/print statements)
			// Print statements handle their own output. Let statements don't have a value to print.
			// Errors are also objects and will be inspected.
			if evaluated.Type() != ERROR_OBJ && evaluated.Inspect() != "null" && programContainsExpression(program) {
				// Avoid printing "null" for statements that don't return values like 'let'
				// and ensure it's not an error that's already printed by printParserErrors or handled by print itself
				io.WriteString(out, evaluated.Inspect())
				io.WriteString(out, "\n")
			} else if evaluated.Type() == ERROR_OBJ {
				// Print evaluation errors directly
				io.WriteString(out, evaluated.Inspect())
				io.WriteString(out, "\n")
			}
		}
	}
}

// programContainsExpression checks if the program primarily consists of an expression statement
// that should have its result printed.
func programContainsExpression(program *Program) bool {
	if len(program.Statements) == 1 {
		_, ok := program.Statements[0].(*ExpressionStatement)
		return ok
	}
	return false
}

// printParserErrors displays parsing errors to the output writer.
func printParserErrors(out io.Writer, errors []string) {
	io.WriteString(out, "Woops! We ran into some issues here!\n")
	io.WriteString(out, " parser errors:\n")
	for _, msg := range errors {
		io.WriteString(out, "\t"+msg+"\n")
	}
}

// --- main.go (actual main function) ---

// main is the entry point of the Aura interpreter.
// It greets the user and starts the Read-Eval-Print Loop (REPL).
func main() {
	currentUser, err := user.Current()
	if err != nil {
		// Fallback if user cannot be determined
		fmt.Println("Hello! This is the Aura programming language!")
	} else {
		fmt.Printf("Hello %s! This is the Aura programming language!\n", currentUser.Username)
	}
	fmt.Println("Feel free to type in commands. Type 'exit' or press Ctrl+D to quit.")
	// Start the REPL, using standard input and output.
	StartREPL(os.Stdin, os.Stdout)
}

