package main

import (
	"bufio"
	"fmt"
	"os"
	"unicode"
)

type TokenType int

const (
	IDENTIFIER TokenType = iota
	NUMBER
	STRING
	KEYWORD
	OPERATOR
	SYMBOL
	EOF
)

type Token struct {
	Type  TokenType
	Value string
}

var keywords = map[string]TokenType{
	"let":    KEYWORD,
	"fn":     KEYWORD,
	"if":     KEYWORD,
	"else":   KEYWORD,
	"for":    KEYWORD,
	"while":  KEYWORD,
	"return": KEYWORD,
}

func lex(source string) []Token {
	var tokens []Token
	i := 0
	for i < len(source) {
		char := rune(source[i])
		if unicode.IsSpace(char) {
			i++
			continue
		}
		if unicode.IsLetter(char) {
			start := i
			for i < len(source) && (unicode.IsLetter(rune(source[i])) || unicode.IsDigit(rune(source[i]))) {
				i++
			}
			word := source[start:i]
			if tokenType, ok := keywords[word]; ok {
				tokens = append(tokens, Token{Type: tokenType, Value: word})
			} else {
				tokens = append(tokens, Token{Type: IDENTIFIER, Value: word})
			}
		} else if unicode.IsDigit(char) {
			start := i
			for i < len(source) && unicode.IsDigit(rune(source[i])) {
				i++
			}
			tokens = append(tokens, Token{Type: NUMBER, Value: source[start:i]})
		} else if char == '"' {
			i++
			start := i
			for i < len(source) && rune(source[i]) != '"' {
				i++
			}
			if i == len(source) {
				fmt.Println("Error: Unclosed string")
				os.Exit(1)
			}
			tokens = append(tokens, Token{Type: STRING, Value: source[start:i]})
			i++
		} else if isOperator(char) {
			tokens = append(tokens, Token{Type: OPERATOR, Value: string(char)})
			i++
		} else if isSymbol(char) {
			tokens = append(tokens, Token{Type: SYMBOL, Value: string(char)})
			i++
		} else {
			fmt.Printf("Error: Invalid character '%c'\n", char)
			os.Exit(1)
		}
	}
	tokens = append(tokens, Token{Type: EOF, Value: ""})
	return tokens
}

func isOperator(char rune) bool {
	return char == '+' || char == '-' || char == '*' || char == '/' || char == '=' || char == '<' || char == '>'
}

func isSymbol(char rune) bool {
	return char == '{' || char == '}' || char == '(' || char == ')' || char == '[' || char == ']' || char == ';' || char == ','
}

func main() {
	file, err := os.Open("source.daily") // Source code in source.daily
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var source string
	for scanner.Scan() {
		source += scanner.Text() + "\n"
	}

	tokens := lex(source)
	for _, token := range tokens {
		fmt.Printf("%v: %s\n", token.Type, token.Value)
	}
}
