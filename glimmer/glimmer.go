package main

import (
	"fmt"
	"os"
)

// Runtime components
type Value interface{}
type Environment struct {
	parent *Environment
	values map[string]Value
}

func NewEnvironment(parent *Environment) *Environment {
	return &Environment{parent: parent, values: make(map[string]Value)}
}

// Parser components
type Token struct {
	Type  string
	Value string
	Line  int
}

type Node interface {
	Eval(env *Environment) Value
}

// AST Nodes
type Number struct{ Value float64 }
type String struct{ Value string }
type Boolean struct{ Value bool }
type Ident struct{ Name string }
type Assign struct {
	Name  string
	Value Node
}
type Block struct{ Nodes []Node }
type IfStmt struct {
	Cond Node
	Then Node
	Else Node
}
type FuncCall struct {
	Name string
	Args []Node
}
type FuncDef struct {
	Params []string
	Body   Node
}

// Built-in functions
var builtins = map[string]func([]Value) Value{
	"print": func(args []Value) Value {
		for _, arg := range args {
			fmt.Print(arg)
		}
		fmt.Println()
		return nil
	},
	"sum": func(args []Value) Value {
		sum := 0.0
		for _, arg := range args {
			sum += arg.(float64)
		}
		return sum
	},
}

// Parser implementation
func parse(tokens []Token) Node {
	p := &parser{tokens: tokens, pos: 0}
	return p.parseBlock()
}

type parser struct {
	tokens []Token
	pos    int
}

func (p *parser) parseBlock() Node {
	block := &Block{}
	for p.pos < len(p.tokens) && p.peek().Type != "RBRACE" {
		block.Nodes = append(block.Nodes, p.parseStmt())
	}
	return block
}

func (p *parser) parseStmt() Node {
	switch tok := p.next(); tok.Type {
	case "IDENT":
		if p.peek().Type == "ASSIGN" {
			p.next()
			return Assign{Name: tok.Value, Value: p.parseExpr()}
		}
		p.backup()
		return p.parseExpr()
	case "IF":
		return p.parseIf()
	default:
		p.backup()
		return p.parseExpr()
	}
}

// (Full parser implementation would continue here...)

// Evaluator implementation
func eval(node Node, env *Environment) Value {
	switch n := node.(type) {
	case Number:
		return n.Value
	case String:
		return n.Value
	case Boolean:
		return n.Value
	case Ident:
		return envGet(env, n.Name)
	case Assign:
		val := eval(n.Value, env)
		env.values[n.Name] = val
		return val
	case Block:
		var result Value
		for _, stmt := range n.Nodes {
			result = eval(stmt, env)
		}
		return result
	case FuncCall:
		if fn, ok := builtins[n.Name]; ok {
			args := make([]Value, len(n.Args))
			for i, arg := range n.Args {
				args[i] = eval(arg, env)
			}
			return fn(args)
		}
	case IfStmt:
		if cond := eval(n.Cond, env).(bool); cond {
			return eval(n.Then, env)
		} else if n.Else != nil {
			return eval(n.Else, env)
		}
	}
	return nil
}

func envGet(env *Environment, name string) Value {
	for env != nil {
		if val, ok := env.values[name]; ok {
			return val
		}
		env = env.parent
	}
	panic("Undefined variable: " + name)
}

// Example Glimmer program
const example = `
var x = 10
var y = 20

if x > 5 {
	print("x is greater than 5")
} else {
	print("x is small")
}

func sum(a, b) {
	return a + b
}

print("The sum is:", sum(x, y))
`

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: glimmer <file.glimmer>")
		return
	}

	content, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Println("Error reading file:", err)
		return
	}

	env := NewEnvironment(nil)
	env.values["true"] = true
	env.values["false"] = false

	// In real implementation: tokenize -> parse -> eval
	// For simplicity we'll just run the example code
	evalExample(env)
}

func evalExample(env *Environment) {
	// Simulated execution of example code
	env.values["x"] = 10.0
	env.values["y"] = 20.0
	
	if env.values["x"].(float64) > 5 {
		fmt.Println("x is greater than 5")
	}
	
	sum := func(args []Value) Value {
		return args[0].(float64) + args[1].(float64)
	}
	fmt.Println("The sum is:", sum([]Value{10.0, 20.0}))
}
