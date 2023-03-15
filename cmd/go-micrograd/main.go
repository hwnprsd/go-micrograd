package main

import (
	"log"

	"github.com/hwnprsd/go-micrograd/pkg/engine"
	"github.com/hwnprsd/go-micrograd/pkg/graph"
)

func NewValue(val float64) *engine.Value {
	return &engine.Value{
		Data: val,
		Grad: 0,
	}
}

func main() {
	log.Println("Hello from Micrograd")
	// z = ((a + b) * c) * d
	// a := NewValue(10)
	// b := NewValue(10)
	// c := a.Mul(b)
	// d := NewValue(10)
	// z := c.Add(d)
	// log.Println(z)
	// z.Backward()
	// graph.DrawDot(z)

	mlp := engine.NewMLP(3, 3, 2, 1)
	xs := []*engine.Value{
		{Data: 3},
		{Data: 4},
		{Data: 10},
	}
	out := mlp.Call(xs)
	ans := out[0]
	ans.Backward()
	graph.DrawDot(ans)
	log.Println(ans)
}
