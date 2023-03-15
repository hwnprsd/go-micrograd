package engine

import (
	"fmt"
	"math"

	"golang.org/x/exp/slices"
)

type Value struct {
	Data           float64
	Grad           float64
	Child1, Child2 *Value
	back           func()
	Op             string
}

func (v *Value) String() string {
	// return fmt.Sprintf("Value { data = %f | grad = %f | parent1 = %s | parent2 = %s }", v.Data, v.Grad, v.Parent1, v.Parent2)
	return fmt.Sprintf("{ ptr = %p | data = %f | p1 = %p | p2 = %p }", v, v.Data, v.Child1, v.Child2)
}

func (v *Value) Add(other *Value) *Value {
	out := Value{Data: v.Data + other.Data}
	out.Child1 = v
	out.Child2 = other
	out.Op = "+"
	out.back = func() {
		v.Grad += out.Grad
		other.Grad += out.Grad
	}
	return &out
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Mul(&Value{Data: -1}))
}

func (v *Value) Mul(other *Value) *Value {
	out := Value{Data: v.Data * other.Data}
	out.Child1 = v
	out.Child2 = other
	out.Op = "*"
	out.back = func() {
		v.Grad += out.Grad * other.Data
		other.Grad += out.Grad * v.Data
	}
	return &out
}

func (v *Value) Exp() *Value {
	out := Value{Data: math.Exp(v.Data)}
	out.Child1 = v
	out.Op = "e"
	out.back = func() {
		v.Grad = out.Data * out.Grad
	}
	return &out
}

func (v *Value) Pow(power float64) *Value {
	out := Value{Data: math.Pow(v.Data, power)}
	out.Child1 = v
	out.Op = "**"
	out.back = func() {
		v.Grad = power * math.Pow(v.Data, power-1)
	}
	return &out
}

func (v *Value) Tanh() *Value {
	x := v.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := Value{Data: t}
	out.Child1 = v
	out.Op = "tanh"
	out.back = func() {
		v.Grad = (1 - math.Pow(t, 2)) * out.Grad
	}
	return &out
}

func (v *Value) Backward() {
	v.Grad = 1
	topoList := []*Value{}
	visited := []*Value{}

	var buildDag func(node *Value)

	buildDag = func(node *Value) {
		if !slices.Contains(visited, node) {
			visited = append(visited, node)
			if node.Child1 != nil {
				buildDag(node.Child1)
			}
			if node.Child2 != nil {
				buildDag(node.Child1)
			}
			topoList = append(topoList, node)
		}
	}
	buildDag(v)
	for i, j := 0, len(topoList)-1; i < j; i, j = i+1, j-1 {
		topoList[i], topoList[j] = topoList[j], topoList[i]
	}
	for _, node := range topoList {
		if node != nil {
			if node.back != nil {
				node.back()
			}
		}
	}

}
