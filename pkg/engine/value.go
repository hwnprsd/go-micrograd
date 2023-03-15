package engine

import (
	"fmt"

	"golang.org/x/exp/slices"
)

type Value struct {
	Data             float64
	Grad             float64
	Parent1, Parent2 *Value
	Back             func()
	Op               string
}

func (v *Value) String() string {
	// return fmt.Sprintf("Value { data = %f | grad = %f | parent1 = %s | parent2 = %s }", v.Data, v.Grad, v.Parent1, v.Parent2)
	return fmt.Sprintf("{ ptr = %p | data = %f | p1 = %p | p2 = %p }", v, v.Data, v.Parent1, v.Parent2)
}

func (v *Value) Add(other *Value) *Value {
	out := Value{Data: v.Data + other.Data}
	out.Parent1 = v
	out.Parent2 = other
	out.Op = "+"
	out.Back = func() {
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
	out.Parent1 = v
	out.Parent2 = other
	out.Op = "*"
	out.Back = func() {
		v.Grad += out.Grad * other.Data
		other.Grad += out.Grad * v.Data
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
			if node.Parent1 != nil {
				buildDag(node.Parent1)
			}
			if node.Parent2 != nil {
				buildDag(node.Parent1)
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
			if node.Back != nil {
				node.Back()
			}
		}
	}

}
