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

	// a := NewValue(1)
	// b := NewValue(2)
	// c := a.Mul(b)
	// d := NewValue(3)
	// z := c.Add(d)
	// x := z.Tanh()
	// log.Println(x)
	// x.Backward()
	// graph.DrawDot(x)

	mlp := engine.NewMLP(3, 3, 2, 1)
	x1 := []*engine.Value{
		{Data: 2.0},
		{Data: 4.0},
		{Data: 6.0},
	}
	x2 := []*engine.Value{
		{Data: 3.0},
		{Data: 6.0},
		{Data: 9.0},
	}
	x3 := []*engine.Value{
		{Data: 1.0},
		{Data: 2.0},
		{Data: 3.0},
	}
	xs := [][]*engine.Value{
		x1, x2, x3,
	}

	ys := []*engine.Value{
		NewValue(1.0),
		NewValue(1.0),
		NewValue(-1.0),
	}

	yPreds := [][]*engine.Value{}

	for _, x := range xs {
		yPreds = append(yPreds, mlp.Call(x))
	}

	log.Println(yPreds)

	stepSize := 1.0

	for i := 0; i < 1; i++ {
		for _, p := range mlp.GetParameters() {
			p.Grad = 0
		}

		yPreds := [][]*engine.Value{}

		for _, x := range xs {
			out := mlp.Call(x)
			yPreds = append(yPreds, out)
			// log.Println(x, out)
		}

		// Mean Squared Loss
		loss := NewValue(0)

		for i, y := range ys {
			loss = loss.Add((yPreds[i][0].Sub(y)).Pow(2))
		}

		log.Printf("Epoch - %d, Loss - %f", i+1, loss.Data)

		loss.Backward()

		for _, p := range mlp.GetParameters() {
			p.Data += p.Grad * -stepSize

		}
	}

	loss := NewValue(0)

	for i, y := range ys {
		loss = loss.Add((yPreds[i][0].Sub(y)).Pow(2))
	}

	graph.DrawDot(loss)

	// log.Printf("Epoch - %d, Loss - %f", i+1, loss.Data)

	// loss.Backward()
	//
	// log.Println(mlp.Layers[0].Neurons[0].Weights[0].Data, mlp.Layers[0].Neurons[0].Weights[0].Grad)
	//
	// log.Println(yPreds)

}
