package engine

import (
	"log"
	"math/rand"
)

type Neuron struct {
	Weights []*Value
	Bias    *Value
}

func NewNeuron(numberInputs int) *Neuron {
	weights := []*Value{}
	for i := 0; i < numberInputs; i++ {
		weights = append(weights, &Value{Data: rand.Float64()})
	}
	bias := &Value{Data: rand.Float64()}
	return &Neuron{weights, bias}
}

func (n *Neuron) Call(x []*Value) *Value {
	// wx + b
	if len(x) != len(n.Weights) {
		log.Fatal("Input length mismatch at a Neuron level")
	}
	sum := &Value{Data: n.Bias.Data}
	for i, w := range n.Weights {
		sum = sum.Add(w.Mul(x[i]))
	}
	act := sum.Tanh()
	return act
}

// ---

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(numberOfInputsPerNeuron int, numberOfNeurons int) *Layer {
	arr := []*Neuron{}
	for i := 0; i < numberOfNeurons; i++ {
		arr = append(arr, NewNeuron(numberOfInputsPerNeuron))
	}
	return &Layer{
		Neurons: arr,
	}
}

func (l *Layer) Call(input []*Value) []*Value {
	out := []*Value{}
	for _, neuron := range l.Neurons {
		out = append(out, neuron.Call(input))
	}
	return out
}

// ---

type MLP struct {
	Layers []*Layer
}

func NewMLP(layerNeuronCounts ...int) *MLP {
	layers := []*Layer{}
	for i := 0; i < len(layerNeuronCounts)-1; i++ {
		layers = append(layers, NewLayer(layerNeuronCounts[i], layerNeuronCounts[i+1]))
	}
	return &MLP{layers}
}

func (m *MLP) Call(x []*Value) []*Value {
	var values []*Value
	values = []*Value{}
	values = append(values, x...)
	for _, layer := range m.Layers {
		values = layer.Call(values)
	}
	return values
}
