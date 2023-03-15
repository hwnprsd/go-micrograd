package graph

import (
	"fmt"
	"log"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
	"github.com/hwnprsd/go-micrograd/pkg/engine"
)

type NodeSet = map[*engine.Value]bool
type EdgeSet = map[*Edge]bool

type Edge struct {
	Child *engine.Value
	Value *engine.Value
}

func Trace(root *engine.Value) (NodeSet, EdgeSet) {
	log.Println("ROOT", root)
	nodes, edges := NodeSet{}, EdgeSet{}
	var build func(val *engine.Value)
	build = func(val *engine.Value) {
		// if val is not in nodes
		if !nodes[val] {
			nodes[val] = true
			if val.Child1 != nil {
				edge := &Edge{
					Child: val.Child1,
					Value: val,
				}
				edges[edge] = true
				build(val.Child1)
			}

			if val.Child2 != nil {
				edge := &Edge{
					Child: val.Child2,
					Value: val,
				}
				edges[edge] = true
				build(val.Child2)
			}
		}
	}
	build(root)
	return nodes, edges
}

func DrawDot(root *engine.Value) {
	nodes, edges := Trace(root)
	g := graphviz.New()
	g.SetLayout(graphviz.Layout(graphviz.XDOT))
	graph, err := g.Graph()
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := graph.Close(); err != nil {
			log.Fatal(err)
		}
		g.Close()
	}()

	for n := range nodes {
		uid := fmt.Sprintf("%p", n)
		node, _ := graph.CreateNode(uid)
		node.SetLabel(fmt.Sprintf("data = %f | grad = %f ", n.Data, n.Grad)).SetShape(cgraph.RectangleShape)
		if n.Op != "" {
			node2, _ := graph.CreateNode(uid + n.Op)
			node2.SetLabel(n.Op)
			graph.CreateEdge("edge"+uid+n.Op, node2, node)
		}
	}

	for e := range edges {
		childUid := fmt.Sprintf("%p", e.Child)
		parentUid := fmt.Sprintf("%p", e.Value) + e.Value.Op
		child, err := graph.Node(childUid)
		if err != nil {
			log.Fatal(err)
		}
		parent, err := graph.Node(parentUid)
		if err != nil {
			log.Fatal(err)
		}
		graph.CreateEdge("edge"+childUid+parentUid, child, parent)
	}

	// 3. write to file directly
	if err := g.RenderFilename(graph, graphviz.PNG, "graph.png"); err != nil {
		log.Fatal(err)
	}
}
