"""The graph IR - Node, Graph, and a GraphBuilder to make building them less painful.

Everything else (interpreter, codegen, passes) works on these. A few things to
know:
  - nodes point at each other by name, not by reference
  - the node list is kept in topological order (a node only refers to earlier ones)
  - Node and Graph are frozen, so passes build new graphs instead of editing
  - validation runs in __post_init__ so you find mistakes right away

Linear uses y = x @ weight + bias, with weight shaped (in_features, out_features).
Note this is NOT PyTorch's (out, in) layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ops import (
    ADD,
    CONST,
    FUSED_LINEAR_RELU,
    INPUT,
    LINEAR,
    MATMUL,
    MUL,
    OUTPUT,
    QUANTIZED_FUSED_LINEAR_RELU,
    QUANTIZED_LINEAR,
    RELU,
    SIGMOID,
    SOFTMAX,
    SUB,
    SUPPORTED_OPS,
    TANH,
)

# re-export the op constants so you can grab them from here too
__all__ = [
    "INPUT",
    "LINEAR",
    "RELU",
    "OUTPUT",
    "FUSED_LINEAR_RELU",
    "ADD",
    "SUB",
    "MUL",
    "MATMUL",
    "SOFTMAX",
    "TANH",
    "SIGMOID",
    "CONST",
    "QUANTIZED_LINEAR",
    "QUANTIZED_FUSED_LINEAR_RELU",
    "SUPPORTED_OPS",
    "Node",
    "Graph",
    "GraphBuilder",
]

# everything's stored as float64
DTYPE = np.float64


@dataclass(frozen=True)
class Node:
    """One op in the graph.

    name is a unique id, op is one of the SUPPORTED_OPS, inputs are the names of
    the nodes feeding this one, shape is the output shape. weight/bias are only
    set for Linear-ish ops (weight is (in, out), bias is (out,)); Const also uses
    weight to hold its value.
    """

    name: str
    op: str
    inputs: Tuple[str, ...] = field(default_factory=tuple)
    shape: Tuple[int, ...] = field(default_factory=tuple)
    weight: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # tidy up the types. have to use object.__setattr__ because it's frozen
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        if self.weight is not None:
            object.__setattr__(self, "weight", np.asarray(self.weight, dtype=DTYPE))
        if self.bias is not None:
            object.__setattr__(self, "bias", np.asarray(self.bias, dtype=DTYPE))
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Node name must be a non-empty string, got {self.name!r}")

        if self.op not in SUPPORTED_OPS:
            raise ValueError(
                f"Node {self.name} has unsupported op {self.op!r}; "
                f"supported ops are {sorted(SUPPORTED_OPS)}"
            )

        if not all(isinstance(i, str) for i in self.inputs):
            raise ValueError(f"Node {self.name} inputs must all be strings, got {self.inputs!r}")

        if not all(isinstance(d, int) and d > 0 for d in self.shape):
            raise ValueError(
                f"Node {self.name} shape must be a tuple of positive ints, got {self.shape!r}"
            )

        # per-op rules
        if self.op == INPUT:
            self._require_no_inputs()
            self._require_no_weights()
        elif self.op == RELU:
            self._require_single_input()
            self._require_no_weights()
        elif self.op == OUTPUT:
            self._require_single_input()
            self._require_no_weights()
        elif self.op in (
            LINEAR,
            FUSED_LINEAR_RELU,
            QUANTIZED_LINEAR,
            QUANTIZED_FUSED_LINEAR_RELU,
        ):
            self._validate_linear()
        elif self.op in (ADD, SUB, MUL, MATMUL):
            self._require_two_inputs()
            self._require_no_weights()
        elif self.op in (SOFTMAX, TANH, SIGMOID):
            self._require_single_input()
            self._require_no_weights()
        elif self.op == CONST:
            self._validate_const()

    def _require_no_inputs(self) -> None:
        if self.inputs:
            raise ValueError(f"{self.op} node {self.name} must have no inputs, got {list(self.inputs)}")

    def _require_single_input(self) -> None:
        if len(self.inputs) != 1:
            raise ValueError(
                f"{self.op} node {self.name} must have exactly one input, got {list(self.inputs)}"
            )

    def _require_two_inputs(self) -> None:
        if len(self.inputs) != 2:
            raise ValueError(
                f"{self.op} node {self.name} must have exactly two inputs, got {list(self.inputs)}"
            )

    def _require_no_weights(self) -> None:
        if self.weight is not None:
            raise ValueError(f"{self.op} node {self.name} must not have a weight")
        if self.bias is not None:
            raise ValueError(f"{self.op} node {self.name} must not have a bias")

    def _validate_linear(self) -> None:
        self._require_single_input()
        if self.weight is None or self.bias is None:
            raise ValueError(f"Linear node {self.name} must have both weight and bias")
        if self.weight.ndim != 2:
            raise ValueError(
                f"Linear node {self.name} weight must be 2D, got shape {self.weight.shape}"
            )
        if self.bias.ndim != 1:
            raise ValueError(
                f"Linear node {self.name} bias must be 1D, got shape {self.bias.shape}"
            )
        if self.weight.shape[1] != self.bias.shape[0]:
            raise ValueError(
                f"Linear node {self.name} bias shape {self.bias.shape} "
                f"does not match weight output dimension {self.weight.shape[1]}"
            )
        expected_shape = (int(self.weight.shape[1]),)
        if self.shape != expected_shape:
            raise ValueError(
                f"Linear node {self.name} shape {self.shape} must equal "
                f"weight output shape {expected_shape}"
            )

    def _validate_const(self) -> None:
        self._require_no_inputs()
        if self.weight is None:
            raise ValueError(f"Const node {self.name} must have a weight (its value)")
        if self.bias is not None:
            raise ValueError(f"Const node {self.name} must not have a bias")
        if self.weight.ndim not in (1, 2):
            raise ValueError(
                f"Const node {self.name} weight must be 1D or 2D, got shape {self.weight.shape}"
            )
        expected_shape = tuple(int(d) for d in self.weight.shape)
        if self.shape != expected_shape:
            raise ValueError(
                f"Const node {self.name} shape {self.shape} must equal "
                f"weight shape {expected_shape}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Dict version for JSON - arrays turn into nested lists."""
        return {
            "name": self.name,
            "op": self.op,
            "inputs": list(self.inputs),
            "shape": list(self.shape),
            "weight": None if self.weight is None else self.weight.tolist(),
            "bias": None if self.bias is None else self.bias.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        weight = d.get("weight")
        bias = d.get("bias")
        return cls(
            name=d["name"],
            op=d["op"],
            inputs=tuple(d.get("inputs", ())),
            shape=tuple(d.get("shape", ())),
            weight=None if weight is None else np.asarray(weight, dtype=DTYPE),
            bias=None if bias is None else np.asarray(bias, dtype=DTYPE),
        )

    def __repr__(self) -> str:  # skip the arrays, they're huge
        return f"{self.name}: {self.op} inputs={list(self.inputs)} shape={self.shape}"


@dataclass(frozen=True)
class Graph:
    """A validated list of nodes in topological order, plus which one is the output."""

    nodes: Tuple[Node, ...]
    output_node: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        self._validate()

    def _validate(self) -> None:
        if not self.nodes:
            raise ValueError("Graph must contain at least one node")

        for n in self.nodes:
            if not isinstance(n, Node):
                raise ValueError(f"Graph nodes must be Node instances, got {type(n).__name__}")

        # names have to be unique
        names = [n.name for n in self.nodes]
        seen_names: set = set()
        for name in names:
            if name in seen_names:
                raise ValueError(f"Duplicate node name: {name}")
            seen_names.add(name)
        all_names = seen_names

        if self.output_node not in all_names:
            raise ValueError(f"output_node {self.output_node!r} does not exist in the graph")

        # every input must be defined earlier - this is what keeps things sorted
        defined: set = set()
        node_by_name = {n.name: n for n in self.nodes}
        for node in self.nodes:
            for inp in node.inputs:
                if inp not in all_names:
                    raise ValueError(f"Node {node.name} references unknown input {inp}")
                if inp not in defined:
                    raise ValueError(
                        f"Graph nodes must be topologically ordered: "
                        f"{node.name} references {inp} before it is defined"
                    )
            defined.add(node.name)

        # check shapes line up between connected nodes
        for node in self.nodes:
            if node.op in (
                LINEAR,
                FUSED_LINEAR_RELU,
                QUANTIZED_LINEAR,
                QUANTIZED_FUSED_LINEAR_RELU,
            ):
                src = node_by_name[node.inputs[0]]
                in_features = int(node.weight.shape[0])
                if src.shape != (in_features,):
                    raise ValueError(
                        f"Linear node {node.name} expected input shape {src.shape}, "
                        f"but weight has input dimension {in_features}"
                    )
            elif node.op in (RELU, OUTPUT):
                src = node_by_name[node.inputs[0]]
                if node.shape != src.shape:
                    raise ValueError(
                        f"{node.op} node {node.name} shape {node.shape} must equal "
                        f"input node {src.name} shape {src.shape}"
                    )
            elif node.op in (ADD, SUB, MUL):
                a = node_by_name[node.inputs[0]]
                b = node_by_name[node.inputs[1]]
                if a.shape != b.shape:
                    raise ValueError(
                        f"{node.op} node {node.name} inputs must have equal shapes, "
                        f"got {a.name} shape {a.shape} and {b.name} shape {b.shape}"
                    )
                if node.shape != a.shape:
                    raise ValueError(
                        f"{node.op} node {node.name} shape {node.shape} must equal "
                        f"input shape {a.shape}"
                    )
            elif node.op == MATMUL:
                a = node_by_name[node.inputs[0]]
                b = node_by_name[node.inputs[1]]
                if len(a.shape) != 2 or len(b.shape) != 2:
                    raise ValueError(
                        f"MatMul node {node.name} inputs must both be 2D, got "
                        f"{a.name} shape {a.shape} and {b.name} shape {b.shape}"
                    )
                m, k = a.shape
                k2, n = b.shape
                if k != k2:
                    raise ValueError(
                        f"MatMul node {node.name} inner dimensions must match, "
                        f"got {a.name} shape {a.shape} and {b.name} shape {b.shape}"
                    )
                if node.shape != (m, n):
                    raise ValueError(
                        f"MatMul node {node.name} shape {node.shape} must equal "
                        f"({m}, {n}) from inputs {a.name} shape {a.shape} and "
                        f"{b.name} shape {b.shape}"
                    )
            elif node.op == SOFTMAX:
                src = node_by_name[node.inputs[0]]
                if len(src.shape) != 1:
                    raise ValueError(
                        f"Softmax node {node.name} input {src.name} must be 1D, "
                        f"got shape {src.shape}"
                    )
                if node.shape != src.shape:
                    raise ValueError(
                        f"Softmax node {node.name} shape {node.shape} must equal "
                        f"input node {src.name} shape {src.shape}"
                    )
            elif node.op in (TANH, SIGMOID):
                src = node_by_name[node.inputs[0]]
                if node.shape != src.shape:
                    raise ValueError(
                        f"{node.op} node {node.name} shape {node.shape} must equal "
                        f"input node {src.name} shape {src.shape}"
                    )

    def node_map(self) -> Dict[str, Node]:
        return {n.name: n for n in self.nodes}

    def get_node(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"No node named {name!r} in graph")

    def input_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.op == INPUT]

    # these return new graphs since Graph is frozen
    def with_node(self, node: Node) -> "Graph":
        return Graph(nodes=self.nodes + (node,), output_node=self.output_node)

    def with_output(self, output_node: str) -> "Graph":
        return Graph(nodes=self.nodes, output_node=output_node)

    @classmethod
    def empty(cls) -> "GraphBuilder":
        return GraphBuilder()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_node": self.output_node,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Graph":
        nodes = tuple(Node.from_dict(nd) for nd in d["nodes"])
        return cls(nodes=nodes, output_node=d["output_node"])

    def summary(self) -> str:
        """Human-readable dump of the graph, one line per node."""
        lines = [f"Graph(output_node={self.output_node!r}, nodes=["]
        for n in self.nodes:
            lines.append(f"  {n.name}: {n.op} inputs={list(n.inputs)} shape={n.shape}")
        lines.append("])")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class GraphBuilder:
    """Helper for building a Graph. Each method adds a node and returns its name,
    so you can chain them:

        b = GraphBuilder()
        x = b.input("x", (4,))
        h = b.relu("relu_0", b.linear("linear_0", x, W0, b0))
        y = b.linear("linear_1", h, W1, b1)
        graph = b.build(output_node=y)

    Validation doesn't happen until build().
    """

    def __init__(self) -> None:
        self._nodes: List[Node] = []
        self._last_output: Optional[str] = None

    def _shape_of(self, name: str) -> Tuple[int, ...]:
        for n in self._nodes:
            if n.name == name:
                return n.shape
        raise ValueError(f"Unknown input node {name!r}; add it before referencing it")

    def input(self, name: str, shape: Tuple[int, ...]) -> str:
        self._nodes.append(Node(name=name, op=INPUT, inputs=(), shape=tuple(shape)))
        return name

    def linear(self, name: str, input_name: str, weight: np.ndarray, bias: np.ndarray) -> str:
        weight = np.asarray(weight, dtype=DTYPE)
        out_features = (int(weight.shape[1]),) if weight.ndim == 2 else ()
        self._nodes.append(
            Node(
                name=name,
                op=LINEAR,
                inputs=(input_name,),
                shape=out_features,
                weight=weight,
                bias=bias,
            )
        )
        return name

    def quantized_linear(
        self, name: str, input_name: str, weight: np.ndarray, bias: np.ndarray
    ) -> str:
        # same as linear - we keep the float weights and quantize later
        weight = np.asarray(weight, dtype=DTYPE)
        out_features = (int(weight.shape[1]),) if weight.ndim == 2 else ()
        self._nodes.append(
            Node(
                name=name,
                op=QUANTIZED_LINEAR,
                inputs=(input_name,),
                shape=out_features,
                weight=weight,
                bias=bias,
            )
        )
        return name

    def quantized_fused_linear_relu(
        self, name: str, input_name: str, weight: np.ndarray, bias: np.ndarray
    ) -> str:
        """Add a ``QuantizedFusedLinearReLU`` node (see :mod:`tinynn.ops`).

        Stores the ORIGINAL float64 weight/bias, identically to ``linear``;
        quantization happens deterministically at eval/codegen time, and the
        ReLU clamp is applied after the final rescale + bias add.
        """
        weight = np.asarray(weight, dtype=DTYPE)
        out_features = (int(weight.shape[1]),) if weight.ndim == 2 else ()
        self._nodes.append(
            Node(
                name=name,
                op=QUANTIZED_FUSED_LINEAR_RELU,
                inputs=(input_name,),
                shape=out_features,
                weight=weight,
                bias=bias,
            )
        )
        return name

    def fused_linear_relu(
        self, name: str, input_name: str, weight: np.ndarray, bias: np.ndarray
    ) -> str:
        weight = np.asarray(weight, dtype=DTYPE)
        out_features = (int(weight.shape[1]),) if weight.ndim == 2 else ()
        self._nodes.append(
            Node(
                name=name,
                op=FUSED_LINEAR_RELU,
                inputs=(input_name,),
                shape=out_features,
                weight=weight,
                bias=bias,
            )
        )
        return name

    def relu(self, name: str, input_name: str) -> str:
        self._nodes.append(
            Node(name=name, op=RELU, inputs=(input_name,), shape=self._shape_of(input_name))
        )
        return name

    def output(self, name: str, input_name: str) -> str:
        self._nodes.append(
            Node(name=name, op=OUTPUT, inputs=(input_name,), shape=self._shape_of(input_name))
        )
        self._last_output = name
        return name

    def add(self, name: str, a: str, b: str) -> str:
        self._nodes.append(
            Node(name=name, op=ADD, inputs=(a, b), shape=self._shape_of(a))
        )
        return name

    def sub(self, name: str, a: str, b: str) -> str:
        self._nodes.append(
            Node(name=name, op=SUB, inputs=(a, b), shape=self._shape_of(a))
        )
        return name

    def mul(self, name: str, a: str, b: str) -> str:
        self._nodes.append(
            Node(name=name, op=MUL, inputs=(a, b), shape=self._shape_of(a))
        )
        return name

    def matmul(self, name: str, a: str, b: str) -> str:
        a_shape = self._shape_of(a)
        b_shape = self._shape_of(b)
        out_shape = (a_shape[0], b_shape[1]) if len(a_shape) == 2 and len(b_shape) == 2 else ()
        self._nodes.append(
            Node(name=name, op=MATMUL, inputs=(a, b), shape=out_shape)
        )
        return name

    def softmax(self, name: str, x: str) -> str:
        self._nodes.append(
            Node(name=name, op=SOFTMAX, inputs=(x,), shape=self._shape_of(x))
        )
        return name

    def tanh(self, name: str, x: str) -> str:
        self._nodes.append(
            Node(name=name, op=TANH, inputs=(x,), shape=self._shape_of(x))
        )
        return name

    def sigmoid(self, name: str, x: str) -> str:
        self._nodes.append(
            Node(name=name, op=SIGMOID, inputs=(x,), shape=self._shape_of(x))
        )
        return name

    def const(self, name: str, value: np.ndarray) -> str:
        value = np.asarray(value, dtype=DTYPE)
        self._nodes.append(
            Node(
                name=name,
                op=CONST,
                inputs=(),
                shape=tuple(int(d) for d in value.shape),
                weight=value,
            )
        )
        return name

    def build(self, output_node: Optional[str] = None) -> Graph:
        if output_node is None:
            output_node = self._last_output
        if output_node is None:
            raise ValueError(
                "No output_node given and no Output node was added; "
                "pass build(output_node=...) explicitly"
            )
        return Graph(nodes=tuple(self._nodes), output_node=output_node)
