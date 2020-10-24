# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import abc


class Node(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self.dependencies = []


class Graph:
    def __init__(self):
        self.nodes = {}  # Nodes by name
        self.node_inst = set()  # Set of Node instances

    def add(self, node: Node):
        if not node or not isinstance(node, Node):
            raise ValueError("Graph.add: Invalid node")

        if node.name in self.nodes:
            raise ValueError(f"Cannot register multiple instances of '{node.name}' \
                             in the graph")

        if node in self.node_inst:
            raise ValueError("Cannot register multiple instances of same node")

        self.nodes[node.name] = node
        self.node_inst.add(node)

        for dep in node.dependencies:
            if dep not in self.nodes:
                raise ValueError(f"Cannot find dependency {dep} in the graph")

    def topological_sort(self) -> list:
        """Return nodes topologically sorted based on their dependencies.

        Returns:
            List of lists of nodes sorted according the dependencies.
        """
        if not self.nodes:
            return []

        layers = []
        visited = set()
        while len(visited) < len(self.nodes):
            new_layer = []
            new_names = set()
            for name, node in self.nodes.items():
                if name in visited:
                    continue

                resolved = True
                for d in node.dependencies:
                    if d not in visited:
                        resolved = False
                        break

                if resolved:
                    new_layer.append(node)
                    new_names.add(name)

            visited = visited.union(new_names)
            layers.append(new_layer)

        return layers


def create_graph() -> Graph:
    graph = Graph()
    return graph
