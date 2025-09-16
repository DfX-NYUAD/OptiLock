from __future__ import annotations
from enum import Enum


class NodeType(Enum):
    PRIM_INPUT = 0,
    PRIM_OUTPUT = 1,
    PRIM_OUTPUT_NODE = 2,
    KEY_INPUT = 3,
    INTERNAL_NODE = 4


class NetlistNode:

    def __init__(self, type, output, gate=None, inputs=[]):
        self.type = type
        self.output = output
        self.gate = gate
        self.inputs = inputs

    @staticmethod
    def parse_key_input(str_line) -> NetlistNode:
        key_var_name = str_line[str_line.find('(') + 1: str_line.find(')')].strip()
        return NetlistNode(NodeType.KEY_INPUT, key_var_name)

    @staticmethod
    def parse_input(str_line) -> NetlistNode:
        input_var_name = str_line[str_line.find('(') + 1: str_line.find(')')].strip()
        type = NodeType.PRIM_INPUT if str_line.startswith('INPUT') else NodeType.PRIM_OUTPUT
        return NetlistNode(type, input_var_name)

    @staticmethod
    def parse_node(str_line) -> NetlistNode:
        parts = str_line.split('=')
        output = parts[0].strip()
        gate = str_line[str_line.find('=') + 1: str_line.find('(')].strip()
        inputs_str = parts[1][parts[1].find('(') + 1: parts[1].find(')')].strip().split(',')

        inputs = []
        for input_str in inputs_str:
            inputs.append(input_str.strip())

        return NetlistNode(NodeType.INTERNAL_NODE, output, gate, inputs)

    def get_inputs(self):
        if self.type == NodeType.PRIM_INPUT:
            return []
        else:
            return self.inputs

    def to_string(self):
        str_node = ''

        if self.type == NodeType.PRIM_INPUT or self.type == NodeType.KEY_INPUT:
            str_node = 'INPUT(' + self.output + ')'
        elif self.type == NodeType.PRIM_OUTPUT:
            str_node = 'OUTPUT(' + self.output + ')'
        else:
            str_node = self.output + ' = ' + self.gate + '('
            str_node += ', '.join(self.inputs) + ')'

        return str_node


class Netlist:

    def __init__(self):
        # self.all_map = {}
        # self.all_nodes = []
        # self.node_map = {}
        # self.dff_map = {}

        # store primary IOs (not nodes!)
        self.inputs_map = {}
        self.outputs_map = {}
        self.ios = []
        self.inputs = []
        self.outputs = []

        # store primary output nodes ("shadow nodes")
        self.primary_output_nodes = []

        # store internal nodes
        self.node_map = {}
        self.nodes = []

        # store key inputs (not nodes!)
        self.key_inputs_map = {}
        self.key_inputs = []

        # self.key_inputs_map = {}
        # self.nodes = []
        # self.dffs = []

        # self.key_inputs = []
        # self.key = []
        # self.key_prefix = ""
        # self.shadow_primary_output = []

    def register_input(self, node):
        if node.output not in self.inputs_map:
            self.inputs_map[node.output] = node
            self.inputs.append(node)

    def register_output(self, node):
        if node.output not in self.outputs_map:
            self.outputs_map[node.output] = node
            self.outputs.append(node)

    def register_internal_node(self, node):
        if node.output not in self.node_map:
            self.node_map[node.output] = node
            self.nodes.append(node)

    # def register_dff_node(self, node):
    #     if node.output not in self.dff_map:
    #         self.dff_map[node.output] = node
    #         self.dffs.append(node)

    def register_key_input(self, node):
        if node.output not in self.key_inputs_map:
            self.key_inputs_map[node.output] = node
            self.key_inputs.append(node)

    def register_node(self, node):
        if node.type == NodeType.KEY_INPUT:
            self.register_key_input(node)
        elif node.type == NodeType.PRIM_INPUT or node.type == NodeType.PRIM_OUTPUT:
            self.ios.append(node)
            if node.type == NodeType.PRIM_INPUT:
                self.register_input(node)
            else:
                self.register_output(node)
        elif node.type == NodeType.INTERNAL_NODE:
            self.register_internal_node(node)
        # elif node.type == NodeType.DFF_NODE:
        #     self.register_dff_node(node)
        else:
            raise ValueError('Unknown node type: ' + str(node.type))

    def update_primary_outputs(self):
        for out_gate in self.outputs:
            for gate in self.nodes:
                if gate.type != NodeType.PRIM_OUTPUT and gate.output == out_gate.output:
                    self.primary_output_nodes.append(gate)
                    gate.type = NodeType.PRIM_OUTPUT_NODE

    def to_string(self):
        netlist_str = ''

        for keys in self.key_inputs:
            netlist_str += keys.to_string() + '\n'

        for io in self.ios:
            netlist_str += io.to_string() + '\n'

        for node in self.nodes:
            netlist_str += node.to_string() + '\n'

        return netlist_str
