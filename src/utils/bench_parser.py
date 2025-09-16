from utils.netlist import Netlist, NetlistNode
from utils.singleton import Singleton
import re


@Singleton
class BenchParser:

    def __init__(self):
        self.var_regex = '[A-Za-z][\w$]*(\.[\w$]+)?(\[\d+])?'
        self.node_regex = self.var_regex + '\s*=\s*' + self.var_regex + '\s*\(\s*(' + self.var_regex + ')(\s*(,\s*(' + self.var_regex + '))*\s*)\s*\)'
        self.key_input_regex = 'INPUT\s*\(\s*keyinput[0-9]+\s*\)'
        self.input_regex = 'INPUT\s*\(\s*' + self.var_regex + '\s*\)'
        self.output_regex = 'OUTPUT\s*\(\s*' + self.var_regex + '\s*\)'
        self.comment_regex = '\s*#\s*'

    def is_node(self, str_line):
        return bool(re.match(self.node_regex, str_line))

    def is_key_input(self, str_line):
        return bool(re.match(self.key_input_regex, str_line))

    def is_input(self, str_line):
        return bool(re.match(self.input_regex, str_line))

    def is_output(self, str_line):
        return bool(re.match(self.output_regex, str_line))

    def is_comment(self, str_line):
        return bool(re.match(self.comment_regex, str_line))

    def parse_file(self, file_name):
        fr = open(file_name, 'r')
        str_lines = [x.strip() for x in fr.readlines()]
        fr.close()

        return self.parse(str_lines)

    def parse(self, str_vec) -> Netlist:
        netlist = Netlist()

        for line in str_vec:
            node = None

            if self.is_comment(line) or not line:
                continue
            elif self.is_key_input(line):
                node = NetlistNode.parse_key_input(line)
            elif self.is_input(line):
                node = NetlistNode.parse_input(line)
            elif self.is_output(line):
                node = NetlistNode.parse_input(line)
            elif self.is_node(line):
                node = NetlistNode.parse_node(line)
            else:
                raise ValueError('BenchParser failed. Unknown string: ' + line)

            netlist.register_node(node)

        # populate primary outpouts' inputs
        netlist.update_primary_outputs()

        return netlist

    def parse_internal_node(self, str_line) -> NetlistNode:
        pass
