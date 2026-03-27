import random
import math
from collections import defaultdict
import itertools

from Representation.representation import Instance

class StructuralStateSpace:
    def __init__(self, max_depth=5):
        # PCFG: P(Children_Signature | Parent_Operator)
        self.pcfg = defaultdict(lambda: defaultdict(int))
        self.root_distribution = defaultdict(int)
        self.max_depth = max_depth
        self.operators = {'AND', 'OR', 'NOT'}

    def _tokenize(self, instance_value):
        """Converts '(AND A (OR B C))' -> ['(', 'AND', 'A', '(', 'OR', 'B', 'C', ')', ')']"""
        return instance_value.replace('(', ' ( ').replace(')', ' ) ').split()

    def _parse_to_tree(self, tokens):
        stack = []
        for t in tokens:
            if t == '(':
                stack.append([])
            elif t == ')':
                if not stack:
                    continue
                curr = stack.pop()
                if not stack:
                    return curr
                stack[-1].append(curr)
            else:
                if stack:
                    stack[-1].append(t)
        return stack[0] if stack else []

    def fit(self, top_instances):
        """Learns a Probabilistic Context-Free Grammar (PCFG) from valid instances."""
        for inst in top_instances:
            tokens = self._tokenize(inst.value)
            tree = self._parse_to_tree(tokens)
            
            if not tree or not isinstance(tree, list):
                continue
                
            root_op = tree[0] if len(tree) > 0 and tree[0] in self.operators else 'AND'
            self.root_distribution[root_op] += 1
            
            self._extract_rules(tree)

    def _extract_rules(self, node):
        if isinstance(node, list) and len(node) > 0:
            op = node[0]
            if op not in self.operators:
                return
                
            children_signature = []
            for child in node[1:]:
                if isinstance(child, list) and len(child) > 0:
                    child_op = child[0]
                    if child_op in self.operators:
                        children_signature.append(child_op)
                        # Recurse and extract rules for the child
                        self._extract_rules(child)
                    else:
                        children_signature.append('▢')
                else:
                    children_signature.append('▢')
            
            # Record the transition P(signature | op)
            if children_signature:
                sig_tuple = tuple(children_signature)
                self.pcfg[op][sig_tuple] += 1

    def generate_scaffold(self):
        """Generates a mathematically valid tree structure from the learned PCFG."""
        # 1. Sample root operator
        if not self.root_distribution:
            root_op = 'AND'
        else:
            ops = list(self.root_distribution.keys())
            weights = list(self.root_distribution.values())
            root_op = random.choices(ops, weights=weights)[0]
            
        # 2. Recursively generate tree
        tree = self._generate_tree(root_op, depth=1)
        
        # 3. Serialize to string
        final_str = self._tree_to_str(tree)
        # Ensure it has outer parentheses
        if not final_str.startswith('('):
            final_str = '(' + final_str + ')'
        return final_str

    def _generate_tree(self, op, depth):
        # Base fallback if depth exceeded or missing rules to prevent infinite recursion
        if depth >= self.max_depth or op not in self.pcfg or not self.pcfg[op]:
            if op == 'NOT':
                return [op, '▢']
            return [op, '▢', '▢'] # Default binary arity for AND/OR
            
        # Sample children signature from PCFG
        signatures = list(self.pcfg[op].keys())
        weights = list(self.pcfg[op].values())
        chosen_sig = random.choices(signatures, weights=weights)[0]
        
        node = [op]
        for child_type in chosen_sig:
            if child_type == '▢':
                node.append('▢')
            elif child_type in self.operators:
                node.append(self._generate_tree(child_type, depth + 1))
            else:
                node.append('▢')
                
        return node

    def _tree_to_str(self, tree):
        if not isinstance(tree, list):
            return str(tree)
        # Process children
        parts = []
        for c in tree:
            parts.append(self._tree_to_str(c))
        return "(" + " ".join(parts) + ")"


class TemplateQuantaleJoin:
    def __init__(self, stv_dict, temperature=1.0):
        self.stv_dict = stv_dict
        self.temperature = max(temperature, 0.01) # Prevent division by zero
        
        self.vars = []
        self.weights = []
        
        if stv_dict:
            # Calculate Softmax probabilities based on STV strength * confidence (Gibbs Measure)
            for k, v in stv_dict.items():
                energy = v[0] * v[1]
                self.vars.append(k)
                # Cap energy to prevent math overflow
                clamped_energy = min(energy, 50.0) 
                self.weights.append(math.exp(clamped_energy / self.temperature))

    def execute(self, scaffold_string):
        """
        Takes '(AND ▢ (OR ▢ ▢))' and sequentially samples from the STV Softmax distribution
        to fill the slots, preserving mathematically grounded exploration (Maximum Entropy Principle).
        """
        final_string = scaffold_string
        
        # Softmax random sampling explicitly supports diversity.
        while '▢' in final_string:
            if self.vars:
                # Sample a variable probabilistically
                chosen_var = random.choices(self.vars, weights=self.weights)[0]
            else:
                # Fallback if dictionary is entirely empty
                chosen_var = "A"
                
            # Replace exactly ONE instance of '▢' per iteration
            final_string = final_string.replace('▢', chosen_var, 1)
            
        return final_string