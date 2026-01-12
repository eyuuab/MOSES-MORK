import collections
import itertools


class TreeNode:
    def __init__(self, label):
        self.label = label
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        # Canonical string representation (DFS Pre-order)
        if self.is_leaf():
            return self.label
        child_strs = " ".join([str(c) for c in self.children])
        return f"({self.label} {child_strs})"

def tokenize(s_expr):
    """Converts s-expression string into a list of significant tokens."""
    return s_expr.replace('(', ' ( ').replace(')', ' ) ').split()

def parse_sexpr(tokens):
    """
    Recursive parser that converts tokens into a TreeNode hierarchy.
    Handles standard (HEAD TAIL) and list-grouping ((...)) structures.
    """
    if len(tokens) == 0:
        raise ValueError("Unexpected end of input")
    
    token = tokens.pop(0)
    
    if token == '(':
        if tokens[0] == '(':
            #((NOT A) B) -> No explicit label, treated as implicit 'GROUP'
            node = TreeNode("GROUP") 
        else:
            node = TreeNode(tokens.pop(0))
        
        while tokens[0] != ')':
            node.add_child(parse_sexpr(tokens))
        
        tokens.pop(0)
        return node
    elif token == ')':
        raise ValueError("Unexpected )")
    else:
        # If the token is a leaf node/literal's liek A, B ...
        return TreeNode(token)


class OrderedTreeMiner:
    def __init__(self, min_support=2):
        self.min_support = min_support
        self.patterns = collections.defaultdict(int)
        self.pattern_map = {} # Maps string repr -> TreeNode obj 

    def _get_subtrees(self, node):
        """
        Generates all 'Induced' subtrees rooted at this node.
        An induced subtree must preserve parent-child relationships.
        
        Returns a list of strings representing subtrees rooted here.
        """
        my_subtrees = [f"({node.label})"]
        
        # Get all subtrees for all children
        child_subtree_groups = [self._get_subtrees(child) for child in node.children]
        
        # Simple Logic: We will form a subtree by taking the current node
        # and attaching ONE of the valid subtrees from each child (or skipping the child).
        # STRICT ORDERED MINING: We preserve the order of children.
        
        
        # For every child, we can either:
        # a) Not include it (None)
        # b) Include one of its subtrees
        
        # Generate cartesian product of children's possibilities
        # We add [''] to represent "skipping this child"
        options_per_child = [[''] + group for group in child_subtree_groups]
        
        for combination in itertools.product(*options_per_child):
            valid_children = [c for c in combination if c]
            
            if not valid_children:
                continue
                
            children_str = " ".join(valid_children)
            subtree_str = f"({node.label} {children_str})"
            my_subtrees.append(subtree_str)
            
        return my_subtrees

    def fit(self, s_expressions):
        """
        Process the list of s-expressions and find frequent patterns.
        """
        self.patterns = collections.defaultdict(int)
        
        for expr in s_expressions:
            tokens = tokenize(expr)
            root = parse_sexpr(tokens)
            
            # To count support correctly (document frequency), we use a set per tree
            seen_in_this_tree = set()
            
            # Generating subtrees for every node
            queue = [root]
            while queue:
                curr = queue.pop(0)
                
                # Generate all subtrees rooted at current node
                subtrees = self._get_subtrees(curr)
                seen_in_this_tree.update(subtrees)
                
                queue.extend(curr.children)
            
            # Update global counts
            for pattern in seen_in_this_tree:
                self.patterns[pattern] += 1
                
        return self

    def get_frequent_patterns(self):
        """Returns sorted list of (pattern, count) tuples."""
        # Filter by min_support
        # Sort by frequency (desc), then length (desc)
        frequent = {k: v for k, v in self.patterns.items() if v >= self.min_support}
        return sorted(frequent.items(), key=lambda item: (-item[1], -len(item[0])))

data = [
    "(AND A B C)",
    "(AND (NOT A) B C)",
    "(AND A (NOT B) C)",
    "(AND (NOT A) (OR (NOT B) C))",
    "(AND A (OR C B))",
    "(AND (OR (NOT A) C) B)",
    "(AND A (NOT C) B)",
    "(AND (NOT A) (OR (NOT C) B))",
    "(AND B C)",
    "(AND (NOT B) C)",
    "(AND B (NOT C))",
    "(AND (NOT B) (NOT C))"
]

print(f"--- Processing {len(data)} S-Expressions ---")

miner = OrderedTreeMiner(min_support=4)
miner.fit(data)
results = miner.get_frequent_patterns()

print(f"\n{'FREQUENCY':<10} | {'PATTERN (S-Expression)'}")
print("-" * 50)


for pattern, freq in results:
    clean_pat = pattern.replace("GROUP", "list")
    print(f"{freq:<10} | {clean_pat}")