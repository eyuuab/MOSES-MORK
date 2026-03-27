import unittest
import sys
import os
from Variation_quantale.ssm import StructuralStateSpace, TemplateQuantaleJoin

# Add parent directory to path to allow importing ssm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockInstance:
    def __init__(self, value):
        self.value = value

class TestStructuralStateSpace(unittest.TestCase):
    def setUp(self):
        self.ssm = StructuralStateSpace(max_depth=3)

    def test_tokenize(self):
        input_str = '(AND A (OR B C))'
        expected = ['(', 'AND', 'A', '(', 'OR', 'B', 'C', ')', ')']
        self.assertEqual(self.ssm._tokenize(input_str), expected)

    def test_parse_to_tree_simple(self):
        tokens = ['(', 'AND', 'A', 'B', ')']
        tree = self.ssm._parse_to_tree(tokens)
        self.assertEqual(tree, ['AND', 'A', 'B'])

    def test_parse_to_tree_nested(self):
        tokens = ['(', 'AND', 'A', '(', 'NOT', 'B', ')', ')']
        tree = self.ssm._parse_to_tree(tokens)
        self.assertEqual(tree, ['AND', 'A', ['NOT', 'B']])

    def test_extract_rules_and_fit(self):
        # Value structure: (AND A (OR B C))
        # Root: AND -> children: A (var), (OR ...) (op)
        # Signature for AND: ('▢', 'OR')
        instances = [MockInstance('(AND A (OR B C))')]
        self.ssm.fit(instances)
        
        self.assertIn('AND', self.ssm.pcfg)
        self.assertIn(('▢', 'OR'), self.ssm.pcfg['AND'])
        self.assertEqual(self.ssm.pcfg['AND'][('▢', 'OR')], 1)
        
        # Check OR rule
        self.assertIn('OR', self.ssm.pcfg)
        # B and C are leaves/vars, so signature is ('▢', '▢')
        self.assertIn(('▢', '▢'), self.ssm.pcfg['OR'])

    def test_generate_scaffold_fallback(self):
        # No fit called, should use default logic
        scaffold = self.ssm.generate_scaffold()
        self.assertTrue(scaffold.startswith('('))
        self.assertTrue(scaffold.endswith(')'))
        # Should likely contain AND and placeholders
        self.assertIn('▢', scaffold)

    def test_generate_scaffold_depth_limit(self):
        # Force a shallow max depth
        self.ssm.max_depth = 1
        # Add a recursive rule: AND -> (AND, AND)
        self.ssm.pcfg['AND'][('AND', 'AND')] = 1
        self.ssm.root_distribution['AND'] = 1
        
        scaffold = self.ssm.generate_scaffold()
        # Even though grammar pushes for recursion, depth limit should force terminals
        # Standard fallback for depth limit is [op, '▢', '▢']
        self.assertEqual(scaffold.replace(" ", ""), "(AND▢▢)")


class TestTemplateQuantaleJoin(unittest.TestCase):
    def setUp(self):
        # STV: {Var: (strength, confidence)}
        self.stv = {
            'X1': (1.0, 1.0),
            'X2': (0.5, 0.5)
        }
        self.joiner = TemplateQuantaleJoin(self.stv, temperature=1.0)

    def test_initialization_weights(self):
        self.assertEqual(len(self.joiner.vars), 2)
        self.assertEqual(len(self.joiner.weights), 2)
        # Check logic: energy X1 = 1.0, X2 = 0.25 -> exp(1) > exp(0.25)
        idx_x1 = self.joiner.vars.index('X1')
        idx_x2 = self.joiner.vars.index('X2')
        self.assertGreater(self.joiner.weights[idx_x1], self.joiner.weights[idx_x2])

    def test_execute_replacement(self):
        scaffold = "(AND ▢ (NOT ▢))"
        result = self.joiner.execute(scaffold)
        
        self.assertNotIn('▢', result)
        self.assertTrue('X1' in result or 'X2' in result)
        # Structure preservation
        self.assertTrue(result.startswith("(AND "))
        self.assertTrue(" (NOT " in result)

    def test_empty_stv_fallback(self):
        empty_joiner = TemplateQuantaleJoin({})
        scaffold = "(OR ▢ ▢)"
        result = empty_joiner.execute(scaffold)
        self.assertEqual(result, "(OR A A)")

if __name__ == '__main__':
    unittest.main()