import unittest
from Representation.representation import (
    knobs_from_truth_table,
)

class TestExp(unittest.TestCase):

    def setUp(self):
        self.ITable = [
            {"A": True,  "B": True},
            {"A": True,  "B": False},
            {"A": False, "B": True},
            {"A": False, "B": False},
        ]
        self.sketch = "(AND $ $)"

    def test_knobs_from_truth_table(self):
        knobs = knobs_from_truth_table(self.ITable)
        self.assertEqual(len(knobs), 2)
        symbols = [k.symbol for k in knobs]
        self.assertIn("A", symbols)
        self.assertIn("B", symbols)
        a_knob = next(k for k in knobs if k.symbol == "A")

if __name__ == '__main__':
    unittest.main()