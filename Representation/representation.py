#Representation

from hyperon import *

# from ..reduct.enf.main import reduce
from Representation.helpers import *

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, Callable, Dict, Tuple
from copy import deepcopy
import random
import re


class Quantale(ABC):
    @abstractmethod
    def join(self, parent1, parent2):
        """Join operation for the quantale."""
        pass

    @abstractmethod
    def product(self, parent1, parent2):
        """Product operation for the quantale."""
        pass
    
    @abstractmethod
    def residium(self, parent1, parent2):
        """Residium operation for the quantale."""
        pass

    @abstractmethod
    def unit(self):
        """Unit operation for the quantale."""
        pass


class KnobVariable:
    """
    Represents a specific 'hole' or decision point in the program sketch.
    Example: In (AND _ _), we have KnobVariable(0) and KnobVariable(1).
    This is similar to a NilVertex in metta-moses.
    """
    def __init__(self, index: int, name: str, domain: List[str]):
        self.index = index
        self.name = name
        self.domain = domain # Possible values e.g. ['A', 'B', 'NOT A']
        self.value = None

    def __repr__(self):
        return f"Var({self.name})"


class Factor:
    """
    Connects KnobVariables. 
    The 'potential' function returns the probability (or PMI score) 
    of a specific configuration of these variables.
    """
    def __init__(
        self,
        variables: List[KnobVariable],
        potential_table: Dict[Tuple[str, ...], float],
        name: str = "factor"
    ):
        self.variables = variables
        self.potential_table = potential_table # The CPT (Conditional Prob Table)
        self.name = name

    def evaluate(self, assigned_values: List[str]) -> float:
        """
        Looks up the score for a specific set of values.
        E.g., if Var1='A' and Var2='B', return 0.9.
        """
        key = tuple(assigned_values)
        # Returning the PMI value for now.
        # TODO: implement PLN's rule evaluation here. 
        return self.potential_table.get(key, 0.001) 

class FactorGraph(Quantale):
    def __init__(self, variables: List[Any], factors: List[Factor]) -> None:
        self.variables = variables
        self.factors = factors

    def add_variable(self, var: Any) -> None:
        self.variables.append(var)

    def add_factor(self, factor: Factor) -> None:
        self.factors.append(factor)

    def neighbors(self, var: Any) -> List[Factor]:
        """Return all factors that involve the given variable."""
        return [f for f in self.factors if var in f.variables]


@dataclass
class Knob:
    symbol: str
    id: int
    Value: List[bool]

    def __hash__(self):
        return hash((self.symbol, self.id)) # Hash based on unique fields
    
    def __eq__(self, other):
        if not isinstance(other, Knob):
            return False
        return self.symbol == other.symbol and self.id == other.id

@dataclass
class Instance:
    value: Any
    id: int
    score: float
    knobs: List[Knob]

    def _get_complexity(self):
        """Weakness implementation in a nutshell: count the number of tokens in the expression."""
        clean = self.value.replace('(', ' ').replace(')', ' ')
        tokens = [t for t in clean.split() if t.strip()]
        return len(tokens)

@dataclass
class Hyperparams:
    """Tunable controls for search breadth, variation, and recovery behavior."""
    mutation_rate: float
    crossover_rate: float
    num_generations: int
    neighborhood_size: int
    max_iter: int = 100
    fg_type: str = "alpha"
    bernoulli_prob: float = 0.5
    uniform_prob: float = 0.5
    initial_population_size: int = 2
    exemplar_selection_size: int = 7
    min_crossover_neighbors: int = 5
    evidence_propagation_steps: int = 20
    max_dist: int = 20

class Deme(Quantale):
    def __init__(self, instances: List[Instance], id: str, q_hyper: Hyperparams) -> None:
        super().__init__()
        self.instances = instances
        self.id = id
        self.q_hyper = q_hyper
        self.generation = 0
        self.factor_graph : FactorGraph | None = None

    # TODO: implement variational quantale operations
    def join(self, parent1: Instance, parent2: Instance) -> Instance:
        pass

    def product(self, parent1: Instance, parent2: Instance) -> Instance:
        pass

    def residium(self, parent1: Instance, parent2: Instance) -> Instance:
        pass

    def unit(self) -> Instance:
        pass

    
    def to_tree(self):
        header = (
            "{Deme "
            + str(self.id)
            + " Generation: "
            + str(self.generation)
            + " Instances: "
            + str(len(self.instances))
            + "}"
        )

        # one entry per instance
        body = [
            {
                "id": instance.id,
                "expression": instance.value,
                "knobs": instance.knobs,
            }
            for instance in self.instances
        ]
        return [header, body]
        # return ["{Deme " + str(self.id) + " Generation: " + str(self.generation) + " Instances: " + str(len(self.instances)) + "}",
        #         [{"Expresstion": (instance.value, instance.knobs) for instance in self.instances}]]
    

  
def knobs_from_truth_table(ITable: List[dict], exclude: str = "O") -> List[Knob]:
    """
    Given a truth table (list of dict rows), extract:
      - unique symbols (keys)
      - unique values per symbol
    and instantiate Knob objects.
    
    Args:
        ITable: List of dictionaries representing truth table rows
        exclude: Column name to exclude from knobs. Typical value is "O".
    """
    if not ITable:
        return []

    values_by_key: dict[str, list[bool]] = {}

    for row in ITable:
        for key, val in row.items():
            if key == exclude:
                continue
            if key not in values_by_key:
                values_by_key[key] = []
            
            values_by_key[key].append(val)

    knobs: List[Knob] = []
    for idx, (symbol, vals) in enumerate(values_by_key.items(), start=1):
        knobs.append(Knob(symbol=symbol, id=idx, Value=vals))

    return knobs

class FitnessOracle:
    def __init__(self, target_vals: List[bool]):
        self.target_vals = target_vals
        self.memo: dict[str, float] = {}

    def get_fitness(self, instance: "Instance") -> float:
        """
        Evaluates the fitness of an individual based on the truth table data
        present in the instance's knobs. Utilizes caching to avoid re-evaluation.
        """
        # Check cache (using the program string as key)
        if instance.value in self.memo:
            # print('Used From Cache CCCCCCC')
            instance.score = self.memo[instance.value]
            return instance.score
        
        
        inputs: dict[str, List[bool]] = {}
        
        # Populate inputs
        row_count = len(self.target_vals)
        for knob in instance.knobs:
            inputs[knob.symbol] = knob.Value
        
        # If target not found or no data, return 0.0(or handle error)
        if row_count == 0:
            return 0.0
        
        try:
            predicted_vals = self._evaluate_expression(instance.value, inputs, row_count)
        except Exception as e:
            # Fallback for malformed expressions or eval errors
            print(f"Evaluation error for {instance.value}: {e}")
            predicted_vals = [False] * row_count

        # Count how many predictions match the target and Compute accuracy
        matches = sum(1 for p, t in zip(predicted_vals, self.target_vals) if p == t)
        accuracy = matches / row_count if row_count > 0 else 0.0
        
        self.memo[instance.value] = accuracy
        instance.score = accuracy
        return accuracy

    def _evaluate_expression(self, expr_str: str, inputs: dict[str, List[bool]], row_count: int) -> List[bool]:
        """
        Parses and evaluates the S boolean expression (AND/OR/NOT).
        """
        # Simple tokenizer: split by parens and spaces
        tokens = re.findall(r'\(|\)|[^\s()]+', expr_str)
    
        token_list = tokens
        self._idx = 0
        
        def next_token():
            if self._idx < len(token_list):
                t = token_list[self._idx]
                self._idx += 1
                return t
            return None
            
        def peek_token():
            if self._idx < len(token_list):
                return token_list[self._idx]
            return None

        def eval_node() -> List[bool]:
            t = next_token()
            if t == '(':
                # Expect operator or list
                op = next_token()
                args_vals = []
                while peek_token() != ')':
                    args_vals.append(eval_node())
                next_token() # consume ')'
                
                if op == 'AND':
                    if not args_vals: return [True] * row_count
                    # Element-wise AND
                    res = args_vals[0][:]
                    for other in args_vals[1:]:
                        for i in range(row_count):
                            res[i] = res[i] and other[i]
                    return res
                elif op == 'OR':
                    if not args_vals: return [False] * row_count
                    res = args_vals[0][:]
                    for other in args_vals[1:]:
                        for i in range(row_count):
                            res[i] = res[i] or other[i]
                    return res
                elif op == 'NOT':
                    if not args_vals: return [False] * row_count
                    # Unary usually, but logic allows arity 1
                    return [not x for x in args_vals[0]]
                else:
                    # Unknown operator or maybe this list is data?
                    # Assuming operators are uppercase symbols
                    return [False] * row_count
            elif t == ')':
                # Should be handled by loop
                return [False] * row_count
            else:
                # Atom (Variable or Literal)
                if t in inputs:
                    return inputs[t]
                elif t == 'True': return [True] * row_count
                elif t == 'False': return [False] * row_count
                # Handle holes $ as False or arbitrary?
                return [False] * row_count

        return eval_node()

# -> program sketch of this structrure is possible: 
# (AND $ $)
# -> their can be any number of $/knobs in the sketch.
# -> Given a truth table some thing like the following instances we can define the knobs:
# [ {"A": True, "B": True, "O": True}, 
#   {"A": True, "B": False, "O": False},
#   {"A": False, "B": True, "O": False},
#   {"A": False, "B": False, "O": False}]
# 
# -> The knobs would be:
# [Knob("A", [True, False], Knob("B", [True, False])]; the result would be ["O", [True, False]]
# 
# -> A deme will look like this, at the start of the program:
# Deme(["(AND A B)"], "01", Hyperparams(0.1, 0.6))
# 
# -> An instance would be:
# Instance(value="(AND A B)", id=1, score=0, knobs=[Knob("A", [True, False]), Knob("B", [True, False]), Knob("O", [True, False])])
# 
# -> The deme would then contain a list of instances after the first mutation/crossover operations:
# -> But first before that we need to sample random instances to populate the deme initially.
# -> thus without changing the parent operator we will have random sampling of knobs from our truth table.
# def sample_random_instances(instance: Instance) -> Instance:
#     from the given instance/sketch randomly assign knob values
#     return new_instance
# -> Then we can populate the deme with a list of such instances.