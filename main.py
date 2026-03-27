from Representation.representation import *
from Representation.helpers import *
from Representation.csv_parser import load_truth_table

from Moses.run_bp_moses import run_bp_moses, _finalize_metapop
from Moses.run_abp_moses import run_abp_moses
import random
import numpy as np
from typing import List


def run_moses(exemplar: Instance, fitness: FitnessOracle, hyperparams: Hyperparams, 
              knobs: List[Knob], target: List[bool], csv_path: str, 
              metapop: List[Instance], max_iter: int = 100, use_ssm=None, fg_type: str = "alpha") -> List[Instance]:
    """
    Unified entry point for running MOSES optimization.
    
    Args:
        exemplar: Initial instance
        fitness: Fitness oracle
        hyperparams: Hyperparameters
        knobs: List of knobs (not strictly used directly by recursion but passed for consistency if needed)
        target: Target values
        csv_path: Path to CSV data
        metapop: Initial metapopulation

    
    Returns: Final metapopulation of instances after evolution.
    """

    print(f"Starting MOSES Run with Strategy: {fg_type.upper()}")
    
    if fg_type.lower() == "beta":
        return run_bp_moses(
            exemplar=exemplar,
            fitness=fitness,
            hyperparams=hyperparams,
            target=target,
            csv_path=csv_path,
            metapop=metapop,
            iteration=1,
            max_iter=max_iter,
            distance=1,
            max_dist=hyperparams.max_dist,
            last_chance=False,
            use_ssm=use_ssm, 
            best_possible_score=1.0
        )
    elif fg_type.lower() == "alpha":
        final_metapop = run_abp_moses(
        exemplar=exemplar, fitness=fitness, hyperparams=hyperparams, knobs=knobs, target=target,
        csv_path=csv_path, metapop=metapop, max_iter=max_iter,
    )
        _finalize_metapop(final_metapop)
        return final_metapop
    else:
        print(f"Unknown fg_type '{fg_type}', defaulting to Alpha FG MOSES.")
        final_metapop = run_abp_moses(
        exemplar=exemplar, fitness=fitness, hyperparams=hyperparams, knobs=knobs, target=target,
        csv_path=csv_path, metapop=metapop, max_iter=max_iter,
        )
        _finalize_metapop(final_metapop)
        return final_metapop


def main(): 

    # np.random.seed(42)
    # random.seed(42)
    metapop = []

    csv_path = "example_data/test_parity_4.csv"
    hyperparams = Hyperparams(
        mutation_rate=0.3,
        crossover_rate=0.5,
        num_generations=30,
        max_iter=100,
        neighborhood_size=20,
        fg_type="beta",
        bernoulli_prob=0.5,
        uniform_prob=0.6,
        initial_population_size=2,
        exemplar_selection_size=7,
        min_crossover_neighbors=5,
        evidence_propagation_steps=30,
        max_dist=50,
        feature_order=4,
    )
    input, target = load_truth_table(csv_path, output_col='O')
    knobs = knobs_from_truth_table(input, exclude='O')
    
    exemplar = Instance(value=f"(AND)", id=0, score=0.0, knobs=knobs)
    fitness = FitnessOracle(target)
    exemplar.score = fitness.get_fitness(exemplar)
    
    print(f"Initial Exemplar: {exemplar.value} | Score: {exemplar.score}")
    metapop.append(exemplar)
    
    final_metapop = run_moses(
        exemplar=exemplar, 
        fitness=fitness, 
        hyperparams=hyperparams, 
        knobs=knobs,
        target=target, 
        csv_path=csv_path, 
        metapop=metapop,
        max_iter=hyperparams.max_iter,
        use_ssm=True,
        fg_type=hyperparams.fg_type,
    )
    
    
if __name__ == "__main__":
    main()