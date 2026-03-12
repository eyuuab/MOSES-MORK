from DependencyMiner.miner import DependencyMiner
from Representation.representation import *
from Representation.helpers import *
from Representation.csv_parser import load_truth_table
from Representation.selection import select_top_k, tournament_selection
from Representation.sampling import sample_from_TTable
from Variation_quantale.crossover import VariationQuantale, crossTopOne
from Variation_quantale.mutation import Mutation
from FactorGraph_EDA.beta_bp import BetaFactorGraph
from Moses.run_bp_moses import run_bp_moses, _finalize_metapop
from Moses.run_abp_moses import run_abp_moses
import random
import math
from typing import List

def run_moses(exemplar: Instance, fitness: FitnessOracle, hyperparams: Hyperparams, 
              knobs: List[Knob], target: List[bool], csv_path: str, 
              metapop: List[Instance], max_iter: int = 100, fg_type: str = "alpha") -> List[Instance]:
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
        max_iter: Maximum iterations
        fg_type: 'beta' for beta-variational MOSES, 'alpha' for standard MOSES
    
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
            max_dist=20,
            last_chance=False,
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


def grid_search_tuning():
    print("--- Starting Hyperparameter Grid Search ---")
    
    # b_probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # u_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    b_probs = [0.5, 0.6, 0.7, 0.8]
    u_probs = [0.5, 0.6, 0.7, 0.8]
    
    
    random.seed(42)
    csv_path = "example_data/test_parity_3.csv"
    input_data, target = load_truth_table(csv_path, output_col='O')
    knobs = knobs_from_truth_table(input_data)
    knobs = [k for k in knobs if k.symbol != 'O']
    fitness = FitnessOracle(target)

    results = []

    for b in b_probs:
        for u in u_probs:
            print(f"\nTesting: Bernoulli={b}, Uniform={u}")
            
            current_hp = Hyperparams(
                mutation_rate=0.3, 
                crossover_rate=0.5, 
                num_generations=15,
                neighborhood_size=20,
                bernoulli_prob=b, 
                uniform_prob=u
            )
            
            exemplar = Instance(value=f"(AND)", id=0, score=0.0, knobs=knobs)
            exemplar.score = fitness.get_fitness(exemplar)
            
            metapop = [exemplar]
            
            final_pop = run_moses(
                exemplar=exemplar, 
                fitness=fitness, 
                hyperparams=current_hp, 
                knobs=knobs,
                target=target, 
                csv_path=csv_path, 
                metapop=metapop, 
                max_iter=5, 
                fg_type="beta"
            )
            
            # Find best score in this run
            if final_pop:
                best_inst = max(final_pop, key=lambda x: x.score)
                results.append({
                    'b': b, 
                    'u': u, 
                    'score': best_inst.score, 
                    'instance': best_inst.value
                })
                print(f"-> Result: Score {best_inst.score:.4f}")
            else:
                print("-> Result: No population return")

    # 4. Report Best
    print("\n--- Tuning Results ---")
    results.sort(key=lambda x: x['score'], reverse=True)
    for r in results:
        print(f"Score: {r['score']:.4f} | B={r['b']}, U={r['u']} | Inst: {r['instance']}")

    print(f"\n*** Best Configuration: B={results[0]['b']}, U={results[0]['u']} ***")

if __name__ == "__main__":
    # main() 
    grid_search_tuning() # Call this instead        




def main(): 
    random.seed(42)
    # grid_search_tuning()
    metapop = []
    csv_path = "example_data/test_parity_3.csv"
    hyperparams = Hyperparams(mutation_rate=0.3, crossover_rate=0.5, num_generations=30, neighborhood_size=20, bernoulli_prob=0.6, uniform_prob=0.8)
    input, target = load_truth_table(csv_path, output_col='O')
    knobs = knobs_from_truth_table(input)
    knobs = [k for k in knobs if k.symbol != 'O']
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
        max_iter=10,
        fg_type="beta"  # Change to "alpha" for alpha version of factor graph, "beta" for BP-based MOSES
    )
    
    
if __name__ == "__main__":
    main()