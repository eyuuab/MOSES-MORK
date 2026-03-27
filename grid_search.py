from Representation.representation import *
from Representation.helpers import *
from Representation.csv_parser import load_truth_table

from main import run_moses

import random
import numpy as np
import datetime

import gc
import concurrent.futures


def evaluate_config(b, u, csv_path, feature_order):
    """Helper function to run a single configuration in an isolated process"""
    # Re-initialize locally so memory doesn't cross process boundaries
    input_data, target = load_truth_table(csv_path, output_col='O')
    knobs = knobs_from_truth_table(input_data)
    knobs = [k for k in knobs if k.symbol != 'O']
    fitness = FitnessOracle(target)
    
    current_hp = Hyperparams(
        mutation_rate=0.3,
        crossover_rate=0.5,
        num_generations=30,
        max_iter=100,
        neighborhood_size=50,
        fg_type="beta",
        bernoulli_prob=b,
        uniform_prob=u,
        feature_order=feature_order,
        initial_population_size=2,
        exemplar_selection_size=7,
        min_crossover_neighbors=5,
        evidence_propagation_steps=30,
        max_dist=50,
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
        max_iter=current_hp.max_iter,
        fg_type=current_hp.fg_type,
    )
    
    if final_pop:
        best_inst = max(final_pop, key=lambda x: x.score)
        return b, u, float(best_inst.score), str(best_inst.value)
    return b, u, None, "No Population"

def grid_search_tuning():

    np.random.seed(42)
    random.seed(42)
    print("--- Starting Hyperparameter Grid Search (MULTIPROCESSED) ---")
    
    b_probs = [0.5, 0.6, 0.7, 0.8]
    u_probs = [0.5, 0.6, 0.7, 0.8]
    csv_paths = ["example_data/test_parity_4.csv", "example_data/test_parity_5.csv"]

    for csv_path in csv_paths:
        feature_order = 4 if "parity_4" in csv_path else 5
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
        log_filename = f"grid_search_results_{csv_path[13:-4]}_{timestamp}.txt"

        results = []

        with open(log_filename, "w") as log_file:
            header = f"--- Starting Grid Search on {csv_path} at {timestamp} ---\n"
            print(header.strip())
            log_file.write(header + "\n")
            log_file.write(f"{'Bernoulli':<10} | {'Uniform':<10} | {'Score':<10} | {'Top Instance'}\n")
            log_file.write("-" * 80 + "\n")

            # Determine max workers: Leaves 1 core free for OS tasks
            import os
            max_cores = max(1, os.cpu_count() - 1)
            print(f"Distributing tasks across {max_cores} CPU cores...")

            # Run parallel execution
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
                # Submit all tasks to the pool
                futures = {
                    executor.submit(evaluate_config, b, u, csv_path, feature_order): (b, u) 
                    for b in b_probs for u in u_probs
                }

                # As each task finishes, write it to the log
                for future in concurrent.futures.as_completed(futures):
                    b, u, score, val = future.result()
                    
                    if score is not None:
                        results.append({'b': b, 'u': u, 'score': score, 'instance': val})
                        print(f"-> Result: [B={b}, U={u}] Score {score:.4f}")
                        log_line = f"{b:<10.1f} | {u:<10.1f} | {score:<10.4f} | {val}\n"
                    else:
                        print(f"-> Result: [B={b}, U={u}] No population return")
                        log_line = f"{b:<10.1f} | {u:<10.1f} | {'N/A':<10} | {val}\n"
                        
                    log_file.write(log_line)
                    log_file.flush()

            # Final summary sorting
            print("\n--- Tuning Results ---")
            log_file.write("\n" + "="*80 + "\n")
            log_file.write("FINAL SUMMARY (Sorted by Score descending)\n")
            log_file.write("="*80 + "\n")

            results.sort(key=lambda x: x['score'], reverse=True)
            
            for r in results:
                summary_line = f"Score: {r['score']:.4f} | B={r['b']}, U={r['u']} | Inst: {r['instance']}"
                print(summary_line)
                log_file.write(summary_line + "\n")

            if results:
                best = results[0]
                best_msg = f"\n*** Best Configuration: B={best['b']}, U={best['u']} with Score {best['score']:.4f} ***"
                print(best_msg)
                log_file.write(best_msg + "\n")


if __name__ == "__main__":
    grid_search_tuning()