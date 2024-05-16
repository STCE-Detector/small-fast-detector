import time
import pygad as ga
from byte_mot20_fitness import fitness_fn


# Callback function to be called after each generation for printing the best solution
def on_generation(ga_instance):
    print("\n")
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    print(f"Solution   = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[0]}")
    print("\n")


# Prepare the PyGAD parameters
ga_params = {
    "num_generations": 100,
    "sol_per_pop": 6,
    "keep_elitism": 2,

    "num_genes": 19,
    "gene_type": [float] * 9 + [int] * 3 + [int] * 3 + [float] + [int] + [int] + [int],
    "gene_space": [{'low': 0.2, 'high': 0.9, 'step': 0.01}] + \
                  [{'low': 0.1, 'high': 0.4, 'step': 0.01}] * 2 + \
                  [{'low': 0.2, 'high': 0.9, 'step': 0.01}] * 3 + \
                  [{'low': 0, 'high': 0.5, 'step': 0.01}] * 3 + \
                  [[0, 1]] * 3 + \
                  [[0, 1, 2, 3, 4, 5]] * 3 + \
                  [{'low': 0, 'high': 1, 'step': 0.01}] + \
                  [[0, 1]] + \
                  [range(1, 151, 10)] + \
                  [range(1, 501, 10)],

    "parent_selection_type": "sss",  # "nsga2" is also a good choice
    "num_parents_mating": 2,
    "crossover_type": "uniform",  # "uniform" is also a good choice
    "crossover_probability": 0.5,
    "mutation_type": "adaptive",  # "adaptive" is also a good choice
    "mutation_probability": [0.25, 0.1],

    "fitness_func": fitness_fn,
    "on_generation": on_generation,
    "save_solutions": True,
    "stop_criteria": "saturate_10",
    "parallel_processing": None,
    "random_seed": 42,
}

# Create an instance of the GA class
ga_instance = ga.GA(**ga_params)

# Run the GA
ga_instance.run()

# Save the GA instance
ga_instance.save(filename="./outputs/studies/byte_mot20_" + str(time.time()))

# Print summary
ga_instance.summary()

# Print the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(
        best_solution_generation=ga_instance.best_solution_generation))
