import time
import pygad as ga
from byte_mot20_fitness import fitness_fn

# Callback function to be called after each generation for printing the best solution
last_fitness = 0
def on_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    print("Solution   = {solution}".format(solution=ga_instance.best_solution()[0]))
    print("\n")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

# Prepare the PyGAD parameters
ga_params = {
    "num_generations": 100,
    "sol_per_pop": 20,

    "num_genes": 10,
    "gene_type": int,
    "gene_space": [0, 1],

    "parent_selection_type": "sss",  # "rank" is also a good choice
    "num_parents_mating": 10,
    "crossover_type": "single_point",  # "uniform" is also a good choice
    "crossover_probability": None,
    "mutation_type": "random",  # "adaptive" is also a good choice
    "mutation_probability": 0.1,

    "fitness_func": fitness_fn,
    "on_generation": on_generation,
    "save_solutions": True,
    "stop_criteria": None,
    "parallel_processing": None,
    "random_seed": 42,
}

# Create an instance of the GA class
ga_instance = ga.GA(**ga_params)

# Run the GA
ga_instance.run()

# Save the GA instance
ga_instance.save(filename="byte_mot20_" + str(time.time()) + ".pkl")

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
