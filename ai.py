import numpy as np
import pygad
import time

def on_generation(ga_instance):
    print(f"Generation {ga_instance.generations_completed} complete.")
    time.sleep(0.5)  # Délai de 0.5 seconde entre les générations

def evaluate_grid(grid, rules):
    # Simule quelques itérations pour évaluer les règles
    min_survive, max_survive, birth = rules
    temp_grid = grid.copy()
    for _ in range(10):  # Simuler 10 itérations
        new_grid = temp_grid.copy()
        for row in range(1, grid.shape[0] - 1):
            for col in range(1, grid.shape[1] - 1):
                neighbors = np.sum(temp_grid[row-1:row+2, col-1:col+2]) - temp_grid[row, col]
                if temp_grid[row, col] == 1 and (neighbors < min_survive or neighbors > max_survive):
                    new_grid[row, col] = 0
                elif temp_grid[row, col] == 0 and neighbors == birth:
                    new_grid[row, col] = 1
        temp_grid = new_grid
    return np.sum(temp_grid)  # Retourne le nombre de cellules vivantes

def fitness_func(ga_instance, solution, solution_idx):
    # Solution : [min_survive, max_survive, birth]
    global current_grid
    score = evaluate_grid(current_grid, solution)
    return score

def optimize_rules(grid):
    global current_grid
    current_grid = grid

    # Configuration de l'algorithme génétique
    ga_instance = pygad.GA(
        num_generations=10,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=3,
        gene_space=[(1, 3), (3, 5), (3, 4)],
        mutation_num_genes=1,  # Assure qu'au moins un gène est muté
        on_generation=lambda ga: print(f"Generation {ga.generations_completed} completed.")  # Callback
    )


  # Lancer l'optimisation
    ga_instance.run()

    # Récupérer les meilleures règles
    best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()
    print(f"Best solution: {best_solution}, Fitness: {best_fitness}")
    return [int(best_solution[0]), int(best_solution[1]), int(best_solution[2])]