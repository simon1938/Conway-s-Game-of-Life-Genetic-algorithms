import numpy as np
import pygad
import time

class GameOfLifeOptimizer:
    def __init__(self, initial_grid, focus='balanced'):
        """
        Initialize the Game of Life Genetic Optimizer
        
        Args:
            initial_grid (np.array): Initial grid configuration
            focus (str): Focus of fitness function
        """
        self.grid = initial_grid
        self.focus = focus

    def update_grid(self, grid, rules):
        """Mettre à jour la grille selon les règles"""
        min_survive, max_survive, birth = rules
        new_grid = np.zeros_like(grid)
        padded_grid = np.pad(grid, pad_width=1, mode='constant', constant_values=0)
        
        for row in range(1, padded_grid.shape[0] - 1):
            for col in range(1, padded_grid.shape[1] - 1):
                neighbors = np.sum(padded_grid[row-1:row+2, col-1:col+2]) - padded_grid[row, col]
                
                if padded_grid[row, col] == 1 and (neighbors < min_survive or neighbors > max_survive):
                    new_grid[row-1, col-1] = 0
                elif padded_grid[row, col] == 0 and neighbors == birth:
                    new_grid[row-1, col-1] = 1
                else:
                    new_grid[row-1, col-1] = padded_grid[row, col]
        return new_grid

    def calculate_fitness(self, grid, rules, max_generations=20):
        """
        Calculer la fitness basée sur le focus
        
        Args:
            grid (np.array): Grille initiale
            rules (list): Règles du jeu
            max_generations (int): Nombre de générations à simuler
        
        Returns:
            float: Score de fitness
        """
        temp_grid = grid.copy()
        grid_history = [temp_grid.copy()]
        
        for _ in range(max_generations):
            temp_grid = self.update_grid(temp_grid, rules)
            grid_history.append(temp_grid.copy())
        
        # Calculs de fitness selon le focus
        if self.focus == 'stability':
            fitness = max_generations - sum(np.sum(np.abs(grid_history[i] - grid_history[i-1])) for i in range(1, len(grid_history)))
        
        elif self.focus == 'oscillation':
            fitness = self.detect_oscillation(grid_history)
        
        elif self.focus == 'movement':
            fitness = self.calculate_movement(grid_history)
        
        else:  # balanced
            stability = max_generations - sum(np.sum(np.abs(grid_history[i] - grid_history[i-1])) for i in range(1, len(grid_history)))
            oscillation = self.detect_oscillation(grid_history)
            movement = self.calculate_movement(grid_history)
            fitness = (stability + oscillation + movement) / 3
        
        return max(0, fitness)

    def detect_oscillation(self, grid_history, period_range=(2, 10)):
        """
        Détecter l'oscillation dans l'historique de la grille
        
        Returns:
            float: Score d'oscillation
        """
        for period in range(period_range[0], period_range[1] + 1):
            if period < len(grid_history):
                if np.array_equal(grid_history[0], grid_history[period]):
                    return len(grid_history) / period
        return 0

    def calculate_movement(self, grid_history):
        """
        Calculer le mouvement du centre de masse
        
        Returns:
            float: Score de mouvement
        """
        def center_of_mass(grid):
            rows, cols = np.where(grid == 1)
            if len(rows) == 0:
                return None
            return np.mean(rows), np.mean(cols)
        
        centers = [center_of_mass(grid) for grid in grid_history if center_of_mass(grid) is not None]
        
        if len(centers) > 1:
            total_distance = sum(np.linalg.norm(np.array(centers[i]) - np.array(centers[i-1])) for i in range(1, len(centers)))
            return total_distance
        return 0

    def optimize_rules(self):
        """
        Optimiser les règles avec un algorithme génétique
        
        Returns:
            list: Meilleures règles trouvées
        """
        def fitness_func(ga_instance, solution, solution_idx):
            return self.calculate_fitness(self.grid, solution)

        ga_instance = pygad.GA(
            num_generations=20,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=3,
            gene_space=[(1, 3), (3, 5), (2, 4)],
            mutation_num_genes=1,
            on_generation=lambda ga: print(f"Generation {ga.generations_completed} completed."),
            init_range_low=1,
            init_range_high=5
        )

        ga_instance.run()
        solution, fitness, _ = ga_instance.best_solution()
        print(f"Best solution: {solution}, Fitness: {fitness}")
        return [int(solution[0]), int(solution[1]), int(solution[2])]

def main():
    # Exemple d'utilisation
    initial_grid = np.random.choice([0, 1], size=(50, 50), p=[0.85, 0.15])
    optimizer = GameOfLifeOptimizer(initial_grid, focus='balanced')
    best_rules = optimizer.optimize_rules()
    print("Best rules found:", best_rules)

if __name__ == "__main__":
    main()