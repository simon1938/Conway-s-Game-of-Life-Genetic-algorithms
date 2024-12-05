import numpy as np
import pygad
import json

class GameTrainer:
    def __init__(self, game):
        self.game = game
        
    def calculate_pattern_fitness(self, grid_history):
        """Calculate fitness based on pattern variety, live cells, and stability"""
        # Convert grids to bytes for comparison
        unique_patterns = len(set(map(lambda g: g.tobytes(), grid_history)))
        
        # Calculate live cells for each generation
        live_cell_scores = [np.sum(grid) for grid in grid_history]
        
        # Skip empty grids
        if not live_cell_scores or max(live_cell_scores) == 0:
            return 0.0
            
        # Calculate stability between generations
        stability_scores = []
        for i in range(1, len(grid_history)):
            diff = np.sum(np.abs(grid_history[i] - grid_history[i-1]))
            stability_scores.append(diff)
            
        if not stability_scores:
            return 0.0
            
        # Calculate final fitness
        mean_live_cells = np.mean(live_cell_scores)
        mean_stability = np.mean(stability_scores)
        std_live_cells = np.std(live_cell_scores)
        
        # Avoid division by zero
        if mean_stability == 0:
            stability_factor = 1.0
        else:
            stability_factor = 1.0 / (mean_stability + 1.0)
            
        fitness = (unique_patterns * 
                  mean_live_cells * 
                  stability_factor * 
                  (std_live_cells + 1))
                  
        return max(0.0, fitness)

    def calculate_fitness(self, positions):
        """Calculate fitness for a given set of initial positions"""
        self.game.set_initial_state(positions)
        grid_history = [self.game.grid.copy()]
        current_grid = self.game.grid.copy()
        
        for _ in range(self.game.config.evolution_steps):
            current_grid = self.game.update_grid()
            grid_history.append(current_grid.copy())
            self.game.grid = current_grid.copy()
            
        return self.calculate_pattern_fitness(grid_history)

    def train(self):
        """Train using genetic algorithm to find optimal starting positions"""
        def on_generation(ga):
            best_solution = ga.best_solution()[0]
            positions = [(int(best_solution[i]), int(best_solution[i+1])) 
                        for i in range(0, len(best_solution), 2)]
            fitness = self.calculate_fitness(positions)
            print(f"Generation {ga.generations_completed}: Best Fitness = {fitness:.2f}")

        def fitness_func(ga_instance, solution, solution_idx):
            positions = [(int(solution[i]), int(solution[i+1])) 
                        for i in range(0, len(solution), 2)]
            fitness = self.calculate_fitness(positions)
            return fitness

        num_genes = self.game.config.initial_cells * 2
        ga_instance = pygad.GA(
            num_generations=self.game.config.training_attempts,  # Utilise la valeur transmise
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=num_genes,
            gene_space=[
                {"low": 0, "high": self.game.grid_width - 1} if i % 2 == 0 
                else {"low": 0, "high": self.game.grid_height - 1} 
                for i in range(num_genes)
            ],
            mutation_num_genes=2,
            mutation_type="random",
            on_generation=on_generation
        )

        print("Starting training...")
        print(f"Initial cells: {self.game.config.initial_cells}")
        print(f"Evolution steps: {self.game.config.evolution_steps}")

        ga_instance.run()
        solution, fitness, _ = ga_instance.best_solution()
        
        if solution is None:
            print("Training failed to find a solution")
            return
            
        best_positions = [(int(solution[i]), int(solution[i+1])) 
                         for i in range(0, len(solution), 2)]
        
        print(f"\nTraining completed!")
        print(f"Best fitness found: {fitness:.2f}")
        print(f"Saving {len(best_positions)} positions to best_positions.json")
        
        with open('best_positions.json', 'w') as f:
            json.dump(best_positions, f)