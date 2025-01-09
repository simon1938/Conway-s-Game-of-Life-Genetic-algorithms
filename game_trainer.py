import numpy as np
import json
import random
import datetime  

class GameTrainer:
    def __init__(self, game, grid_file=None,args=None):
        self.game = game
        self.grid_file = grid_file
        self.population_size = 10
        self.num_generations = self.game.config.training_attempts
        self.mutation_rate = 0.1
        self.elite_size = 2
        self.args = args
        self.training_history = {
            'fitness_history': [],
            'best_fitness_per_gen': [],
            'avg_fitness_per_gen': [],
            'params': {
                'population_size': self.population_size,
                'num_generations': self.num_generations,
                'mutation_rate': self.mutation_rate,
                'elite_size': self.elite_size,
                'initial_cells': self.game.config.initial_cells,
                'evolution_steps': self.game.config.evolution_steps,
                'grid_width': self.game.grid_width,
                'grid_height': self.game.grid_height,
                'grid_file': self.grid_file
            }
        }

    def generate_initial_population(self):
        """Generate initial population of random solutions or from a grid file"""
        num_genes = self.game.config.initial_cells * 2
        print("Generating initial population...")
        if self.grid_file:
            try:
                with open(self.grid_file, 'r') as f:
                    positions = json.load(f)
                print(f"Loaded {len(positions)} positions from {self.grid_file}")
                # Convert the positions into a single flattened solution
                population = [
                    [x if i % 2 == 0 else y for i, (x, y) in enumerate(positions)]
                    for _ in range(self.population_size)
                ]
                return population
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading grid file: {e}. Falling back to random initialization.")

        # Default to random population if no file is provided or loading fails
        print("No valid grid file found. Generating random initial population.")
        population = []
        for _ in range(self.population_size):
            solution = [
                random.randint(0, self.game.grid_width - 1) if i % 2 == 0 
                else random.randint(0, self.game.grid_height - 1)
                for i in range(num_genes)
            ]
            population.append(solution)
        return population
    
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

    def select_parents(self, population, fitness_scores):
        """Tournament selection method"""
        parents = []
        for _ in range(len(population)):
            # Select two random individuals
            tournament = random.sample(list(range(len(population))), 3)
            tournament_fitness = [fitness_scores[i] for i in tournament]
            winner = tournament[tournament_fitness.index(max(tournament_fitness))]
            parents.append(population[winner])
        return parents

    def crossover(self, parent1, parent2):
        """Single point crossover"""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, solution):
        """Mutation with random gene replacement"""
        for i in range(len(solution)):
            if random.random() < self.mutation_rate:
                if i % 2 == 0:
                    solution[i] = random.randint(0, self.game.grid_width - 1)
                else:
                    solution[i] = random.randint(0, self.game.grid_height - 1)
        return solution

    def save_training_history(self):
        """Save training history and parameters to a file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'training_history_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        print(f"Training history saved to {filename}")
        
        # Save config separately
        config_filename = 'config_train.json'
        config_data = {
            'timestamp': timestamp,
            'args': vars(self.args),
            'training_params': self.training_history['params']
        }
        with open(config_filename, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Training config saved to {config_filename}")

    def train(self):
        """Train using custom genetic algorithm to find optimal starting positions"""
        print("Starting training...")
        print(f"Initial cells: {self.game.config.initial_cells}")
        print(f"Evolution steps: {self.game.config.evolution_steps}")

        # Generate initial population
        population = self.generate_initial_population()

        best_overall_fitness = 0
        best_overall_solution = None

        # Evolution loop
        for generation in range(self.num_generations):
            # Calculate fitness for current population
            fitness_scores = [self.calculate_fitness(
                [(int(solution[i]), int(solution[i+1])) 
                 for i in range(0, len(solution), 2)]
            ) for solution in population]

            # Track best solution in this generation
            current_best_index = fitness_scores.index(max(fitness_scores))
            current_best_fitness = fitness_scores[current_best_index]
            current_best_solution = population[current_best_index]

            # Save generation statistics
            self.training_history['best_fitness_per_gen'].append(current_best_fitness)
            self.training_history['avg_fitness_per_gen'].append(np.mean(fitness_scores))
            self.training_history['fitness_history'].append(list(fitness_scores))

            # Update overall best
            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_solution = current_best_solution

            print(f"Generation {generation}: Best Fitness = {current_best_fitness:.2f}")

            # Selection
            parents = self.select_parents(population, fitness_scores)

            # Next generation
            next_population = []

            # Elitism: keep best solutions
            sorted_indices = sorted(range(len(fitness_scores)), 
                                    key=lambda k: fitness_scores[k], 
                                    reverse=True)
            elite = [population[i] for i in sorted_indices[:self.elite_size]]
            next_population.extend(elite)

            # Crossover and mutation
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1))
                if len(next_population) < self.population_size:
                    next_population.append(self.mutate(child2))

            population = next_population

        # Final solution
        if best_overall_solution is None:
            print("Training failed to find a solution")
            return

        best_positions = [(int(best_overall_solution[i]), int(best_overall_solution[i+1])) 
                          for i in range(0, len(best_overall_solution), 2)]
        
        print(f"\nTraining completed!")
        print(f"Best fitness found: {best_overall_fitness:.2f}")
       
        print(f"Args: cells={self.args.cells}, training_attempts={self.args.training_attempts}, steps={self.args.steps}, grid_file={self.args.grid_file}")
        print(f"Saving {len(best_positions)} positions to best_positions.json")
        
        # Save the training history
        self.save_training_history()
        
        # Save the best positions
        with open('best_positions.json', 'w') as f:
            json.dump(best_positions, f)