import pygame
import numpy as np
import pygad
import json
import argparse
from dataclasses import dataclass

@dataclass
class GameConfig:
    width: int = 800
    height: int = 600
    cell_size: int = 10
    initial_cells: int = 50
    evolution_steps: int = 20
    training_attempts: int = 20

class GameOfLife:
    def __init__(self, config: GameConfig):
        self.config = config
        self.grid_width = config.width // config.cell_size
        self.grid_height = config.height // config.cell_size
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        self.rules = [2, 3, 3]

    def update_grid(self):
        new_grid = np.zeros_like(self.grid)
        padded_grid = np.pad(self.grid, pad_width=1, mode="constant")
        
        for row in range(1, padded_grid.shape[0] - 1):
            for col in range(1, padded_grid.shape[1] - 1):
                neighbors = np.sum(padded_grid[row-1:row+2, col-1:col+2]) - padded_grid[row, col]
                if padded_grid[row, col] == 1 and (neighbors < self.rules[0] or neighbors > self.rules[1]):
                    new_grid[row-1, col-1] = 0
                elif padded_grid[row, col] == 0 and neighbors == self.rules[2]:
                    new_grid[row-1, col-1] = 1
                else:
                    new_grid[row-1, col-1] = padded_grid[row, col]
        return new_grid

    def calculate_pattern_fitness(self, grid_history):
        unique_patterns = len(set(map(lambda g: g.tobytes(), grid_history)))
        live_cell_scores = [np.sum(grid) for grid in grid_history]
        stability_scores = [np.sum(np.abs(grid_history[i] - grid_history[i-1])) 
                          for i in range(1, len(grid_history))]
        
        return (unique_patterns * 
                np.mean(live_cell_scores) * 
                (1 / (np.mean(stability_scores) + 1)) * 
                np.std(live_cell_scores))

    def calculate_fitness(self, positions):
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        for x, y in positions:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                self.grid[y, x] = 1

        grid_history = [self.grid.copy()]
        temp_grid = self.grid.copy()
        
        for _ in range(self.config.evolution_steps):
            temp_grid = self.update_grid()
            grid_history.append(temp_grid.copy())
            
        return self.calculate_pattern_fitness(grid_history)

    def train(self):
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
            print(f"Testing solution {solution_idx} - Fitness: {fitness:.2f}")
            return fitness

        print("Starting training...")
        print(f"Initial cells: {self.config.initial_cells}")
        print(f"Evolution steps: {self.config.evolution_steps}")

        num_genes = self.config.initial_cells * 2
        ga_instance = pygad.GA(
            num_generations=20,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=num_genes,
            gene_space=[ 
                {"low": 0, "high": self.grid_width - 1} if i % 2 == 0 
                else {"low": 0, "high": self.grid_height - 1} 
                for i in range(num_genes)
            ],
            mutation_num_genes=2,
            mutation_type="random",
            on_generation=on_generation
        )

        ga_instance.run()
        solution, fitness, _ = ga_instance.best_solution()
        
        best_positions = [(int(solution[i]), int(solution[i+1])) 
                         for i in range(0, len(solution), 2)]
        
        print(f"\nTraining completed!")
        print(f"Best fitness found: {fitness:.2f}")
        print(f"Saving {len(best_positions)} positions to best_positions.json")
        
        with open('best_positions.json', 'w') as f:
            json.dump(best_positions, f)

    def visualize(self):
        pygame.init()
        screen = pygame.display.set_mode((self.config.width, self.config.height))
        clock = pygame.time.Clock()
        
        try:
            with open('best_positions.json', 'r') as f:
                positions = json.load(f)
                print(f"Loaded {len(positions)} positions from best_positions.json")
        except FileNotFoundError:
            print("Error: best_positions.json not found. Run training first.")
            return

        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        for x, y in positions:
            self.grid[y, x] = 1

        running = True
        generation = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    if self.grid[row, col] == 1:
                        pygame.draw.rect(screen, (255, 255, 255),
                                     (col * self.config.cell_size, 
                                      row * self.config.cell_size,
                                      self.config.cell_size - 1,
                                      self.config.cell_size - 1))

            pygame.display.flip()
            self.grid = self.update_grid()
            clock.tick(10)
            generation += 1
            pygame.display.set_caption(f"Generation: {generation}")

        pygame.quit()

def visualize_initial_grid(config: GameConfig):
    """Affiche uniquement la grille de départ, avec option pour continuer le jeu."""
    pygame.init()
    screen = pygame.display.set_mode((config.width, config.height))
    clock = pygame.time.Clock()

    try:
        with open('best_positions.json', 'r') as f:
            positions = json.load(f)
            print(f"Loaded {len(positions)} positions from best_positions.json")
    except FileNotFoundError:
        print("Error: best_positions.json not found. Run training first.")
        return

    # Initialiser la grille
    grid_width = config.width // config.cell_size
    grid_height = config.height // config.cell_size
    grid = np.zeros((grid_height, grid_width), dtype=int)

    for x, y in positions:
        grid[y, x] = 1

    running = True
    evolving = False
    generation = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:  # Appuyer sur E pour commencer l'évolution
                    evolving = True
                    print("🔄 Lancement de l'évolution classique !")

        screen.fill((0, 0, 0))

        # Dessiner la grille
        for row in range(grid_height):
            for col in range(grid_width):
                if grid[row, col] == 1:
                    pygame.draw.rect(screen, (255, 255, 255),
                                     (col * config.cell_size,
                                      row * config.cell_size,
                                      config.cell_size - 1,
                                      config.cell_size - 1))

        pygame.display.flip()

        # Si le mode évolution est activé, mettre à jour la grille
        if evolving:
            grid = update_grid(grid, [2, 3, 3])  # Appeler update_grid avec les règles classiques
            generation += 1
            pygame.display.set_caption(f"Game of Life - Generation: {generation}")

        clock.tick(10)

    pygame.quit()


def update_grid(grid, rules):
    """Met à jour la grille selon les règles classiques de Conway."""
    new_grid = np.zeros_like(grid)
    padded_grid = np.pad(grid, pad_width=1, mode="constant", constant_values=0)

    for row in range(1, padded_grid.shape[0] - 1):
        for col in range(1, padded_grid.shape[1] - 1):
            neighbors = np.sum(padded_grid[row-1:row+2, col-1:col+2]) - padded_grid[row, col]

            if padded_grid[row, col] == 1 and (neighbors < rules[0] or neighbors > rules[1]):
                new_grid[row-1, col-1] = 0
            elif padded_grid[row, col] == 0 and neighbors == rules[2]:
                new_grid[row-1, col-1] = 1
            else:
                new_grid[row-1, col-1] = padded_grid[row, col]

    return new_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'visualize', 'visualize-start'])
    parser.add_argument('--cells', type=int, default=50)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()

    config = GameConfig(
        initial_cells=args.cells,
        evolution_steps=args.steps
    )

    game = GameOfLife(config)

    if args.mode == 'train':
        game.train()
    elif args.mode == 'visualize':
        game.visualize()
    elif args.mode == 'visualize-start':
        visualize_initial_grid(config)


if __name__ == "__main__":
    main()
