import pygame
import numpy as np
import pygad

WIDTH, HEIGHT = 800, 600
CELL_SIZE = 10
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class GameOfLifeAI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Game of Life with AI")
        self.clock = pygame.time.Clock()
        
        # Méthode de réinitialisation séparée
        self.reset_grid()
        
        self.rules = [2, 3, 3]
        self.generation = 0

    def reset_grid(self):
        """Réinitialise la grille de manière aléatoire"""
        self.grid = np.random.choice([0, 1], 
                                     size=(GRID_HEIGHT, GRID_WIDTH), 
                                     p=[0.85, 0.15])  # Moins de cellules vivantes

    def update_grid(self, grid, rules):
        min_survive, max_survive, birth = rules
        new_grid = np.zeros_like(grid)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                neighbors = np.sum(grid[max(0, row-1):min(grid.shape[0], row+2),
                                         max(0, col-1):min(grid.shape[1], col+2)]) - grid[row, col]
                
                if grid[row, col] == 1 and (neighbors < min_survive or neighbors > max_survive):
                    new_grid[row, col] = 0
                elif grid[row, col] == 0 and neighbors == birth:
                    new_grid[row, col] = 1
                else:
                    new_grid[row, col] = grid[row, col]
        return new_grid

    def draw_grid(self, grid):
        self.screen.fill(BLACK)
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                color = WHITE if grid[row, col] == 1 else BLACK
                pygame.draw.rect(self.screen, color, 
                                 (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()

    def optimize_rules(self):
        def fitness_func(ga_instance, solution, solution_idx):
            temp_grid = self.grid.copy()
            for _ in range(10):
                temp_grid = self.update_grid(temp_grid, solution)
            
            # Favoriser configurations stables et dynamiques
            live_cells = np.sum(temp_grid)
            return live_cells * (1 + np.std(temp_grid))

        ga_instance = pygad.GA(
            num_generations=5,  # 5 générations internes de l'algo génétique
            num_parents_mating=3,
            fitness_func=fitness_func,
            sol_per_pop=8,
            num_genes=3,
            gene_space=[(1, 3), (3, 5), (2, 4)],
            mutation_num_genes=1,
            on_generation=lambda ga: print(f"Genetic Algorithm Gen {ga.generations_completed}")
        )

        ga_instance.run()
        solution, fitness, _ = ga_instance.best_solution()
        return [int(solution[0]), int(solution[1]), int(solution[2])]

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.grid = self.update_grid(self.grid, self.rules)
            self.draw_grid(self.grid)

            # Optimiser et reset à chaque 50 générations
            if self.generation % 50 == 0 and self.generation > 0:
                print(f"🔄 Optimizing rules at generation {self.generation}")
                self.rules = self.optimize_rules()
                self.reset_grid()  # Réinitialisation après optimisation
                print(f"🆕 New rules: {self.rules}")

            self.clock.tick(10)
            self.generation += 1

        pygame.quit()

if __name__ == "__main__":
    game = GameOfLifeAI()
    game.run()