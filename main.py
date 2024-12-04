import pygame
import numpy as np
import pygad
import time

WIDTH, HEIGHT = 800, 600
CELL_SIZE = 10
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class GameOfLifeAI:
    def __init__(self, focus='balanced'):
        """
        Initialize the Game of Life AI
        
        Args:
            focus (str): Focus of fitness function
            Options: 
            - 'stability': Focus on stable patterns
            - 'oscillation': Focus on oscillating patterns
            - 'movement': Focus on moving patterns
            - 'balanced': Consider all aspects
        """
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"Game of Life AI - {focus.capitalize()} Focus")
        self.clock = pygame.time.Clock()
        
        self.focus = focus
        self.reset_grid()
        
        self.rules = [2, 3, 3]
        self.generation = 0

    def reset_grid(self):
        """Réinitialise la grille de manière aléatoire"""
        self.grid = np.random.choice([0, 1], 
                                     size=(GRID_HEIGHT, GRID_WIDTH), 
                                     p=[0.85, 0.15])  # Moins de cellules vivantes

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

    def draw_grid(self, grid):
        """Dessiner la grille"""
        self.screen.fill(BLACK)
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                color = WHITE if grid[row, col] == 1 else BLACK
                pygame.draw.rect(self.screen, color, 
                                 (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()

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
            # Score de stabilité : moins de changements
            fitness = max_generations - sum(np.sum(np.abs(grid_history[i] - grid_history[i-1])) for i in range(1, len(grid_history)))
        
        elif self.focus == 'oscillation':
            # Score d'oscillation : recherche des motifs cycliques
            fitness = self.detect_oscillation(grid_history)
        
        elif self.focus == 'movement':
            # Score de mouvement : déplacement du centre de masse
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
            # Distance totale parcourue
            total_distance = sum(np.linalg.norm(np.array(centers[i]) - np.array(centers[i-1])) for i in range(1, len(centers)))
            return total_distance
        return 0

    def optimize_rules(self):
        """Optimiser les règles avec un algorithme génétique"""
        def fitness_func(ga_instance, solution, solution_idx):
            return self.calculate_fitness(self.grid, solution)

        ga_instance = pygad.GA(
            num_generations=10,
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
        """Boucle principale du jeu"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    # Changer le focus avec des touches
                    if event.key == pygame.K_s:
                        self.focus = 'stability'
                        print("🎯 Focus: Stabilité")
                    elif event.key == pygame.K_o:
                        self.focus = 'oscillation'
                        print("🎯 Focus: Oscillation")
                    elif event.key == pygame.K_m:
                        self.focus = 'movement'
                        print("🎯 Focus: Mouvement")
                    elif event.key == pygame.K_b:
                        self.focus = 'balanced'
                        print("🎯 Focus: Équilibré")

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
    game = GameOfLifeAI(focus='balanced')
    game.run()