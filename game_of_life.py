import pygame
import numpy as np
import json
import argparse
from dataclasses import dataclass
from game_trainer import GameTrainer

@dataclass
class GameConfig:
    width: int = 800
    height: int = 600
    cell_size: int = 10
    initial_cells: int = 500
    evolution_steps: int = 80
    training_attempts: int = 20

class GameOfLife:
    def __init__(self, config: GameConfig):
        self.config = config
        self.grid_width = config.width // config.cell_size
        self.grid_height = config.height // config.cell_size
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        self.rules = [2, 3, 3]  # Survie, surpopulation, naissance

    def update_grid(self):
        """Update the grid according to Conway's Game of Life rules"""
        new_grid = np.zeros_like(self.grid)
        padded_grid = np.pad(self.grid, pad_width=1, mode="constant")
        
        for row in range(1, padded_grid.shape[0] - 1):
            for col in range(1, padded_grid.shape[1] - 1):
                # Calculate number of living neighbors
                neighbors = np.sum(padded_grid[row-1:row+2, col-1:col+2]) - padded_grid[row, col]
                
                # Apply rules
                current_cell = padded_grid[row, col]
                if current_cell == 1:
                    # Survival rules
                    new_grid[row-1, col-1] = 1 if self.rules[0] <= neighbors <= self.rules[1] else 0
                else:
                    # Birth rule
                    new_grid[row-1, col-1] = 1 if neighbors == self.rules[2] else 0
                    
        return new_grid

    def set_initial_state(self, positions):
        """Initialize the grid with given positions"""
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        for x, y in positions:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                self.grid[y, x] = 1

    def visualize(self, start_paused=False):
        """Visualize the game state"""
        pygame.init()
        screen = pygame.display.set_mode((self.config.width, self.config.height))
        clock = pygame.time.Clock()
        
        try:
            with open('best_positions.json', 'r') as f:
                positions = json.load(f)
                print(f"Loaded {len(positions)} positions from best_positions.json")
                self.set_initial_state(positions)
        except FileNotFoundError:
            print("Error: best_positions.json not found. Run training first.")
            return

        running = True
        evolving = not start_paused
        generation = 0
        fps = 10

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_e:
                        evolving = True
                        print("🔄 Starting evolution!")
                    elif event.key == pygame.K_SPACE:
                        evolving = not evolving
                    elif event.key == pygame.K_UP:
                        fps = min(60, fps + 5)
                    elif event.key == pygame.K_DOWN:
                        fps = max(1, fps - 5)

            screen.fill((0, 0, 0))
            
            # Draw grid
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    if self.grid[row, col] == 1:
                        pygame.draw.rect(screen, (255, 255, 255),
                                     (col * self.config.cell_size, 
                                      row * self.config.cell_size,
                                      self.config.cell_size - 1,
                                      self.config.cell_size - 1))

            pygame.display.flip()
            
            if evolving:
                self.grid = self.update_grid()
                generation += 1
                pygame.display.set_caption(f"Generation: {generation} - FPS: {fps}")

            clock.tick(fps)

        pygame.quit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'visualize', 'visualize-start'])
    parser.add_argument('--cells', type=int, default=200, help="Number of initial cells")
    parser.add_argument('--training_attempts', type=int, default=10, help="Number of training attempts")
    parser.add_argument('--steps', type=int, default=20, help="Number of evolution steps") 

    args = parser.parse_args()

    config = GameConfig(    
        initial_cells=args.cells,        
        training_attempts=args.training_attempts,
        evolution_steps=args.steps
    )

    game = GameOfLife(config)

    if args.mode == 'visualize':
        game.visualize(start_paused=False)
    elif args.mode == 'visualize-start':
        game.visualize(start_paused=True)
    elif args.mode == 'train':
        trainer = GameTrainer(game)
        trainer.train()

if __name__ == "__main__":
    main()