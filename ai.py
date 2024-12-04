import numpy as np
import pygad

class GameOfLifeOptimizer:
    def __init__(self, grid_size, rules, focus="balanced"):
        """
        Initialiser l'optimisateur
        Args:
            grid_size (tuple): Dimensions de la grille (rows, cols)
            rules (list): Règles de Conway
            focus (str): Critère de fitness ('stability', 'oscillation', 'movement', 'balanced')
        """
        self.grid_size = grid_size
        self.rules = rules
        self.focus = focus

    def update_grid(self, grid):
        """Mettre à jour la grille selon les règles"""
        new_grid = np.zeros_like(grid)
        padded_grid = np.pad(grid, pad_width=1, mode="constant", constant_values=0)

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

    def calculate_fitness(self, initial_positions, max_generations=20):
        """
        Calculer le score de fitness basé sur l'évolution
        Args:
            initial_positions (list): Liste des positions des cellules vivantes [(x, y), ...]
            max_generations (int): Nombre de générations à simuler
        Returns:
            float: Score de fitness
        """
        # Vérifier que les positions sont dans les limites de la grille
        filtered_positions = [
            (x, y) for x, y in initial_positions 
            if 0 <= x < self.grid_size[1] and 0 <= y < self.grid_size[0]
        ]

        grid = np.zeros(self.grid_size, dtype=int)
        for x, y in filtered_positions:
            grid[y, x] = 1  # Attention à l'ordre y, x

        temp_grid = grid.copy()
        grid_history = [temp_grid.copy()]

        for _ in range(max_generations):
            temp_grid = self.update_grid(temp_grid)
            grid_history.append(temp_grid.copy())

        # Critères de fitness multiples
        return self.calculate_pattern_fitness(grid_history)

    def calculate_pattern_fitness(self, grid_history):
        """
        Calculer la fitness basée sur différents critères
        Returns:
            float: Score de fitness
        """
        # Diversité des patterns
        unique_patterns = len(set(map(lambda g: g.tobytes(), grid_history)))
        
        # Nombre de cellules vivantes
        live_cell_scores = [np.sum(grid) for grid in grid_history]
        
        # Stabilité (variation minimale)
        stability_scores = [np.sum(np.abs(grid_history[i] - grid_history[i-1])) for i in range(1, len(grid_history))]
        
        # Combinaison des scores selon le focus
        if self.focus == 'stability':
            return unique_patterns * (1 / (np.mean(stability_scores) + 1))
        elif self.focus == 'movement':
            return unique_patterns * np.std(live_cell_scores)
        elif self.focus == 'oscillation':
            return unique_patterns * len(grid_history)
        else:  # balanced
            return (unique_patterns * 
                    np.mean(live_cell_scores) * 
                    (1 / (np.mean(stability_scores) + 1)) * 
                    np.std(live_cell_scores))

    def optimize_layout(self, num_positions=10, attempts=20):
        """
        Optimiser la disposition des cellules initiales
        Args:
            num_positions (int): Nombre de positions à optimiser
            attempts (int): Nombre de tentatives d'optimisation
        Returns:
            list: Meilleures positions des cellules
            """
        def fitness_func(ga_instance, solution, solution_idx):
            # Convertir la solution en paires de coordonnées
            positions = [
                (int(solution[i]), int(solution[i+1])) 
                for i in range(0, len(solution), 2)
            ]
            return self.calculate_fitness(positions)

        best_positions = None
        best_fitness = float('-inf')

        for _ in range(attempts):
            ga_instance = pygad.GA(
            num_generations=20,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=num_positions * 2,
            gene_space=[
                {"low": 0, "high": self.grid_size[1] - 1} if i % 2 == 0 
                else {"low": 0, "high": self.grid_size[0] - 1} 
                for i in range(num_positions * 2)
            ],
            mutation_num_genes=2,
            mutation_type="random",
            on_generation=lambda ga: print(f"Génération {ga.generations_completed}")  # Remplace delay_after_gen
        )


            ga_instance.run()
            solution, fitness, _ = ga_instance.best_solution()

            # Mettre à jour les meilleures positions si nécessaire
            if fitness > best_fitness:
                best_fitness = fitness
                best_positions = [
                    (int(solution[i]), int(solution[i+1])) 
                    for i in range(0, len(solution), 2)
                ]

        print(f"🎯 Meilleures positions trouvées - Fitness: {best_fitness}")
        return best_positions or []
def main():
    # Test de l'optimisateur
    optimizer = GameOfLifeOptimizer((50, 50), [2, 3, 3], focus='balanced')
    best_positions = optimizer.optimize_layout()
    print("Positions optimisées:", best_positions)

if __name__ == "__main__":
    main()