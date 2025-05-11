import numpy as np
from scipy.spatial import Voronoi, cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pyDOE2 import lhs
from scipy.stats import qmc

def generate_constrained_population_nd(m, n, d):

    populations = []
    for _ in range(m):
        # dim=10
        # points = np.random.rand(n, d)
        # points /= np.sum(points, axis=1, keepdims=True)
        points = []
        while len(points) < n:
            point = np.random.rand(d - 1)
            last_dim = 1 - np.sum(point)
            if 0 <= last_dim <= 1:
                points.append(np.append(point, last_dim))
        populations.append(np.array(points))
    return populations

def monte_carlo_voronoi_nd(points, num_random_points=100000):

    d = points.shape[1]
    # random_points = []
    random_points = np.random.rand(num_random_points, d)

    random_points /= np.sum(random_points, axis=1, keepdims=True)
    random_points = np.array(random_points)


    # distances = np.linalg.norm(random_points[:, None, :] - points[None, :, :], axis=2)
    # closest_indices = np.argmin(distances, axis=1)

    tree = cKDTree(points)

    _, closest_indices = tree.query(random_points)

    voronoi_cells = {i: random_points[closest_indices == i] for i in range(len(points))}
    return voronoi_cells

def calculate_count_variance(voronoi_cells):
    counts = [len(cell_points) for cell_points in voronoi_cells.values()]
    if len(counts) == 0:
        return float('inf')
    counts = counts / np.sum(counts)
    return np.var(counts)

def fitness_function(area_variance):
    return 1 / ((1 + area_variance) * (1 + area_variance))

def tournament_selection(populations, fitness_values, k=3):
    selected_indices = []
    for _ in range(len(populations)):
        candidates = np.random.choice(len(populations), k, replace=False)
        best_index = candidates[np.argmax([fitness_values[i] for i in candidates])]
        selected_indices.append(best_index)
    return selected_indices

def crossover_and_mutate(parents, alpha_range=(0, 1), mutation_std=0.05):

    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            parent_a, parent_b = np.array(parents[i]), np.array(parents[i + 1])
            alpha = np.random.uniform(*alpha_range)
            child = alpha * parent_a + (1 - alpha) * parent_b

            for j in range(len(child)):

                random_dir = np.random.normal(0, mutation_std, size=child.shape[1]-1)
                last_dim = -np.sum(random_dir)
                direction = np.append(random_dir,last_dim)
                new_point = child[j] +  direction

                for dim in range(len(new_point)):
                    if new_point[dim] <0 or new_point[dim] >1:
                        if direction[dim] != 0:
                            t = (0 - child[j, dim]) / direction[dim] if new_point[dim] < 0 else (1 - child[j, dim]) /direction[dim]
                            new_point = child[j] + t * direction
                        else:
                            continue

                child[j] = new_point
            offspring.append(child)
    return np.array(offspring)

def plot_voronoi_3d(points, voronoi_cells, generation=None,sampled_points=None):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c='red', label="Population Points",
               s=100)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', label="Population Points", s=100)
    # ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c='red', label="Population Points", s=50)


    num_cells = len(voronoi_cells)
    colormap = cm.get_cmap("viridis", num_cells)
    norm = mcolors.Normalize(vmin=0, vmax=num_cells - 1)

    for i, (cell_index, cell_points) in enumerate(voronoi_cells.items()):
        if len(cell_points) > 0:
            color = colormap(norm(i))
            ax.scatter(cell_points[:, 0], cell_points[:, 1], cell_points[:, 2], color=color, alpha=0.1,
                       label=f"Cell {cell_index}")


    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    Z[Z < 0] = np.nan
    # ax.plot_surface(X, Y, Z, alpha=0.2, color='gray', label="Constrained Hyperplane")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_zlim(0, 1)
    title = "3D Voronoi Diagram on Hyperplane x1 + x2 + x3 = 1"
    if generation is not None:
        title += f" - Generation {generation}"
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_voronoi_2d(points, voronoi_cells, generation=None,sampled_points=None):

    fig, ax = plt.subplots(figsize=(10, 10))

    num_cells = len(voronoi_cells)
    colormap = cm.get_cmap("viridis", num_cells)
    norm = mcolors.Normalize(vmin=0, vmax=num_cells - 1)

    for i, (cell_index, cell_points) in enumerate(voronoi_cells.items()):
        if len(cell_points) > 0:
            color = colormap(norm(i))
            ax.scatter(cell_points[:, 0], cell_points[:, 1], color=color, alpha=0.3,
                       label=f"Cell {cell_index}")

    ax.scatter(points[:, 0], points[:, 1], c='black', label="Population Points", s=50)
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], c='red', label="Population Points", s=50)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    title = "2D Voronoi Diagram on Hyperplane x1 + x2 + x3 = 1"
    if generation is not None:
        title += f" - Generation {generation}"
    ax.set_title(title)
    ax.legend()
    plt.show()


def sample_uniformly_from_voronoi(voronoi_cells):

    sampled_points = []
    for cell_points in voronoi_cells.values():
        if len(cell_points) > 0:
            sampled_points.append(cell_points[np.random.choice(len(cell_points))])
    return np.array(sampled_points)




def sample_n_points_from_voronoi(voronoi_cells, n=16):

    sampled_points = []
    num_cells = len(voronoi_cells)
    cell_indices = list(voronoi_cells.keys())
    cell_index = 0
    while len(sampled_points) < n:
        cell_key = cell_indices[cell_index % num_cells]
        cell_points = voronoi_cells[cell_key]
        if len(cell_points) > 0:
            sampled_points.append(cell_points[np.random.choice(len(cell_points))])
        cell_index += 1
    return np.array(sampled_points)

def get_best_voronoi(m=20, n=16, d=3, num_generations=100):


    populations = generate_constrained_population_nd(m, n, d)

    fitness_values = []
    maxn = 0
    voronoi_results = []
    for pop in populations:
        pop = np.array(pop)
        voronoi_cells = monte_carlo_voronoi_nd(pop)
        voronoi_results.append(voronoi_cells)
        count_variance = calculate_count_variance(voronoi_cells)
        fitness_values.append(fitness_function(count_variance))


    for generation in range(num_generations):

        selected_indices = tournament_selection(populations, fitness_values, k=3)
        parents = [np.array(populations[i]) for i in selected_indices]

        offspring = crossover_and_mutate(parents)

        offspring_fitness = []
        for child in offspring:
            voronoi_cells = monte_carlo_voronoi_nd(np.array(child))
            voronoi_results.append(voronoi_cells)
            count_variance = calculate_count_variance(voronoi_cells)
            offspring_fitness.append(fitness_function(count_variance))

        combined_population = parents + offspring.tolist()
        combined_fitness = fitness_values[:len(parents)] + offspring_fitness

        sorted_indices = np.argsort(combined_fitness)[::-1]
        populations = [combined_population[i] for i in sorted_indices[:m]]
        fitness_values = [combined_fitness[i] for i in sorted_indices[:m]]



        best_population = np.array(populations[0])

        print(f"Generation {generation + 1}: Best Fitness = {fitness_values[0]:.8f}")
        best_voronoi_cells = monte_carlo_voronoi_nd(best_population)

            # plot_voronoi_3d(best_population, best_voronoi_cells, generation, sampled_points)


        # if d == 3 and fitness_values[0]>maxn:
        #     maxn = fitness_values[0]
        #     best_population = np.array(populations[0])
        #
        #     print(f"Generation {generation + 1}: Best Fitness = {fitness_values[0]:.8f}")
        #     # print(best_population)
        #     best_voronoi_cells = monte_carlo_voronoi_nd(best_population)
        #     sampled_points = sample_n_points_from_voronoi(best_voronoi_cells)
        #     plot_voronoi_3d(best_population, best_voronoi_cells, generation,sampled_points)

    # return populations[0], fitness_values[0]
    return best_voronoi_cells