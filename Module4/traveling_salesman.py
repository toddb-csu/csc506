# Todd Bartoszkiewicz
# CSC506: Introduction to Data Structures and Algorithms
# Module 4: Portfolio Milestone
import math
import random
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import heapq

sys.setrecursionlimit(2000)


class City:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def distance(self, other_city):
        return math.sqrt((self.x - other_city.x)**2 + (self.y - other_city.y)**2)

    def __repr__(self):
        return f"City(id={self.id}, x={self.x}, y={self.y})"


def generate_cities(num_cities):
    cities = []
    northernmost_point = 49.0
    southernmost_point = 25.2
    westernmost_point = -124.5
    easternmost_point = -67.0
    for i in range(num_cities):
        x = random.uniform(westernmost_point, easternmost_point)
        y = random.uniform(southernmost_point, northernmost_point)
        cities.append(City(i, x, y))
    return cities


def calculate_tour_length(tour, cities):
    total_length = 0
    num_cities = len(tour)
    for i in range(num_cities):
        city1_index = tour[i]
        city2_index = tour[(i + 1) % num_cities] # Connect back to the start
        total_length += cities[city1_index].distance(cities[city2_index])
    return total_length


# Nearest Neighbor
def nearest_neighbor(cities):
    num_cities = len(cities)
    if num_cities == 0:
        return [], 0

    start_city_index = 0
    tour = [start_city_index]
    visited = {start_city_index}

    current_city_index = start_city_index

    while len(visited) < num_cities:
        nearest_city_index = -1
        min_distance = float('inf')

        for i in range(num_cities):
            if i not in visited:
                dist = cities[current_city_index].distance(cities[i])
                if dist < min_distance:
                    min_distance = dist
                    nearest_city_index = i

        if nearest_city_index != -1:
            tour.append(nearest_city_index)
            visited.add(nearest_city_index)
            current_city_index = nearest_city_index
        else:
            break

    return tour, calculate_tour_length(tour, cities)


def mst_approximation(cities):
    num_cities = len(cities)
    if num_cities == 0:
        return [], 0
    if num_cities == 1:
        return [0], 0

    adj = {i: [] for i in range(num_cities)}
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist = cities[i].distance(cities[j])
            adj[i].append((j, dist))
            adj[j].append((i, dist))

    mst_adj = {i: [] for i in range(num_cities)}
    visited_mst = [False] * num_cities
    min_edge = [(0, 0)]
    heapq.heapify(min_edge)
    total_mst_weight = 0

    while min_edge:
        weight, u = heapq.heappop(min_edge)

        if visited_mst[u]:
            continue

        visited_mst[u] = True
        total_mst_weight += weight

        if u != 0 or weight != 0:
             pass

    mst_adj = {i: [] for i in range(num_cities)}
    visited_prim = [False] * num_cities
    min_dist = [float('inf')] * num_cities
    parent = [-1] * num_cities
    min_dist[0] = 0

    pq = [(0, 0)]

    while pq:
        dist, u = heapq.heappop(pq)

        if visited_prim[u]:
            continue

        visited_prim[u] = True

        # If it's not the first node and has a parent, add edge to MST adj list
        if parent[u] != -1:
            p = parent[u]
            mst_adj[u].append(p)
            mst_adj[p].append(u)

        # Explore neighbors
        for v in range(num_cities):
            if u != v and not visited_prim[v]:
                 weight = cities[u].distance(cities[v])
                 if weight < min_dist[v]:
                    min_dist[v] = weight
                    parent[v] = u
                    heapq.heappush(pq, (weight, v))

    tour_order = []
    visited_dfs = [False] * num_cities

    def dfs(u):
        visited_dfs[u] = True
        tour_order.append(u)
        for v in mst_adj[u]:
            if not visited_dfs[v]:
                dfs(v)

    dfs(0)

    return tour_order, calculate_tour_length(tour_order, cities)


# Genetic Algorithm
def genetic_algorithm(cities, pop_size=100, generations=500, mutation_rate=0.01, elitism_count=1):
    num_cities = len(cities)
    if num_cities == 0:
        return [], 0
    if num_cities == 1:
        return [0], 0

    def create_individual():
        tour = list(range(num_cities))
        random.shuffle(tour)
        return tour

    def calculate_fitness(tour):
        length = calculate_tour_length(tour, cities)
        return 1 / length if length > 0 else 0

    def order_crossover(parent1, parent2):
        size = len(parent1)
        child = [-1] * size
        start, end = sorted(random.sample(range(size), 2))

        child[start:end+1] = parent1[start:end+1]

        parent2_genes = [gene for gene in parent2 if gene not in child]
        current_pos = (end + 1) % size
        for gene in parent2_genes:
            while child[current_pos] != -1:
                current_pos = (current_pos + 1) % size
            child[current_pos] = gene

        return child

    def swap_mutation(tour):
        size = len(tour)
        if size > 1:
            idx1, idx2 = random.sample(range(size), 2)
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
        return tour

    def tournament_selection(population, fitnesses, k=3):
        tournament = random.sample(list(zip(population, fitnesses)), k)
        winner_tour, winner_fitness = max(tournament, key=lambda item: item[1])
        return winner_tour

    population = [create_individual() for _ in range(pop_size)]
    best_tour = None
    best_fitness = -1

    for generation in range(generations):
        fitnesses = [calculate_fitness(tour) for tour in population]

        current_best_fitness = max(fitnesses)
        current_best_index = fitnesses.index(current_best_fitness)

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_tour = list(population[current_best_index]) # Make a copy

        new_population = []

        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elitism_count]
        for i in elite_indices:
             new_population.append(population[i])

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            if random.random() < 0.9: # Crossover probability
                 child = order_crossover(parent1, parent2)
            else:
                 child = list(random.choice([parent1, parent2])) # Just copy one parent

            if random.random() < mutation_rate:
                 child = swap_mutation(child)

            new_population.append(child)

        population = new_population

    return best_tour, calculate_tour_length(best_tour, cities)


def evaluate_algorithms(city_sizes, num_runs_per_size=5):
    results = {
        'NN': {'lengths': [], 'times': []},
        'MST': {'lengths': [], 'times': []},
        'GA': {'lengths': [], 'times': []}
    }
    data_points = []

    for size in city_sizes:
        print(f"\nEvaluating for N = {size} cities...")
        nn_lengths, nn_times = [], []
        mst_lengths, mst_times = [], []
        ga_lengths, ga_times = [], []

        for run in range(num_runs_per_size):
            print(f"  Run {run + 1}/{num_runs_per_size}...")
            cities = generate_cities(size)

            # Nearest Neighbor
            start_time = time.time()
            nn_tour, nn_length = nearest_neighbor(cities)
            end_time = time.time()
            nn_times.append(end_time - start_time)
            nn_lengths.append(nn_length)
            print(f"    NN: Length = {nn_length:.2f}, Time = {nn_times[-1]:.4f}s")
            data_points.append({'N': size, 'Algorithm': 'NN', 'Length': nn_length, 'Time': nn_times[-1]})

            # MST
            start_time = time.time()
            mst_tour, mst_length = mst_approximation(cities)
            end_time = time.time()
            mst_times.append(end_time - start_time)
            mst_lengths.append(mst_length)
            print(f"    MST: Length = {mst_length:.2f}, Time = {mst_times[-1]:.4f}s")
            data_points.append({'N': size, 'Algorithm': 'MST', 'Length': mst_length, 'Time': mst_times[-1]})

            # Genetic Algorithm
            ga_params = {'pop_size': 100, 'generations': 500, 'mutation_rate': 0.02}
            if size > 200:
                 ga_params['generations'] = 200
                 ga_params['pop_size'] = 50

            start_time = time.time()
            ga_tour, ga_length = genetic_algorithm(cities, **ga_params)
            end_time = time.time()
            ga_times.append(end_time - start_time)
            ga_lengths.append(ga_length)
            print(f"    GA: Length = {ga_length:.2f}, Time = {ga_times[-1]:.4f}s")
            data_points.append({'N': size, 'Algorithm': 'GA', 'Length': ga_length, 'Time': ga_times[-1]})

        results['NN']['lengths'].append(np.mean(nn_lengths))
        results['NN']['times'].append(np.mean(nn_times))
        results['MST']['lengths'].append(np.mean(mst_lengths))
        results['MST']['times'].append(np.mean(mst_times))
        results['GA']['lengths'].append(np.mean(ga_lengths))
        results['GA']['times'].append(np.mean(ga_times))

    return results, data_points


def plot_results(city_sizes, results):
    algorithms = ['NN', 'MST', 'GA']
    colors = {'NN': 'blue', 'MST': 'green', 'GA': 'red'}

    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        plt.plot(city_sizes, results[algo]['lengths'], marker='o', linestyle='-', color=colors[algo], label=algo)
    plt.xlabel("Number of Cities (N)")
    plt.ylabel("Average Tour Length")
    plt.title("Average Tour Length vs. Number of Cities")
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        plt.plot(city_sizes, results[algo]['times'], marker='o', linestyle='-', color=colors[algo], label=algo)
    plt.xlabel("Number of Cities (N)")
    plt.ylabel("Average Execution Time (seconds)")
    plt.title("Average Execution Time vs. Number of Cities")
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_city_sizes = [10, 20, 30, 40, 50, 100]

    num_evaluation_runs = 5

    evaluation_results, raw_data = evaluate_algorithms(test_city_sizes, num_runs_per_size=num_evaluation_runs)

    print("\n--- Results ---")
    print("N | NN Length | NN Time (s) | MST Length | MST Time (s) | GA Length | GA Time (s)")
    print("-" * 80)
    for i, size in enumerate(test_city_sizes):
         nn_l = evaluation_results['NN']['lengths'][i]
         nn_t = evaluation_results['NN']['times'][i]
         mst_l = evaluation_results['MST']['lengths'][i]
         mst_t = evaluation_results['MST']['times'][i]
         ga_l = evaluation_results['GA']['lengths'][i]
         ga_t = evaluation_results['GA']['times'][i]
         print(f"{size} | {nn_l:.2f} | {nn_t:.4f} | {mst_l:.2f} | {mst_t:.4f} | {ga_l:.2f} | {ga_t:.4f}")

    plot_results(test_city_sizes, evaluation_results)

    if len(test_city_sizes) > 0:
        sample_size = test_city_sizes[0]
        sample_cities = generate_cities(sample_size)
        nn_tour, _ = nearest_neighbor(sample_cities)
        mst_tour, _ = mst_approximation(sample_cities)
        ga_tour, _ = genetic_algorithm(sample_cities, pop_size=50, generations=100)

        def plot_tour(tour, cities, title):
            plt.figure(figsize=(6, 6))
            x = [c.x for c in cities]
            y = [c.y for c in cities]
            plt.scatter(x, y, color='red', zorder=5)
            for i, city in enumerate(cities):
                plt.text(city.x, city.y, str(city.id), fontsize=9, ha='right')

            tour_x = [cities[tour[i]].x for i in range(len(tour))]
            tour_y = [cities[tour[i]].y for i in range(len(tour))]
            # Add closing segment
            tour_x.append(cities[tour[0]].x)
            tour_y.append(cities[tour[0]].y)

            plt.plot(tour_x, tour_y, linestyle='-', marker='o')
            plt.title(title)
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.grid(True)
            plt.show()

        plot_tour(nn_tour, sample_cities, f"NN Tour (N={sample_size})")
        plot_tour(mst_tour, sample_cities, f"MST Tour (N={sample_size})")
        plot_tour(ga_tour, sample_cities, f"GA Tour (N={sample_size})")
