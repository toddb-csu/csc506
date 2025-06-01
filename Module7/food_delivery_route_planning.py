# Todd Bartoszkiewicz
# CSC506: Introduction to Data Structures and Algorithms
# Module 7: Critical Thinking
import heapq
from collections import defaultdict
import random


def dijkstra(dijkstra_graph, start, end):
    distances = {node: float('inf') for node in dijkstra_graph}
    previous_nodes = {node: None for node in dijkstra_graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue

        if current_node == end:
            break

        for neighbor in dijkstra_graph[current_node]:
            new_weight = dijkstra_graph[current_node][neighbor]['base_time'] * dijkstra_graph[current_node][neighbor]['traffic_factor']
            new_distance = current_distance + new_weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (new_distance, neighbor))

    path = []
    current = end
    while current:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()

    return path, distances[end] if path[0] == start else None


if __name__ == "__main__":
    # Initialize graph with travel times
    graph = defaultdict(dict)
    traffic_updates = {}

    graph['Restaurant']['A'] = {'base_time': 5, 'traffic_factor': 1.0}
    graph['A']['Restaurant'] = {'base_time': 5, 'traffic_factor': 1.0}
    graph['A']['B'] = {'base_time': 8, 'traffic_factor': 1.0}
    graph['B']['A'] = {'base_time': 8, 'traffic_factor': 1.0}
    graph['B']['C'] = {'base_time': 6, 'traffic_factor': 1.0}
    graph['C']['B'] = {'base_time': 6, 'traffic_factor': 1.0}
    graph['C']['D'] = {'base_time': 7, 'traffic_factor': 1.0}
    graph['D']['C'] = {'base_time': 7, 'traffic_factor': 1.0}
    graph['D']['E'] = {'base_time': 4, 'traffic_factor': 1.0}
    graph['E']['D'] = {'base_time': 4, 'traffic_factor': 1.0}
    graph['E']['Customer'] = {'base_time': 3, 'traffic_factor': 1.0}
    graph['Customer']['E'] = {'base_time': 3, 'traffic_factor': 1.0}
    graph['A']['D'] = {'base_time': 15, 'traffic_factor': 1.0}
    graph['D']['A'] = {'base_time': 15, 'traffic_factor': 1.0}
    graph['B']['E'] = {'base_time': 10, 'traffic_factor': 1.0}
    graph['E']['B'] = {'base_time': 10, 'traffic_factor': 1.0}

    # Find optimal path without traffic
    path_base, time_base = dijkstra(graph, 'Restaurant', 'Customer')
    print(f"Base Optimal Path: {' -> '.join(path_base)}")
    print(f"Estimated Time: {time_base:.1f} minutes\n")

    # Apply traffic simulations
    traffic_edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('A', 'D'), ('B', 'E')]
    for u, v in traffic_edges:
        # 1x-3x delay
        traffic_factor = random.uniform(1.0, 3.0)
        if u in graph and v in graph[u]:
            graph[u][v]['traffic_factor'] = traffic_factor
        if v in graph and u in graph[v]:
            graph[u][v]['traffic_factor'] = traffic_factor

    # Find optimal path with traffic
    path_traffic, time_traffic = dijkstra(graph, 'Restaurant', 'Customer')
    print(f"Traffic-Adjusted Optimal Path: {' -> '.join(path_traffic)}")
    print(f"Estimated Time With Traffic: {time_traffic:.1f} minutes\n")

    # Display traffic factors
    print("Traffic Conditions:")
    for u in graph:
        for v in graph[u]:
            if u < v:
                weight = graph[u][v]['base_time'] * graph[u][v]['traffic_factor']
                base = graph[u][v]['base_time']
                factor = graph[u][v]['traffic_factor']
                print(f"{u}-{v}: Base={base} min, Factor={factor:.2f}x -> {weight:.1f} min")
