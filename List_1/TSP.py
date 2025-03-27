import csv
import heapq
import sys
import time
import random
from collections import defaultdict
from datetime import datetime, timedelta
import math


def parse_time(time_str):
    hours, minutes, seconds = map(int, time_str.split(":"))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours} h")
    if minutes > 0:
        parts.append(f"{minutes} min")
    if seconds > 0:
        parts.append(f"{seconds} sec")
    return " ".join(parts)


def load_graph(file_path):
    graph = defaultdict(list)
    stops = defaultdict(list)

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                departure_time = parse_time(str(row['departure_time']))
                arrival_time = parse_time(str(row['arrival_time']))
                travel_time = (arrival_time - departure_time).seconds

                start_lat = float(row['start_stop_lat'])
                start_lon = float(row['start_stop_lon'])
                end_lat = float(row['end_stop_lat'])
                end_lon = float(row['end_stop_lon'])

                stops[row['start_stop']].append((start_lat, start_lon))
                stops[row['end_stop']].append((end_lat, end_lon))

                graph[row['start_stop']].append(
                    (str(row['end_stop']), str(row['line']), departure_time, arrival_time, travel_time))
            except ValueError as e:
                print(f"Warning: Skipping row due to error: {e}")

    # Merging multiple stops with the same name
    averaged_stops = {}
    for stop, locations in stops.items():
        avg_lat = sum(lat for lat, lon in locations) / len(locations)
        avg_lon = sum(lon for lat, lon in locations) / len(locations)
        averaged_stops[stop] = (avg_lat, avg_lon)

    return graph, averaged_stops


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2)
         * math.sin(delta_lambda / 2.0) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def dijkstra(graph, start, end, start_time, add_astar_coeff):
    queue = [(0, start, start_time, None, [], 0)]
    visited = {}
    end_min_cost = float('inf')
    end_path = ()
    end_min_travel_time = float('inf')

    while queue:
        cost, node, current_time, current_line, path, actual_travel_time = heapq.heappop(queue)

        if node in visited and visited[node] <= current_time:
            continue

        if cost > end_min_cost:
            continue

        path = path + [(current_line if current_line else "START", node, current_time)]

        #print(f"Actual: {actual_travel_time}, Cost: {cost}")

        if node == end:
            if actual_travel_time < end_min_travel_time:
                end_min_travel_time = actual_travel_time
                end_path = path
                end_min_cost = cost
            continue

        visited[node] = current_time

        for neighbor, line, dep_time, arr_time, travel_time in graph.get(node, []):
            if dep_time >= current_time:
                wait_time = (dep_time - current_time).seconds
                real_travel_time = actual_travel_time + travel_time + wait_time

                heuristic = 0
                if add_astar_coeff:
                    lat1, lon1 = stops[neighbor]
                    lat2, lon2 = stops[end]
                    heuristic = haversine(lat1, lon1, lat2, lon2) * 120  # 1 km = 120 s

                heapq.heappush(queue, (real_travel_time + heuristic, neighbor,
                                       arr_time, line, path, real_travel_time))

    #print(f"Final Actual: {end_min_travel_time}, Final Cost: {end_min_cost}")
    return end_min_travel_time, end_path


def dijkstra_min_transfers(graph, start, end, start_time, add_astar_coeff):
    queue = [(0, start, start_time, None, [], 0)]
    visited = {}
    end_min_transfers = float('inf')
    end_path = ()
    end_min_travel_time = float('inf')

    while queue:
        transfers, node, current_time, current_line, path, actual_travel_time = heapq.heappop(queue)

        if node in visited and visited[node] <= (transfers, current_time):
            continue

        if transfers > end_min_transfers:
            continue

        path = path + [(current_line if current_line else "START", node, current_time)]

        if node == end:
            if (transfers < end_min_transfers or
                    (transfers == end_min_transfers and actual_travel_time < end_min_travel_time)):
                end_min_transfers = transfers
                end_path = path
                end_min_travel_time = actual_travel_time
            continue

        visited[node] = (transfers, current_time)

        for neighbor, line, dep_time, arr_time, travel_time in graph.get(node, []):
            if dep_time >= current_time:
                wait_time = (dep_time - current_time).seconds
                real_travel_time = actual_travel_time + travel_time + wait_time

                new_transfers = transfers + (1 if line != current_line and current_line is not None else 0)

                heuristic = 0
                if add_astar_coeff:
                    lat1, lon1 = stops[neighbor]
                    lat2, lon2 = stops[end]
                    heuristic = haversine(lat1, lon1, lat2, lon2) * 0.001  # 1 km = 0.001 stop

                heapq.heappush(queue, (new_transfers + heuristic, neighbor, arr_time, line, path, real_travel_time))

    return end_min_transfers, end_path


def choose_dijkstra(graph, start, end, start_time, criterion):
    if criterion == 't':
        travel_time, path = dijkstra(graph, start, end, start_time, True)

    elif criterion == 'p':
        travel_time, path = dijkstra_min_transfers(graph, start, end, start_time, True)

    else:
        raise NotImplementedError("Incorrect criterion. Please choose 't' or 'p'.")

    return travel_time, path


def compute_distance_matrix(graph, stops, locations, start_time, criterion):
    distance_matrix = {}
    for i in locations:
        distance_matrix[i] = {}
        for j in locations:
            if i != j:
                distance_matrix[i][j], _ = choose_dijkstra(graph, i, j, start_time, 't')
    return distance_matrix


def local_search(graph, stops, locations, start_time, distance_matrix, max_iterations=100):

    best_route = locations[:]
    random.shuffle(best_route[1:-1])
    best_cost = calculate_route_cost(graph, best_route, start_time, distance_matrix)

    for _ in range(max_iterations):
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                new_route = best_route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                new_cost = calculate_route_cost(graph, new_route, start_time, distance_matrix)

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = new_route
                    improved = True
        if not improved:
            break

    return best_route, best_cost


def tabu_search(graph, stops, locations, start_time, distance_matrix, max_iterations=100, tabu_size=10, stagnation_limit=1000):

    best_route = locations[:]

    if best_route[-1] != best_route[0]:
        best_route.append(best_route[0])

    random.shuffle(best_route[1:-1])
    best_cost = calculate_route_cost(graph, best_route, start_time, distance_matrix)

    tabu_list = []
    best_solution = best_route[:]
    best_solution_cost = best_cost
    stagnation_counter = 0

    for iteration in range(max_iterations):
        best_candidate = None
        best_candidate_cost = float('inf')

        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                candidate = best_route[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                candidate_cost = calculate_route_cost(graph, candidate, start_time, distance_matrix)

                move = (candidate[i], candidate[j])

                if move in tabu_list and candidate_cost >= best_solution_cost:
                    continue

                if candidate_cost < best_candidate_cost:
                    best_candidate = candidate
                    best_candidate_cost = candidate_cost

        if best_candidate:
            best_route = best_candidate
            best_cost = best_candidate_cost
            if best_cost < best_solution_cost:
                best_solution = best_route[:]
                best_solution_cost = best_cost
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            move = (best_candidate[i], best_candidate[j])
            tabu_list.append(move)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

        else:
            stagnation_counter += 1

        if stagnation_counter >= stagnation_limit:
            print("Tabu Search: Terminated because of stagnation.")
            break

    return best_solution, best_solution_cost


def calculate_route_cost(graph, route, start_time, distance_matrix):
    # print("Location_index_map:", location_index_map)
    # print("Route:", route)

    total_cost = 0
    current_time = start_time
    for i in range(len(route) - 1):

        travel_time = distance_matrix[route[i]][route[i+1]]

        total_cost += travel_time
        current_time += timedelta(seconds=travel_time)
    return total_cost


if __name__ == "__main__":
    graph, stops = load_graph("Datasource/data.csv")
    locations = "OSIEDLE SOBIESKIEGO;PL. GRUNWALDZKI;DWORZEC GŁÓWNY;Kowale (Stacja kolejowa);KRZYKI".split(";")
    start_time = "05:12:00"
    criterion = 't'

    start_time = parse_time(start_time)

    distance_matrix = compute_distance_matrix(graph, stops, locations, start_time, criterion)

    start_time_measure = time.time()
    best_route, best_cost = local_search(graph, stops, locations + [locations[0]], start_time, distance_matrix)
    end_time_measure = time.time()
    print("Best route (Local Search):", " -> ".join(best_route))
    print("Total cost:", format_duration(best_cost))
    print(f"\nExecution time: {end_time_measure - start_time_measure:.4f} s", file=sys.stderr)

    start_time_measure = time.time()
    best_route, best_cost = tabu_search(graph, stops, locations + [locations[0]], start_time, distance_matrix)
    end_time_measure = time.time()
    print("Best route (Tabu Search):", " -> ".join(best_route))
    print("Total cost:", format_duration(best_cost))
    print(f"\nExecution time: {end_time_measure - start_time_measure:.4f} s", file=sys.stderr)

