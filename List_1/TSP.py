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


def local_search(graph, stops, locations, start_time, max_iterations=20):

    best_route = locations[:]
    random.shuffle(best_route[1:-1])
    best_cost, timings, path = calculate_route_cost(graph, best_route, start_time, criterion)
    best_timings = timings

    for _ in range(max_iterations):
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                new_route = best_route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                new_cost, timings, path = calculate_route_cost(graph, new_route, start_time, criterion)

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = new_route
                    best_timings = timings
                    improved = True
        if not improved:
            break

    return best_route, best_cost, best_timings


def tabu_search(graph, stops, locations, start_time, max_iterations=5, tabu_size=300):

    best_route = locations[:]

    if best_route[-1] != best_route[0]:
        best_route.append(best_route[0])

    random.shuffle(best_route[1:-1])
    best_cost, timings, path = calculate_route_cost(graph, best_route, start_time, criterion)

    tabu_list = []
    best_solution = best_route[:]
    best_solution_cost = best_cost
    best_timings = timings
    best_path = path

    for iteration in range(max_iterations):
        best_candidate = None
        best_candidate_cost = float('inf')
        best_candidate_timings = []
        best_candidate_path = [()]

        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                candidate = best_route[:]
                move = (candidate[j], candidate[i])

                if move in tabu_list:
                    continue

                candidate[i], candidate[j] = candidate[j], candidate[i]
                candidate_cost, candidate_timings, candidate_path = calculate_route_cost(graph, candidate, start_time, criterion)

                if candidate_cost >= best_solution_cost:
                    continue

                if candidate_cost < best_candidate_cost:
                    best_candidate = candidate
                    best_candidate_cost = candidate_cost
                    best_candidate_timings = candidate_timings
                    best_candidate_path = candidate_path

                move = (best_candidate[i], best_candidate[j])
                tabu_list.append(move)
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)

        if best_candidate:
            best_route = best_candidate
            best_cost = best_candidate_cost
            if best_cost < best_solution_cost:
                best_solution = best_route[:]
                best_solution_cost = best_cost
                best_timings = best_candidate_timings
                best_path = best_candidate_path

    return best_solution, best_solution_cost, best_timings, best_path


def calculate_route_cost(graph, route, start_time, criterion):
    # print("Location_index_map:", location_index_map)
    # print("Route:", route)

    total_cost = 0
    timings = []
    current_time = start_time
    timings.append(current_time)
    path = []

    if criterion == 't':
        for i in range(len(route) - 1):
            # travel_time = distance_matrix[route[i]][route[i+1]]
            travel_time, path_part = dijkstra(graph, route[i], route[i + 1], current_time, True)

            total_cost += travel_time
            current_time = path_part[-1][2]
            timings.append(current_time)
            path = path + path_part

    elif criterion == 'p':
        for i in range(len(route) - 1):
            transfer_count, path_part = dijkstra_min_transfers(graph, route[i], route[i + 1], current_time, True)

            total_cost += int(transfer_count)
            current_time = path_part[-1][2]
            timings.append(current_time)
            path = path + path_part

    else:
        raise NotImplementedError(f"Incorrect criterion. Please choose 't' or 'p'.")

    return total_cost, timings, path


if __name__ == "__main__":
    print("Please start inputting data:")

    start = sys.stdin.readline().strip()
    locations = sys.stdin.readline().strip().split(";")
    start_time = sys.stdin.readline().strip()
    criterion = sys.stdin.readline().strip()

    print("Waiting for program to finish running...")

    graph, stops = load_graph("Datasource/data.csv")

    start_time = parse_time(start_time)

    # distance_matrix = compute_distance_matrix(graph, stops, locations, start_time, criterion)

    # start_time_measure = time.time()
    # best_route, best_cost, best_timings =
    # local_search(graph, stops, [start] + locations + [start], start_time)
    # end_time_measure = time.time()
    #
    # for i in range(len(best_route)):
    #     print(f"{best_timings[i]} {best_route[i]}")
    # print("Total cost:", format_duration(best_cost))
    # print(f"\nExecution time: {end_time_measure - start_time_measure:.4f} s", file=sys.stderr)

    start_time_measure = time.time()
    best_route, best_cost, best_timings, best_path = tabu_search(graph, stops, [start] + locations + [start], start_time)
    end_time_measure = time.time()

    for i in range(len(best_route)):
        print(f"{best_timings[i]} {best_route[i]}")
    print("Total cost:", format_duration(best_cost) if criterion == 't' else best_cost)

    print("\nWhole route:")
    for j in range(len(best_path)):
        print(f"Line: {best_path[j][0]} Stop: {best_path[j][1]} Time: {best_path[j][2]}")

    print(f"\nExecution time: {end_time_measure - start_time_measure:.4f} s", file=sys.stderr)

# OSIEDLE SOBIESKIEGO
# KRZYKI;Kowale (Stacja kolejowa)
# 12:00:00
