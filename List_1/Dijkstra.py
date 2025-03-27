import csv
import heapq
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
import math


def parse_time(time_str):
    try:
        hours, minutes, seconds = map(int, time_str.split(":"))
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}")


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


def find_shortest_path(start, end, criterion, start_time, graph, stops):
    start_time = parse_time(start_time)
    start_time_measure = time.time()
    travel_time = -1
    transfer_count = -1
    if criterion == 't':
        travel_time, path = dijkstra(graph, start, end, start_time, True)

    elif criterion == 'p':
        transfer_count, path = dijkstra_min_transfers(graph, start, end, start_time, True)

    else:
        raise NotImplementedError("Incorrect criterion. Please choose 't' or 'p'.")
    end_time_measure = time.time()

    if path:

        for i in range(0, len(path)):
            print(path[i])

        start_line, start_stop, s_time = "", "", 0
        print(f"{start_time} {start} -> {end}:")
        for i in range(1, len(path)):
            prev_line, prev_stop, prev_time = path[i - 1]
            line, stop, time_point = path[i]

            if line != prev_line:
                if prev_line != "START":
                    print(f"Line {start_line}: {start_stop} ({s_time}) -> "
                          f"{prev_stop} ({prev_time})")
                start_line = line
                start_stop = prev_stop
                s_time = prev_time

        prev_line, prev_stop, prev_time = path[-1]
        print(f"Line {start_line}: {start_stop} ({s_time}) -> {prev_stop} ({prev_time})")

        if travel_time > -1:
            print(f"\nTime en route: {format_duration(travel_time)}\n\n")
        else:
            print(f"Transfers: {(int) (transfer_count)}")

    else:
        print("Route not found")

    print(f"\nExecution time: {end_time_measure - start_time_measure:.4f} s", file=sys.stderr)

if __name__ == "__main__":
    graph, stops = load_graph("Datasource/data.csv")
    # find_shortest_path( "Pola", "Broniewskiego", "t", "02:12:00", graph, stops)
    # find_shortest_path("KRZYKI", "OSIEDLE SOBIESKIEGO", "p", "05:12:00", graph, stops)
    # find_shortest_path("Kowale (Stacja kolejowa)", "Daszyńskiego", "p", "13:48:00", graph, stops)
    find_shortest_path("Zajezdnia Obornicka", "PORT LOTNICZY", "p", "20:50:00", graph, stops)
