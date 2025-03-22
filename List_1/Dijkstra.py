import csv
import heapq
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta


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


def dijkstra(graph, start, end, start_time, add_astar_coeff):
    queue = [(0, start, start_time, None, [])]
    visited = {}
    end_min_cost = float('inf')
    end_path = ()

    while queue:
        cost, node, current_time, current_line, path = heapq.heappop(queue)

        if node in visited and visited[node] <= current_time:
            continue

        if cost > end_min_cost:
            continue

        path = path + [(current_line if current_line else "START", node, current_time)]

        if node == end:
            end_min_cost = cost
            end_path = path
            continue

        visited[node] = current_time

        for neighbor, line, dep_time, arr_time, travel_time in graph.get(node, []):
            if dep_time >= current_time:
                wait_time = (dep_time - current_time).seconds
                total_travel_time = cost + travel_time + wait_time

                heapq.heappush(queue, (total_travel_time, neighbor, arr_time, line, path))

    return end_min_cost, end_path


def find_shortest_path(start, end, criterion, start_time, graph, stops):
    start_time = parse_time(start_time)
    start_time_measure = time.time()
    if criterion == 't':
        cost, path = dijkstra(graph, start, end, start_time, False)

        for i in range(0, len(path)):
            print(path[i])

    else:
        raise NotImplementedError("A* for transfers is not yet implemented")
    end_time_measure = time.time()

    if path:

        start_line, start_stop, s_time = "", "", 0
        print(f"{start_time} {start} -> {end}:")
        for i in range(1, len(path)):
            prev_line, prev_stop, prev_time = path[i - 1]
            line, stop, time_point = path[i]

            if line != prev_line:
                if prev_line != "START":
                    print(f"Line {start_line}: {start_stop} ({s_time}) -> {prev_stop} ({prev_time})")
                start_line = line
                start_stop = prev_stop
                s_time = prev_time

        prev_line, prev_stop, prev_time = path[-1]
        print(f"Line {start_line}: {start_stop} ({s_time}) -> {prev_stop} ({prev_time})")

        print(f"\nTime en route: {format_duration(cost)}\n\n")

    else:
        print("Route not found")

    print(f"\nExecution time: {end_time_measure - start_time_measure:.4f} s", file=sys.stderr)

if __name__ == "__main__":
    graph, stops = load_graph("Datasource/data.csv")
    find_shortest_path( "Pola", "Broniweskiego",
                       "t", "19:10:00", graph, stops)
