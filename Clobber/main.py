import heapq
from collections import deque
import time

class Position:
    def __init__(self, n, m, board, move, score=None):
        self.n = n
        self.m = m
        self.board = board
        self.move = move
        self.winner = '_'
        self.score = score

    def getBoard(self):
        return self.board

    def printBoard(self):
        for row in self.board:
           print(*row, sep=' ')



class Node:
    def __init__(self, position):
        self.position = position
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def getPosition(self):
        return self.position


def generate_starting_position(n, m):
    board = []
    for i in range(m):
        row = []
        for j in range(n):
            if abs(j - i) % 2 == 0:
                row.append('B')
            else:
                row.append('W')
        board.append(row)

    start_pos = Position(n, m, board, 'W')
    return start_pos

seen = set()

def generate_next_possible_positions(position, k=10):
    global seen
    color = position.move
    board = position.getBoard()
    won = True
    result = []

    for i in range(position.m):
        for j in range(position.n):
            if board[i][j] != color:
                continue
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < position.m and 0 <= nj < position.n:
                    if board[ni][nj] not in ('_', color):
                        won = False
                        temp = [row[:] for row in board]
                        temp[ni][nj] = color
                        temp[i][j] = '_'

                        result.append(Node(Position(position.n, position.m, temp, 'W' if color == 'B' else 'B')))
                        if is_winning_position(result[-1].position):
                             result[-1].position.winner = color
                        result[-1].position.score = heuristic_score_simple(result[-1].position)
                        # key = board_to_tuple(temp)
                        # if key not in seen:
                        #     seen.add(key)
                        #     result.append(Node(Position(position.n, position.m, temp, 'W' if color == 'B' else 'B')))

    if len(result) > k:
        result = top_n_results(result, k, color)

    return result, won


def board_to_tuple(board):
    return tuple(tuple(row) for row in board)

def top_n_results(result, k, color):
    if color == 'W':
        return heapq.nlargest(k, result, key=lambda node: node.position.score)
    else:
        return heapq.nsmallest(k, result, key=lambda node: node.position.score)

def generate_decision_tree(root, depth):
    return generate_decision_tree_rec(root, 'W', depth, 0)

def generate_decision_tree_rec(curr_node, color, depth, curr_depth):

    children, won = generate_next_possible_positions(curr_node.position)

    if won:
        curr_node.position.winner = 'B' if color == 'W' else 'W'

    if depth <= curr_depth:
        return curr_node

    for elem in children:
        curr_node.add_child(elem)

    for elem in curr_node.children:
        generate_decision_tree_rec(elem, 'B' if color == 'W' else 'W', depth, curr_depth + 1)

    return curr_node


def is_winning_position(position):

    color = position.move
    board = position.getBoard()

    for i in range(position.m):
        for j in range(position.n):
            if board[i][j] != color:
                continue
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < position.m and 0 <= nj < position.n:
                    if board[ni][nj] not in ('_', color):
                        return False
    return True


def print_tree(tree):
    queue = deque()
    queue.append(tree)

    w = 0
    b = 0

    level = 0
    while queue:
        level_size = len(queue)
        print(f"Level {level}:")

        for _ in range(level_size):
            node = queue.popleft()
            node.position.printBoard()
            print("Winner: ", node.position.winner)
            print("To move: ", node.position.move)
            if node.position.score:
                print("Score: ", node.position.score)
            if node.position.winner == 'B':
                b += 1
            elif node.position.winner == 'W':
                w += 1
            print()
            queue.extend(node.children)

        level += 1

    print("W: ", w, " B: ", b)


def heuristic_score_dumb(position):
    if position.winner == 'W':
        return 1
    elif position.winner == 'B':
        return -1
    board = position.getBoard()
    # color = position.move
    # opponent = 'B' if color == 'W' else 'W'
    color = 'W'
    opponent = 'B'

    def count_movable_pieces(color):
        movable = 0
        for i in range(position.m):
            for j in range(position.n):
                if board[i][j] != color:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < position.m and 0 <= nj < position.n:
                        if board[ni][nj] == opponent:
                            movable += 1
                            break
        return movable

    return (count_movable_pieces(color) - count_movable_pieces(opponent)) / 100


def heuristic_score_simple(position):
    if position.winner == 'W':
        return 1
    elif position.winner == 'B':
        return -1
    board = position.getBoard()
    # color = position.move
    # opponent = 'B' if color == 'W' else 'W'
    color = 'W'
    opponent = 'B'

    def count_legal_moves(color):
        moves = 0
        for i in range(position.m):
            for j in range(position.n):
                if board[i][j] != color:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < position.m and 0 <= nj < position.n:
                        if board[ni][nj] == opponent:
                            moves += 1
        return moves

    def count_isolated(color):
        isolated = 0
        for i in range(position.m):
            for j in range(position.n):
                if board[i][j] != color:
                    continue
                neighbors = 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < position.m and 0 <= nj < position.n:
                        if board[ni][nj] != '_':
                            neighbors += 1
                if neighbors == 0:
                    isolated += 1
        return isolated

    mobility_score = count_legal_moves(color) - count_legal_moves(opponent)
    isolation_score = count_isolated(opponent) - count_isolated(color)

    return (3 * mobility_score + isolation_score) / 100


def heuristic_score_advanced(position):

    if position.winner == 'W':
        return 1
    elif position.winner == 'B':
        return -1

    board = position.getBoard()
    # color = position.move
    # opponent = 'B' if color == 'W' else 'W'
    color = 'W'
    opponent = 'B'

    # Mobility Score
    player_moves = count_legal_moves(position, color)
    opponent_moves = count_legal_moves(position, opponent)
    mobility_score = player_moves - opponent_moves

    # Component Analysis
    components = identify_components(position)
    component_score = evaluate_components(position, components, color)

    # Parity Considerations
    parity_score = evaluate_parity(position, color)

    # Distance and isolation analysis
    distance_score = evaluate_piece_distances(position, color)

    # Combine scores with appropriate weights
    final_score = (
                          3.0 * mobility_score +
                          2.0 * component_score +
                          1.5 * parity_score +
                          1.0 * distance_score
                  ) / 100.0

    return final_score


def count_legal_moves(position, color):

    board = position.getBoard()
    opponent = 'B' if color == 'W' else 'W'
    moves = 0

    for i in range(position.m):
        for j in range(position.n):
            if board[i][j] != color:
                continue
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < position.m and 0 <= nj < position.n:
                    if board[ni][nj] == opponent:
                        moves += 1

    return moves


def identify_components(position):

    board = position.getBoard()
    visited = [[False for _ in range(position.n)] for _ in range(position.m)]
    components = []

    for i in range(position.m):
        for j in range(position.n):
            if board[i][j] != '_' and not visited[i][j]:
                # Found a new component
                component = []
                queue = [(i, j)]
                visited[i][j] = True

                while queue:
                    curr_i, curr_j = queue.pop(0)
                    component.append((curr_i, curr_j, board[curr_i][curr_j]))

                    # Add adjacent pieces of any player to the component
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = curr_i + dx, curr_j + dy
                        if (0 <= ni < position.m and 0 <= nj < position.n and
                                board[ni][nj] != '_' and not visited[ni][nj]):
                            queue.append((ni, nj))
                            visited[ni][nj] = True

                components.append(component)

    return components


def evaluate_components(position, components, color):

    opponent = 'B' if color == 'W' else 'W'
    component_score = 0

    for component in components:

        player_pieces = sum(1 for _, _, piece in component if piece == color)
        opponent_pieces = sum(1 for _, _, piece in component if piece == opponent)


        if len(component) <= 6:
            nimber = calculate_component_value(position, component, color)
            component_score += nimber
        else:

            player_moves = count_component_moves(position, component, color)
            opponent_moves = count_component_moves(position, component, opponent)
            component_score += 0.3 * (player_moves - opponent_moves)

            piece_diff = player_pieces - opponent_pieces
            component_score += 0.1 * piece_diff

    return component_score


def calculate_component_value(position, component, color):

    opponent = 'B' if color == 'W' else 'W'
    size = len(component)

    positions = [(r, c) for r, c, _ in component]

    is_line_shape = check_if_line(positions)

    if size == 2:
        p1 = component[0][2]
        p2 = component[1][2]
        if p1 != p2:
            return 1.0 if color == p1 else -1.0
        return 0.0

    elif size == 3 and is_line_shape:

        pieces = [piece for _, _, piece in component]
        if pieces.count(color) > pieces.count(opponent):
            return 0.7
        else:
            return -0.7

    elif size == 4 and is_line_shape:
        pieces = [piece for _, _, piece in component]
        if pieces[0] != pieces[1] and pieces[1] != pieces[2] and pieces[2] != pieces[3]:
            return -0.5
        return 0.3

    pieces = [piece for _, _, piece in component]
    player_count = pieces.count(color)
    opponent_count = pieces.count(opponent)

    if size % 2 == 1:
        if player_count > opponent_count:
            return 0.5
        else:
            return -0.5
    else:
        if player_count > opponent_count:
            return 0.3
        else:
            return -0.3


def check_if_line(positions):

    if len(positions) <= 2:
        return True


    if all(r == positions[0][0] for r, _ in positions):

        cols = sorted([c for _, c in positions])
        return all(cols[i + 1] - cols[i] == 1 for i in range(len(cols) - 1))


    if all(c == positions[0][1] for _, c in positions):

        rows = sorted([r for r, _ in positions])
        return all(rows[i + 1] - rows[i] == 1 for i in range(len(rows) - 1))

    return False


def count_component_moves(position, component, color):

    opponent = 'B' if color == 'W' else 'W'
    moves = 0

    for i, j, piece in component:
        if piece != color:
            continue
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + dx, j + dy
            if 0 <= ni < position.m and 0 <= nj < position.n:
                if (ni, nj, opponent) in component:
                    moves += 1

    return moves


def evaluate_parity(position, color):

    board = position.getBoard()
    opponent = 'B' if color == 'W' else 'W'
    parity_score = 0

    for i in range(position.m):
        row_pieces = [board[i][j] for j in range(position.n) if board[i][j] != '_']
        if len(row_pieces) % 2 == 1:
            player_pieces = row_pieces.count(color)
            opponent_pieces = row_pieces.count(opponent)
            if player_pieces > opponent_pieces:
                parity_score += 0.5
            else:
                parity_score -= 0.5

    for j in range(position.n):
        col_pieces = [board[i][j] for i in range(position.m) if board[i][j] != '_']
        if len(col_pieces) % 2 == 1:
            player_pieces = col_pieces.count(color)
            opponent_pieces = col_pieces.count(opponent)
            if player_pieces > opponent_pieces:
                parity_score += 0.5
            else:
                parity_score -= 0.5

    return parity_score


def evaluate_piece_distances(position, color):

    board = position.getBoard()
    opponent = 'B' if color == 'W' else 'W'
    distance_score = 0

    player_positions = []
    opponent_positions = []

    for i in range(position.m):
        for j in range(position.n):
            if board[i][j] == color:
                player_positions.append((i, j))
            elif board[i][j] == opponent:
                opponent_positions.append((i, j))

    for i, j in player_positions:

        adjacent_opponents = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + dx, j + dy
            if 0 <= ni < position.m and 0 <= nj < position.n:
                if board[ni][nj] == opponent:
                    adjacent_opponents += 1

        if adjacent_opponents > 0:
            distance_score += 0.3 * adjacent_opponents
        else:
            min_distance = float('inf')
            for oi, oj in opponent_positions:
                dist = abs(i - oi) + abs(j - oj)
                min_distance = min(min_distance, dist)

            if min_distance > 2:
                distance_score -= 0.2
            elif min_distance == 2:
                distance_score += 0.1

    isolated_player = count_isolated_pieces(position, color)
    isolated_opponent = count_isolated_pieces(position, opponent)
    distance_score -= 0.5 * (isolated_player - isolated_opponent)

    return distance_score


def count_isolated_pieces(position, color):

    board = position.getBoard()
    isolated = 0

    for i in range(position.m):
        for j in range(position.n):
            if board[i][j] != color:
                continue

            has_neighbor = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < position.m and 0 <= nj < position.n:
                    if board[ni][nj] != '_':
                        has_neighbor = True
                        break

            if not has_neighbor:
                isolated += 1

    return isolated


def minimax(node, depth, alpha=float('-inf'), beta=float('inf'), strat='s'):
    # Base cases
    if node.position.winner == 'W':
        return 1  # White wins
    elif node.position.winner == 'B':
        return -1  # Black wins
    elif depth == 0 or not node.children:
        if strat == 's':
            return heuristic_score_simple(node.position)
        elif strat == 'a':
            return heuristic_score_advanced(node.position)
        elif strat == 'd':
            return heuristic_score_dumb(node.position)
        else:
            return heuristic_score_simple(node.position)

    is_maximizing = node.position.move == 'W'

    if is_maximizing:
        best_score = float('-inf')
        for child in node.children:
            score = minimax(child, depth - 1, alpha, beta, strat)
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # Beta cutoff
        return best_score
    else:
        best_score = float('inf')
        for child in node.children:
            score = minimax(child, depth - 1, alpha, beta, strat)
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # Alpha cutoff
        return best_score


def choose_best_move(root, depth, strat='s'):
    best_score = float('-inf') if root.position.move == 'W' else float('inf')
    best_child = None

    alpha = float('-inf')
    beta = float('inf')

    for child in root.children:
        score = minimax(child, depth - 1, alpha, beta, strat)

        if root.position.move == 'W':
            if score > best_score:
                best_score = score
                best_child = child

            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score = score
                best_child = child

            beta = min(beta, best_score)

        if beta <= alpha:
            break

    if len(root.children) == 0:
        print('No children for ', root.position.printBoard())

    return best_child, best_score


def play(node, depth, strat='a'):
    node.position.printBoard()
    print()

    if is_winning_position(node.position):
        return 'W' if node.position.move == 'B' else 'B'

    tree = generate_decision_tree(node, depth)

    best_move, _ = choose_best_move(tree, depth, strat)

    return play(best_move, depth, strat)


def minimax_with_tree_generation(position, depth, alpha=float('-inf'), beta=float('inf'),
                                 isRoot=False, strat='s', k=3):

    if position.winner == 'W':
        return None, 1 if isRoot else 1  # White wins
    elif position.winner == 'B':
        return None, -1 if isRoot else -1  # Black wins
    elif depth == 0:
        if strat == 's':
            score = heuristic_score_simple(position)
        elif strat == 'a':
            score = heuristic_score_advanced(position)
        elif strat == 'd':
            score = heuristic_score_dumb(position)
        else:
            score = heuristic_score_simple(position)
        return None, score if isRoot else score

    color = position.move
    is_maximizing = (color == 'W')

    next_positions, won = generate_next_possible_positions_optimized(position, k, strat)

    if won:
        position.winner = 'B' if color == 'W' else 'W'
        return None, 1 if color == 'W' else -1

    if not next_positions:
        return None, -1 if color == 'W' else 1

    best_move = None

    if is_maximizing:
        best_score = float('-inf')
        for next_pos in next_positions:
            _, score = minimax_with_tree_generation(next_pos.position, depth - 1, alpha, beta, False, strat, k)
            if score > best_score:
                best_score = score
                best_move = next_pos
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_move, best_score if isRoot else best_score
    else:
        best_score = float('inf')
        for next_pos in next_positions:
            _, score = minimax_with_tree_generation(next_pos.position, depth - 1, alpha, beta, False, strat, k)
            if score < best_score:
                best_score = score
                best_move = next_pos
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_move, best_score if isRoot else best_score


def generate_next_possible_positions_optimized(position, k=3, strat='s'):

    color = position.move
    board = position.getBoard()
    won = True
    result = []

    for i in range(position.m):
        for j in range(position.n):
            if board[i][j] != color:
                continue
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < position.m and 0 <= nj < position.n:
                    if board[ni][nj] not in ('_', color):
                        won = False
                        temp = [row[:] for row in board]
                        temp[ni][nj] = color
                        temp[i][j] = '_'

                        new_position = Position(position.n, position.m, temp, 'W' if color == 'B' else 'B')
                        if is_winning_position(new_position):
                            new_position.winner = color

                        if strat == 's':
                            new_position.score = heuristic_score_simple(new_position)
                        elif strat == 'a':
                            new_position.score = heuristic_score_advanced(new_position)
                        elif strat == 'd':
                            new_position.score = heuristic_score_dumb(new_position)
                        else:
                            new_position.score = heuristic_score_simple(new_position)
                        result.append(Node(new_position))

    if len(result) > k:
        result = top_n_results(result, k, color)

    return result, won


def play_optimized(position, depth, strat='a', k=7):

    position.printBoard()
    print()

    if is_winning_position(position):
        return 'W' if position.move == 'B' else 'B'

    best_move, _ = minimax_with_tree_generation(position, depth,
                                                float('-inf'), float('inf'),
                                                True, strat, k)

    if best_move is None:
        print("Brak możliwych ruchów!")
        return 'W' if position.move == 'B' else 'B'

    return play_optimized(best_move.position, depth, strat, k)


if __name__ == "__main__":
    n = 10 # width
    m = 10 # height
    depth = 5
    start_pos = generate_starting_position(n, m)

    # print("Starting position:\n")
    # start_pos.printBoard()

    # tree = generate_decision_tree(Node(start_pos), depth, 2)
    #
    # print_tree(tree)
    #
    # best_move, best_score = choose_best_move(tree, depth)
    # print(f"Best move for {start_pos.move}:")
    # if best_move:
    #     best_move.position.printBoard()

    start_time_1 = time.time()
    # print("Winner non-optimized: ", play(Node(start_pos), depth))
    end_time_1 = time.time()
    start_time_2 = time.time()
    print("Winner optimized: ", play_optimized(start_pos, depth))
    end_time_2 = time.time()
    print(f"Execution time (non-optimized): {end_time_1 - start_time_1:.2f}s")
    print(f"Execution time (optimized): {end_time_2 - start_time_2:.2f}s")


