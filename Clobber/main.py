from collections import deque

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

def generate_next_possible_positions(position):
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
                        key = board_to_tuple(temp)
                        if key not in seen:
                            seen.add(key)
                            result.append(Node(Position(position.n, position.m, temp, 'W' if color == 'B' else 'B')))

    return result, won


def board_to_tuple(board):
    return tuple(tuple(row) for row in board)


def generate_decision_tree(root, depth):
    tree = root
    return generate_decision_tree_rec(tree, 'W', depth, 0)

def generate_decision_tree_rec(curr_node, color, depth, curr_depth):

    children, won = generate_next_possible_positions(curr_node.position)

    if len(children) > 0:
        for elem in children:
            curr_node.add_child(elem)
    if won:
        curr_node.position.winner = 'B' if color == 'W' else 'W'

    if depth > curr_depth and len(curr_node.children) > 0:
        for elem in curr_node.children:
            generate_decision_tree_rec(elem, 'B' if color == 'W' else 'W', depth, curr_depth + 1)
    else:
        curr_node.position.score = heuristic_score(curr_node.position)

    return curr_node


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


def heuristic_score(position):
    if position.winner == 'W':
        return 1
    elif position.winner == 'B':
        return -1
    board = position.getBoard()
    color = position.move
    opponent = 'B' if color == 'W' else 'W'

    def count_legal_moves(color):
        moves = 0
        for i in range(position.m):
            for j in range(position.n):
                if board[i][j] != color:
                    continue
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
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
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
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


def minimax(node, is_maximizing):

    if node.position.winner == 'W':
        return 1  # White wins
    elif node.position.winner == 'B':
        return -1  # Black wins
    elif not node.children:
        return heuristic_score(node.position)

    if is_maximizing:
        best_score = float('-inf')
        for child in node.children:
            score = minimax(child, False)
            best_score = max(best_score, score)
        return best_score
    else:
        best_score = float('inf')
        for child in node.children:
            score = minimax(child, True)
            best_score = min(best_score, score)
        return best_score



if __name__ == "__main__":
    n = 4 # width
    m = 4 # height
    start_pos = generate_starting_position(n, m)

    # print("Starting position:\n")
    # start_pos.printBoard()

    tree = generate_decision_tree(Node(start_pos), 10)

    print_tree(tree)