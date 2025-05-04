from collections import deque

class Position:
    def __init__(self, n, m, board):
        self.n = n
        self.m = m
        self.board = board
        self.winner = '_'

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

    start_pos = Position(n, m, board)
    return start_pos

seen = set()

def generate_next_possible_positions(position, color):
    global seen
    won = True
    result = []
    board = position.getBoard()
    for i in range(position.m):
        for j in range(position.n):
            if board[i][j] != color:
                continue
            if i > 0:
                if board[i - 1][j] != color and board[i - 1][j] != '_':
                    won = False
                    temp = [row[:] for row in board]
                    temp[i - 1][j] = color
                    temp[i][j] = '_'
                    key = board_to_tuple(temp)
                    if key not in seen:
                        seen.add(key)
                        result.append(Node(Position(position.n, position.m, temp)))
            if i < position.n - 1:
                if board[i + 1][j] != color and board[i + 1][j] != '_':
                    won = False
                    temp = [row[:] for row in board]
                    temp[i + 1][j] = color
                    temp[i][j] = '_'
                    key = board_to_tuple(temp)
                    if key not in seen:
                        seen.add(key)
                        result.append(Node(Position(position.n, position.m, temp)))
            if j > 0:
                if board[i][j - 1] != color and board[i][j - 1] != '_':
                    won = False
                    temp = [row[:] for row in board]
                    temp[i][j - 1] = color
                    temp[i][j] = '_'
                    key = board_to_tuple(temp)
                    if key not in seen:
                        seen.add(key)
                        result.append(Node(Position(position.n, position.m, temp)))
            if j < position.n - 1:
                if board[i][j + 1] != color and board[i][j + 1] != '_':
                    won = False
                    temp = [row[:] for row in board]
                    temp[i][j + 1] = color
                    temp[i][j] = '_'
                    key = board_to_tuple(temp)
                    if key not in seen:
                        seen.add(key)
                        result.append(Node(Position(position.n, position.m, temp)))

    return result, won


def board_to_tuple(board):
    return tuple(tuple(row) for row in board)


def generate_decision_tree(root, depth):
    tree = root
    return generate_decision_tree_rec(tree, 'W', depth, 1)

def generate_decision_tree_rec(curr_node, color, depth, curr_depth):

    children, won = generate_next_possible_positions(curr_node.position, color)

    if len(children) > 0:
        for elem in children:
            curr_node.add_child(elem)
    if won:
        curr_node.position.winner = 'B' if color == 'W' else 'W'

    if depth > curr_depth and len(curr_node.children) > 0:
        for elem in curr_node.children:
            generate_decision_tree_rec(elem, 'B' if color == 'W' else 'W', depth, curr_depth + 1)


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
            print(node.position.winner)
            if node.position.winner == 'B':
                b += 1
            elif node.position.winner == 'W':
                w += 1
            print()
            queue.extend(node.children)

        level += 1

    print("W: ", w, " B: ", b)


if __name__ == "__main__":
    n = 4 # width
    m = 4 # height
    start_pos = generate_starting_position(n, m)

    # print("Starting position:\n")
    # start_pos.printBoard()

    tree = generate_decision_tree(Node(start_pos), 20)

    print_tree(tree)