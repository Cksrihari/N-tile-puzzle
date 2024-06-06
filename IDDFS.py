"""
Seed,Case Id,Start State,Solution Found,Number of Moves,Number of Nodes Opened,Computing Time
883,1,"[2, 2, [[5, 3, 2], [8, 4, 6], [1, 7, 0]]]",1,24,35185,0.48798704147338867
883,2,"[0, 2, [[5, 3, 0], [2, 6, 1], [7, 8, 4]]]",0,30,95765,1.4489610195159912
883,3,"[0, 0, [[0, 3, 5], [4, 8, 6], [2, 7, 1]]]",1,24,26367,0.497391939163208
883,4,"[0, 0, [[0, 2, 8], [1, 7, 6], [4, 3, 5]]]",1,28,57393,0.8942091464996338
883,5,"[0, 2, [[5, 4, 0], [1, 7, 6], [2, 3, 8]]]",0,30,95765,1.4436440467834473
883,6,"[2, 1, [[5, 2, 8], [3, 6, 7], [4, 0, 1]]]",0,30,62320,1.4263300895690918
883,7,"[1, 1, [[8, 7, 3], [4, 0, 1], [5, 6, 2]]]",0,30,112461,1.6082160472869873
883,8,"[2, 2, [[1, 2, 6], [8, 4, 7], [5, 3, 0]]]",1,26,4251,0.7134757041931152
883,9,"[2, 2, [[1, 5, 7], [8, 6, 4], [2, 3, 0]]]",0,30,48677,1.2648756504058838
883,10,"[0, 2, [[2, 7, 0], [4, 8, 5], [1, 6, 3]]]",1,24,8324,0.40061116218566895

"""
import time
import csv
import random


class Node:
    def __init__(self, state, parent=None, action=None, depth=0):
        # Initialize a node with a state, its parent node, action taken, and depth in the search tree
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth

    def __eq__(self, other):
        # Checking if two nodes have identical states
        return self.state == other.state

    def __hash__(self):
        # Hash function for the node (used for hashing in sets and dictionaries)
        return hash(self.state)


# The seed for randomization, derived from the last three digits of a registration number
seed = 883
# Maximum depth for the iterative deepening depth-first search (IDDFS) algorithm
max_depth = 30


class PuzzleSolver:
    def __init__(self, n=3):
        # Initializing the PuzzleSolver with the dimension of the puzzle grid (default is 3x3)
        self.n = n
        random.seed(seed)

    def generate_random_states(self, num_states=10):
        # Generating random initial states for the puzzle
        start_state = list(range(8, -1, -1))
        start_states = []
        for i in range(num_states):
            shuffled_state = start_state[:]
            random.shuffle(shuffled_state)
            grid = [shuffled_state[i:i + 3] for i in range(0, 9, 3)]
            blank_pos = next((r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 0)
            start_states.append([*blank_pos, grid])
        return start_states

    def state_to_matrix(self, state):
        # Converting a state representation to a matrix format
        matrix = tuple(state[i:i + self.n] for i in range(0, self.n ** 2, self.n))
        zero_position = next((i, row.index(0)) for i, row in enumerate(matrix) if 0 in row)
        return zero_position + (matrix,)

    def generate_start_states(self, template_state, count=10):
        # Generating initial states based on a template state
        start_states = []
        for _ in range(count):
            shuffled_list = random.sample([item for sublist in template_state[2] for item in sublist], self.n ** 2)
            shuffled_matrix = tuple(tuple(shuffled_list[i:i + self.n]) for i in range(0, self.n ** 2, self.n))
            blank_position = shuffled_list.index(0)
            blank_x, blank_y = divmod(blank_position, self.n)
            start_states.append((blank_x, blank_y, shuffled_matrix))
        return start_states

    @staticmethod
    def get_children(node):
        # Retrieving child nodes of a given node
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        children = []
        x, y, matrix = node.state
        n = len(matrix)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                swapped_matrix = [list(row) for row in matrix]
                swapped_matrix[x][y], swapped_matrix[nx][ny] = swapped_matrix[nx][ny], swapped_matrix[x][y]
                children.append(Node((nx, ny, tuple(map(tuple, swapped_matrix))), node, (dx, dy), node.depth + 1))
        return children

    def iddfs(self, root, goal):
        # Performing Iterative Deepening Depth-First Search (IDDFS)
        for depth in range(max_depth + 1):
            visited = set()
            result, nodes_opened = self.dls(root, goal, depth, visited, 0)
            if result is not None:
                return result, nodes_opened
        return None, nodes_opened

    def depth_limited_search(self, node, goal, depth, visited, nodes_opened):
        # Performing Depth-Limited Search (DLS)
        if node.depth > depth:
            return None, nodes_opened

        visited.add((node.state[0], node.state[1], tuple(map(tuple, node.state[2]))))

        if node.state == goal:
            return node, nodes_opened
        for child in PuzzleSolver.get_children(node):
            if child.state not in visited:
                nodes_opened += 1
                result, nodes_opened = self.dls(child, goal, depth, visited, nodes_opened)
                if result is not None:
                    return result, nodes_opened
        return None, nodes_opened

    def solve_puzzle(self, start_state, goal_state, max_depth=30):
        # Solving the puzzle using Iterative Deepening Depth-First Search (IDDFS)
        root = Node(start_state)
        start_time = time.time()
        solution, nodes = self.iddfs(root, goal_state)
        end_time = time.time()
        moves = solution.depth if solution else max_depth
        solution = 1 if solution else 0
        computing_time = end_time - start_time
        return start_state, solution, moves, nodes, computing_time

    @staticmethod
    def save_results_to_csv(results, filename="IDDFS_output.csv"):
        # Saving results to a CSV file
        headers = ["Seed", "Case Id", "Start State", "Solution Found", "Number of Moves", "Number of Nodes Opened",
                   "Computing Time"]
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for case_num, data in enumerate(results, 1):
                writer.writerow([seed] + [case_num] + list(data))


if __name__ == '__main__':
    solver = PuzzleSolver()
    goal_state = (1, 1, ((1, 2, 3), (8, 0, 4), (7, 6, 5)))
    start_states = solver.generate_random_states()
    results = [solver.solve_puzzle(start_state, goal_state) for start_state in start_states]
    solver.save_results_to_csv(results)
    print("Results saved in IDDFS_output.csv")
