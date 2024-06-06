"""
Case Number,Start State,Solution Found,Number of Moves,Nodes Opened,Computing Time
1,"[2, 2, [[5, 3, 2], [8, 4, 6], [1, 7, 0]]]",1,24,5112,0.02782583236694336
2,"[0, 2, [[5, 3, 0], [2, 6, 1], [7, 8, 4]]]",1,24,6793,0.03594613075256348
3,"[0, 0, [[0, 3, 5], [4, 8, 6], [2, 7, 1]]]",1,22,534,0.002753734588623047
4,"[0, 0, [[0, 2, 8], [1, 7, 6], [4, 3, 5]]]",1,22,1883,0.009737253189086914
5,"[0, 2, [[5, 4, 0], [1, 7, 6], [2, 3, 8]]]",0,0,1724557,10.000075578689575
6,"[2, 1, [[5, 2, 8], [3, 6, 7], [4, 0, 1]]]",0,0,1681900,10.000102281570435
7,"[1, 1, [[8, 7, 3], [4, 0, 1], [5, 6, 2]]]",1,24,2991,0.015620946884155273
8,"[2, 2, [[1, 2, 6], [8, 4, 7], [5, 3, 0]]]",1,20,1243,0.006437063217163086
9,"[2, 2, [[1, 5, 7], [8, 6, 4], [2, 3, 0]]]",0,0,1720778,10.000067949295044
10,"[0, 2, [[2, 7, 0], [4, 8, 5], [1, 6, 3]]]",1,22,926,0.004828929901123047

"""

import random
import time
import csv
from datetime import datetime, timedelta


class NTilePuzzle:
    def __init__(self, initial, goal):
        # Set up the puzzle with initial and goal states
        self.start_state = initial[2]
        self.goal_state = goal[2]
        self.size = len(self.start_state)
        # Creating a dictionary to store the target positions of each tile in the goal state
        self.target_positions = {val: (r, c) for r, row in enumerate(self.goal_state) for c, val in enumerate(row)}

    def tile_distance(self, state):
        # Calculating the Manhattan distance of each tile from its goal position
        distance = sum(abs(r - self.target_positions[val][0]) + abs(c - self.target_positions[val][1])
                       for r, row in enumerate(state) for c, val in enumerate(row) if val)
        return distance

    def create_successor_states(self, state):
        # Generating possible successor states by moving the blank tile
        row_index, col_index = next((r, c) for r, row in enumerate(state) for c, val in enumerate(row) if not val)
        successors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_r, new_c = row_index + dr, col_index + dc
            if 0 <= new_r < self.size and 0 <= new_c < self.size:
                new_state = [list(row) for row in state]
                new_state[row_index][col_index], new_state[new_r][new_c] = new_state[new_r][new_c], new_state[row_index][col_index]
                successors.append(new_state)
        return successors

    def ida_star(self, max_depth=30, timeout_seconds=10):
        nodes_opened = [0]

        def search(path, g, bound, start_time, nodes_opened):
            # Recursive function for IDA* search
            nodes_opened[0] += 1
            current = path[-1]
            f = g + self.tile_distance(current)
            if f > bound or g > max_depth or datetime.now() - start_time > timedelta(seconds=timeout_seconds):
                return f, False, nodes_opened[0]
            if current == self.goal_state:
                return g, True, nodes_opened[0]
            min_bound = float('inf')
            for s in self.create_successor_states(current):
                if s not in path:
                    path.append(s)
                    t, found, nodes_opened_count = search(path, g+1, bound, start_time, nodes_opened)
                    if found:
                        return t, True, nodes_opened_count
                    if t < min_bound:
                        min_bound = t
                    path.pop()
            return min_bound, False, nodes_opened[0]

        start_time = datetime.now()
        bound = self.tile_distance(self.start_state)
        path = [self.start_state]
        while True:
            t, found, nodes_opened_count = search(path, 0, bound, start_time, nodes_opened)
            if found or datetime.now() - start_time > timedelta(seconds=timeout_seconds):
                return path if found else None, nodes_opened_count
            bound = t


def solve_puzzle(start_state, goal_state):
    # Solving the puzzle using the IDA* algorithm
    start_time = time.time()
    puzzle = NTilePuzzle(start_state, goal_state)
    solution_path, nodes_opened = puzzle.ida_star()
    end_time = time.time()

    solution = solution_path is not None
    moves = len(solution_path) - 1 if solution else 0
    time_taken = end_time - start_time
    return start_state, solution, moves, nodes_opened, time_taken


def generate_start_states(seed, num_states=10):
    # Generating random start states for the puzzle
    template_state = list(range(8, -1, -1))
    random.seed(seed)
    start_states = []
    for _ in range(num_states):
        shuffled_state = template_state[:]
        random.shuffle(shuffled_state)
        grid = [shuffled_state[i:i+3] for i in range(0, 9, 3)]
        blank_pos = next((r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 0)
        start_states.append([*blank_pos, grid])
    return start_states


def main_idastar():
    seed = 883
    goal_state = [1, 1, [[1, 2, 3], [8, 0, 4], [7, 6, 5]]]
    start_states = generate_start_states(seed)

    with open('IDAstar_output.csv', 'w', newline='') as csvfile:
        fieldnames = ['Case Number', 'Start State', 'Solution Found', 'Number of Moves', 'Nodes Opened', 'Computing Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, start_state in enumerate(start_states, 1):
            result = solve_puzzle(start_state, goal_state)
            writer.writerow({
                'Case Number': i,
                'Start State': str(result[0]),
                'Solution Found': 1 if result[1] else 0,
                'Number of Moves': result[2],
                'Nodes Opened': result[3],
                'Computing Time': result[4]
            })
    print("Results saved in IDAStar_output.csv")


if __name__ == '__main__':
    main_idastar()
