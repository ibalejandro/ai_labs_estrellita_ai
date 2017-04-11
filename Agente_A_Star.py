import copy

class AgenteAStar:

    GRID_WIDTH_HEIGHT = 5

    init_prob = 1 / 25
    one_half = 1 / 2
    one_third = 1 / 3
    one_fourth = 1 / 4
    belief_prev_t = [[init_prob, init_prob, init_prob, init_prob, init_prob],
                     [init_prob, init_prob, init_prob, init_prob, init_prob],
                     [init_prob, init_prob, init_prob, init_prob, init_prob],
                     [init_prob, init_prob, init_prob, init_prob, init_prob],
                     [init_prob, init_prob, init_prob, init_prob, init_prob]]
    belief_after_t = [[init_prob, init_prob, init_prob, init_prob, init_prob],
                      [init_prob, init_prob, init_prob, init_prob, init_prob],
                      [init_prob, init_prob, init_prob, init_prob, init_prob],
                      [init_prob, init_prob, init_prob, init_prob, init_prob],
                      [init_prob, init_prob, init_prob, init_prob, init_prob]]
    belief_after_t_and_obs = [[init_prob, init_prob, init_prob, init_prob, init_prob],
                              [init_prob, init_prob, init_prob, init_prob, init_prob],
                              [init_prob, init_prob, init_prob, init_prob, init_prob],
                              [init_prob, init_prob, init_prob, init_prob, init_prob],
                              [init_prob, init_prob, init_prob, init_prob, init_prob]]
    transition_prob = {(0, 0): {(0, 1): one_half, (1, 0): one_half},
                       (0, 1): {(0, 0): one_third, (0, 2): one_third, (1, 1): one_third},
                       (0, 2): {(0, 1): one_third, (0, 3): one_third, (1, 2): one_third},
                       (0, 3): {(0, 2): one_third, (0, 4): one_third, (1, 3): one_third},
                       (0, 4): {(0, 3): one_half, (1, 4): one_half},
                       (1, 0): {(0, 0): one_third, (1, 1): one_third, (2, 0): one_third},
                       (1, 1): {(0, 1): one_fourth, (1, 0): one_fourth, (1, 2): one_fourth, (2, 1): one_fourth},
                       (1, 2): {(0, 2): one_fourth, (1, 1): one_fourth, (1, 3): one_fourth, (2, 2): one_fourth},
                       (1, 3): {(0, 3): one_fourth, (1, 2): one_fourth, (1, 4): one_fourth, (2, 3): one_fourth},
                       (1, 4): {(0, 4): one_third, (1, 3): one_third, (2, 4): one_third},
                       (2, 0): {(1, 0): one_third, (2, 1): one_third, (3, 0): one_third},
                       (2, 1): {(1, 1): one_fourth, (2, 0): one_fourth, (2, 2): one_fourth, (3, 1): one_fourth},
                       (2, 2): {(1, 2): one_fourth, (2, 1): one_fourth, (2, 3): one_fourth, (3, 2): one_fourth},
                       (2, 3): {(1, 3): one_fourth, (2, 2): one_fourth, (2, 4): one_fourth, (3, 3): one_fourth},
                       (2, 4): {(1, 4): one_third, (2, 3): one_third, (3, 4): one_third},
                       (3, 0): {(2, 0): one_third, (3, 1): one_third, (4, 0): one_third},
                       (3, 1): {(2, 1): one_fourth, (3, 0): one_fourth, (3, 2): one_fourth, (4, 1): one_fourth},
                       (3, 2): {(2, 2): one_fourth, (3, 1): one_fourth, (3, 3): one_fourth, (4, 2): one_fourth},
                       (3, 3): {(2, 3): one_fourth, (3, 2): one_fourth, (3, 4): one_fourth, (4, 3): one_fourth},
                       (3, 4): {(2, 4): one_third, (3, 3): one_third, (4, 4): one_third},
                       (4, 0): {(3, 0): one_half, (4, 1): one_half},
                       (4, 1): {(3, 1): one_third, (4, 0): one_third, (4, 2): one_third},
                       (4, 2): {(3, 2): one_third, (4, 1): one_third, (4, 3): one_third},
                       (4, 3): {(3, 3): one_third, (4, 2): one_third, (4, 4): one_third},
                       (4, 4): {(3, 4): one_half, (4, 3): one_half}}
    sonar_prob_given_dist = {0: {"green": 0.4, "yellow": 0.3, "orange": 0.2, "red": 0.1},
                             1: {"green": 0.3, "yellow": 0.4, "orange": 0.2, "red": 0.1},
                             2: {"green": 0.2, "yellow": 0.5, "orange": 0.2, "red": 0.1},
                             3: {"green": 0.15, "yellow": 0.25, "orange": 0.4, "red": 0.2},
                             4: {"green": 0.1, "yellow": 0.2, "orange": 0.25, "red": 0.45},
                             5: {"green": 0.4, "yellow": 0.3, "orange": 0.2, "red": 0.1},
                             6: {"green": 0.3, "yellow": 0.4, "orange": 0.2, "red": 0.1},
                             7: {"green": 0.2, "yellow": 0.5, "orange": 0.2, "red": 0.1},
                             8: {"green": 0.15, "yellow": 0.25, "orange": 0.4, "red": 0.2}}

    def __init__(self):
        None

    def update_belief(self):
        self.elapse_time()
        self.belief_prev_t = copy.deepcopy(self.belief_after_t)
        print("-----------")
        print("UPDATED BELIEF AFTER ELAPSING TIME:")
        print("-----------")
        self.print_formatted_grid(self.belief_after_t)

    def elapse_time(self):
        for i in range(0, self.GRID_WIDTH_HEIGHT):
            for j in range(0, self.GRID_WIDTH_HEIGHT):
                position = (i, j)
                b_prime = 0
                for key, value in self.transition_prob.items():
                    if value.__contains__(position):
                        b_prime += (value[position] * self.belief_prev_t[key[0]][key[1]])
                self.belief_after_t[position[0]][position[1]] = b_prime

    def print_formatted_grid(self, grid):
        for i in range(0, self.GRID_WIDTH_HEIGHT):
            row = "["
            for j in range(0, self.GRID_WIDTH_HEIGHT):
                if j < self.GRID_WIDTH_HEIGHT - 1:
                    row += str(grid[i][j]) + ", "
                else:
                    row += str(grid[i][j]) + "]"
            print(row)