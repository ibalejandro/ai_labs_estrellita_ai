import copy
import random

class AgenteAStar:

    SHOOT = 1
    OBSERVE = 2
    MOVE = 3

    GRID_WIDTH_HEIGHT = 5
    GRID_SIZE = 25

    init_prob = 1 / GRID_SIZE
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
    sonar_prob_given_dist = {0: {"verde": 0.7, "amarillo": 0.15, "anaranjado": 0.1, "rojo": 0.05},
                             1: {"verde": 0.65, "amarillo": 0.2, "anaranjado": 0.1, "rojo": 0.05},
                             2: {"verde": 0.6, "amarillo": 0.2, "anaranjado": 0.15, "rojo": 0.05},
                             3: {"verde": 0.55, "amarillo": 0.25, "anaranjado": 0.15, "rojo": 0.05},
                             4: {"verde": 0.5, "amarillo": 0.25, "anaranjado": 0.2, "rojo": 0.05},
                             5: {"verde": 0.5, "amarillo": 0.2, "anaranjado": 0.25, "rojo": 0.05},
                             6: {"verde": 0.1, "amarillo": 0.15, "anaranjado": 0.5, "rojo": 0.25},
                             7: {"verde": 0.15, "amarillo": 0.15, "anaranjado": 0.2, "rojo": 0.5},
                             8: {"verde": 0.05, "amarillo": 0.15, "anaranjado": 0.2, "rojo": 0.6}}

    # Constructor.
    def __init__(self):
        self.current_player = 0  # There is no current_player at that moment.
        self.turns_count = 0  # Variable to count the turns of the game
        self.measurement_position = ()
        self.prev_action = []

    def get_action_to_take(self, current_player, action_result, adversary_action, star_position):


        if not adversary_action:
            # No adversary action means first turn.
            action = self.OBSERVE
            # At the beginning, the adversary could be in any of the 25 positions of the grid with equal probability.
            action_param = random.randint(1, self.GRID_SIZE)
        else:
            adv_action = adversary_action[0]
            adv_action_param = adversary_action[1]
            adv_action_result = adversary_action[2]

        # if action == self.SHOOT:
        #
        # elif action == self.OBSERVE:
        #     self.update_belief_with_obs(measurement_position, "rojo")
        # elif action == self.MOVE:
        #
        #
        action_to_take = [action, action_param]
        self.prev_action = action_to_take
        return action_to_take


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

    def set_measurement_position(self, measurement_position):
        self.measurement_position = measurement_position

    def update_belief_with_obs(self, measurement_position, measurement_color):
        self.incorporate_observation(measurement_position, measurement_color)
        self.belief_prev_t = copy.deepcopy(self.belief_after_t_and_obs)
        print("-----------")
        print("UPDATED BELIEF AFTER OBSERVATION:")
        print("-----------")
        self.print_formatted_grid(self.belief_after_t_and_obs)

    def incorporate_observation(self, measurement_position, measurement_color):
        normalization_factor = 0
        for i in range(0, self.GRID_WIDTH_HEIGHT):
            for j in range(0, self.GRID_WIDTH_HEIGHT):
                position = (i, j)
                dist_to_measurement = self.calc_distance(measurement_position, position)
                prob_color_given_dist_to_measurement = \
                    self.sonar_prob_given_dist[dist_to_measurement][measurement_color]
                b_prime = self.belief_after_t[position[0]][position[1]]
                self.belief_after_t_and_obs[position[0]][position[1]] = prob_color_given_dist_to_measurement * b_prime
                normalization_factor += self.belief_after_t_and_obs[position[0]][position[1]]
        # Normalization.
        for i in range(0, self.GRID_WIDTH_HEIGHT):
            for j in range(0, self.GRID_WIDTH_HEIGHT):
                self.belief_after_t_and_obs[i][j] /= normalization_factor

# Utils

    def calc_distance(self, pos_1, pos_2):
        return abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])

    def print_formatted_grid(self, grid):
        for i in range(0, self.GRID_WIDTH_HEIGHT):
            row = "["
            for j in range(0, self.GRID_WIDTH_HEIGHT):
                if j < self.GRID_WIDTH_HEIGHT - 1:
                    row += str(grid[i][j]) + ", "
                else:
                    row += str(grid[i][j]) + "]"
            print(row)

    def convert_tuple_to_index(self, tuple):
        return (tuple[0] * self.GRID_WIDTH_HEIGHT) + tuple[1] + 1

    def convert_index_to_tuple(self, index):
        i = int((index - 1) / self.GRID_WIDTH_HEIGHT)
        j = (index - 1) - (i * self.GRID_WIDTH_HEIGHT)
        return (i, j)