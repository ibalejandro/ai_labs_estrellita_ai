import copy
import random




class AgenteAStar:

    SHOOT = 1
    OBSERVE = 2
    MOVE = 3

    RED = "rojo"
    ORANGE = "anaranjado"
    YELLOW = "amarillo"
    GREEN = "verde"

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
    sonar_prob_given_dist = {0: {GREEN: 0.4, YELLOW: 0.3, ORANGE: 0.2, RED: 0.1},
                             1: {GREEN: 0.3, YELLOW: 0.4, ORANGE: 0.2, RED: 0.1},
                             2: {GREEN: 0.2, YELLOW: 0.5, ORANGE: 0.2, RED: 0.1},
                             3: {GREEN: 0.15, YELLOW: 0.25, ORANGE: 0.4, RED: 0.2},
                             4: {GREEN: 0.1, YELLOW: 0.2, ORANGE: 0.25, RED: 0.45},
                             5: {GREEN: 0.1, YELLOW: 0.2, ORANGE: 0.25, RED: 0.45},
                             6: {GREEN: 0.1, YELLOW: 0.2, ORANGE: 0.25, RED: 0.45},
                             7: {GREEN: 0.1, YELLOW: 0.2, ORANGE: 0.25, RED: 0.45},
                             8: {GREEN: 0.1, YELLOW: 0.2, ORANGE: 0.25, RED: 0.45}}
    risk_level_for_colors = {GREEN: 4, YELLOW: 3, ORANGE: 2, RED: 1}

    # Constructor.
    def __init__(self):
        self.current_player = 0  # There is no current_player at that moment.
        self.turns_count = 0  # Variable to count the turns of the game
        self.prev_action = []
        self.star_position = ()

    def get_action_to_take(self, current_player, action_result, adversary_action, star_position):
        self.star_position = self.convert_index_to_tuple(star_position)
        if not adversary_action:
            print("Random observation")
            # No adversary action means first turn.
            action = self.OBSERVE
            # At the beginning, the adversary could be in any of the 25 positions of the grid with equal probability.
            action_param = random.randint(1, self.GRID_SIZE)
        else:
            adv_action = adversary_action[0]
            adv_action_param = adversary_action[1]
            adv_action_result = adversary_action[2]

            print("prev_action", self.prev_action)
            if self.prev_action[0] == self.OBSERVE:
                print("prev_action was OBSERVE")
                measurement_position = self.convert_index_to_tuple(self.prev_action[1])
                self.update_belief()
                self.update_belief_with_obs(measurement_position, action_result)

            if self.calculate_risk_level(adv_action, adv_action_param, adv_action_result) >= 4:
                print("Move")
                action = self.MOVE
                action_param = self.get_best_index_to_move(adv_action_param)
                self.star_position = action_param
            else:
                print("Observe or shoot")
                action = self.OBSERVE
                action_param = 8

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

    def calculate_risk_level(self, adv_action, adv_action_param, adv_action_result):
        if adv_action == self.SHOOT or adv_action == self.OBSERVE:
            adv_sight = self.convert_index_to_tuple(adv_action_param)
            distance_to_star_position = self.calc_distance(self.star_position, adv_sight)
            risk_level = self.get_risk_level_given_distance(adv_action, distance_to_star_position, adv_action_result)
        else:
            # Adversary moved.
            risk_level = 0
        print("Risk level:", risk_level)
        return risk_level

    def get_risk_level_given_distance(self, action, distance, adv_action_result):
        if action == self.SHOOT:
            if distance == 0:
                # Adversary shot us.
                risk_level = 5
            elif distance == 1:
                risk_level = 4
            elif 2 <= distance <= 4:
                risk_level = 3
            elif 5 <= distance <= 6:
                risk_level = 2
            else:
                risk_level = 1
        else:
            if distance == 0:
                risk_level = 4 + (self.risk_level_for_colors[adv_action_result] * self.sonar_prob_given_dist[distance][adv_action_result])
            elif 1 <= distance <= 2:
                risk_level = 3 + (self.risk_level_for_colors[adv_action_result] * self.sonar_prob_given_dist[distance][adv_action_result])
            elif 3 <= distance <= 4:
                risk_level = 2 + (self.risk_level_for_colors[adv_action_result] * self.sonar_prob_given_dist[distance][adv_action_result])
            elif 5 <= distance <= 6:
                risk_level = 1 + (self.risk_level_for_colors[adv_action_result] * self.sonar_prob_given_dist[distance][adv_action_result])
            else:
                risk_level = 0 + (self.risk_level_for_colors[adv_action_result] * self.sonar_prob_given_dist[distance][adv_action_result])
        return risk_level

    def get_best_index_to_move(self, adv_action_param):
        adv_sight = self.convert_index_to_tuple(adv_action_param)
        i = self.star_position[0]
        j = self.star_position[1]
        max_distance = 0
        if i - 1 >= 0: # Up.
            possible_position = (i - 1, j)
            distance_to_adv_sight = self.calc_distance(possible_position, adv_sight)
            distance_to_adv_sight += self.count_possible_movements(possible_position)
            if distance_to_adv_sight > max_distance:
                max_distance = distance_to_adv_sight
                best_position = possible_position
        if j + 1 <= 4: # Right.
            possible_position = (i, j + 1)
            distance_to_adv_sight = self.calc_distance(possible_position, adv_sight)
            distance_to_adv_sight += self.count_possible_movements(possible_position)
            if distance_to_adv_sight > max_distance:
                max_distance = distance_to_adv_sight
                best_position = possible_position
        if i + 1 <= 4: # Down.
            possible_position = (i + 1, j)
            distance_to_adv_sight = self.calc_distance(possible_position, adv_sight)
            distance_to_adv_sight += self.count_possible_movements(possible_position)
            if distance_to_adv_sight > max_distance:
                max_distance = distance_to_adv_sight
                best_position = possible_position
        if j - 1 >= 0: # Left.
            possible_position = (i, j - 1)
            distance_to_adv_sight = self.calc_distance(possible_position, adv_sight)
            distance_to_adv_sight += self.count_possible_movements(possible_position)
            if distance_to_adv_sight > max_distance:
                max_distance = distance_to_adv_sight
                best_position = possible_position
        return self.convert_tuple_to_index(best_position)

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

    def count_possible_movements(self, position):
        possible_movements = 0
        i = position[0]
        j = position[1]
        if i - 1 >= 0: # Up.
            possible_movements += 1
        if j + 1 <= 4: # Right.
            possible_movements += 1
        if i + 1 <= 4: # Down.
            possible_movements += 1
        if j - 1 >= 0: # Left.
            possible_movements += 1
        return possible_movements