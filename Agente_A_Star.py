import copy
import random
import numpy as np


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

    RISK_LEVEL_LIMIT = 4
    MIN_PROB_TO_SHOOT = 0.13

    init_prob = 1 / GRID_SIZE
    one_half = 1 / 2
    one_third = 1 / 3
    one_fourth = 1 / 4

    FA = "Forward Algorithm"
    PF = "Particle Filtering"

    PARTICLES_QUANTITY = 50

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
    start_particles = [[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]
    particles_after_t = [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]
    empty_grid = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
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
    sonar_prob_given_dist = {0: {GREEN: 0.70, YELLOW: 0.15, ORANGE: 0.1, RED: 0.05},
                             1: {GREEN: 0.17, YELLOW: 0.6, ORANGE: 0.17, RED: 0.06},
                             2: {GREEN: 0.06, YELLOW: 0.17, ORANGE: 0.6, RED: 0.17},
                             3: {GREEN: 0.05, YELLOW: 0.12, ORANGE: 0.23, RED: 0.6},
                             4: {GREEN: 0.05, YELLOW: 0.1, ORANGE: 0.15, RED: 0.8},
                             5: {GREEN: 0.05, YELLOW: 0.1, ORANGE: 0.15, RED: 0.8},
                             6: {GREEN: 0.05, YELLOW: 0.1, ORANGE: 0.15, RED: 0.8},
                             7: {GREEN: 0.05, YELLOW: 0.1, ORANGE: 0.15, RED: 0.8},
                             8: {GREEN: 0.05, YELLOW: 0.1, ORANGE: 0.15, RED: 0.8}}
    # Every color returned to the adversary's measurement has a different associated risk for the agent.
    risk_level_for_colors = {GREEN: 4, YELLOW: 3, ORANGE: 2, RED: 1}

    # Constructor.
    def __init__(self):
        self.current_player = 0  # There is no current_player at that moment.
        self.turns_count = 0  # Variable to count the turns of the game.
        self.prev_action = []
        self.star_position = ()  # The start position of the star.
        self.belief_update_algorithm = self.FA  # The algorithm that is being used to update beliefs.

    def get_action_to_take(self, current_player, action_result, adversary_action, star_position):
        print("Before executing:")
        print("Action result:", action_result, "Adversary action:", adversary_action)
        self.current_player = current_player
        self.star_position = self.convert_index_to_tuple(star_position)

        if adversary_action[0] is None:
            # It only occurs in the first turn of the game (no adversary action). The most intelligent action is to
            # observe anywhere.
            print("First turn = Random observation.")
            action = self.OBSERVE
            # At the beginning, the adversary could be in any of the 25 positions of the grid with equal probability.
            action_param = random.randint(1, self.GRID_SIZE)
        else:
            if not self.prev_action:
                # There was no previous action of the agent. It´s its first turn as the second player.
                print("First turn for second player = Random observation.")
                action = self.OBSERVE
                action_param = random.randint(1, self.GRID_SIZE)
            else:
                print("Agent previous action =", self.prev_action, "Result for that action =", action_result)
                # The adversary action is decomposed into three parameters.
                adv_action = adversary_action[0]
                adv_action_param = adversary_action[1]
                adv_action_result = adversary_action[2]

                if self.prev_action[0] == self.SHOOT:
                    print("Previous action was SHOOT.")
                    if action_result:  # If action_result = 1, the adversary was hit and the condition is True.
                        # The agent hit the adversary.
                        # The belief update algorithm changes when the adversary is hit for the first time.
                        print("Belief update algorithm changed to Particle Filtering.")
                        self.belief_update_algorithm = self.PF
                        # The whole particles are agglomerated in the position the agent hit the adversary.
                        tuple_hit = self.convert_index_to_tuple(self.prev_action[1])
                        self.start_particles = copy.deepcopy(self.empty_grid)
                        self.start_particles[tuple_hit[0]][tuple_hit[1]] = self.PARTICLES_QUANTITY
                        self.print_formatted_grid(self.start_particles)
                        if adv_action != self.MOVE:
                            # If the agent hit the adversary in the last turn and he didn't move away, then the agent
                            # must shoot on the same position and hit again.
                            action_to_take = self.prev_action
                            self.turns_count += 2
                            # The action to take is the same as the previous one. No more computation is necessary.
                            print("Action to take:", action_to_take)
                            return action_to_take
                        else:
                            # The adversary was hit and he moved away. An update using Particle Filtering is simulated
                            # with a sensor YELLOW observation on the previous position of the adversary (which is good
                            # for the neighbors of that position due to distance = 1).
                            self.execute_particle_filtering(self.prev_action[1], self.YELLOW)
                    else:
                        # If the agent didn't hit the adversary in the last turn, then the probability for that
                        # particular position and the near region get smoothed whereas the probability for the far
                        # region increases.
                        if self.belief_update_algorithm == self.FA:
                            self.execute_forward_algorithm(self.prev_action[1], self.ORANGE)
                        else:
                            self.redistribute_particles(self.prev_action[1])
                            self.execute_particle_filtering(self.prev_action[1], self.ORANGE)
                elif self.prev_action[0] == self.OBSERVE:
                    # The belief is updated given the measurement position and the observed color.
                    print("Previous action was OBSERVE.")
                    if self.belief_update_algorithm == self.FA:
                        self.execute_forward_algorithm(self.prev_action[1], action_result)
                    else:
                        self.execute_particle_filtering(self.prev_action[1], action_result)

                if self.calculate_risk_level(adv_action, adv_action_param, adv_action_result) >= self.RISK_LEVEL_LIMIT:
                    print("The agent is at risk and it has to MOVE.")
                    action = self.MOVE
                    action_param = self.get_best_index_to_move(adv_action_param)
                    self.star_position = action_param  # The star position is updated with that movement.
                else:
                    # The agent is not at risk.
                    max_probability, max_elem_index = self.get_max_value_in_table_with_index(self.belief_prev_t)
                    if max_probability > self.MIN_PROB_TO_SHOOT:
                        print("The agent has to SHOOT.")
                        action = self.SHOOT
                        action_param = max_elem_index
                    else:
                        # The agent is not sure enough to SHOOT. The probabilities are not very significant.
                        print("The agent has to OBSERVE.")
                        action = self.OBSERVE
                        action_param = max_elem_index

        action_to_take = [action, action_param]
        self.prev_action = action_to_take  # The action that is going to be taken is stored for the next turn.
        self.turns_count += 2  # When this algorithm executes again, the current turn will be incremented by two.
        print("Action to take:", action_to_take)
        return action_to_take

    def execute_forward_algorithm(self, measurement_index, measurement_color):
        measurement_position = self.convert_index_to_tuple(measurement_index)
        self.update_belief()
        self.update_belief_with_obs(measurement_position, measurement_color)

    # Executes the elapse time phase of the Forward Algorithm and replaces the previous belief.
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
                        # It is possible to go from the current value (a position) to the position (i, j) being
                        # analyzed.
                        # Value[position] is the probability to go from the value (a position) to the position (i, j)
                        # being analyzed.
                        # self.belief_prev_t[key[0]][key[1]] means the probability of the adversary being in that
                        # position (key[0]], [key[1]) on the previous time (previous belief_grid).
                        b_prime += (value[position] * self.belief_prev_t[key[0]][key[1]])
                # The position being analyzed is updated with its new belief value.
                self.belief_after_t[position[0]][position[1]] = b_prime

    def update_belief_with_obs(self, measurement_position, measurement_color):
        self.incorporate_observation(measurement_position, measurement_color)
        # The original belief table is updated after elapsing time and incorporating observation.
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
                # The belief of the position being analyzed (i, j) after elapsing time is updated with the observation
                # incorporation.
                self.belief_after_t_and_obs[position[0]][position[1]] = prob_color_given_dist_to_measurement * b_prime
                normalization_factor += self.belief_after_t_and_obs[position[0]][position[1]]
        # Normalization.
        for i in range(0, self.GRID_WIDTH_HEIGHT):
            for j in range(0, self.GRID_WIDTH_HEIGHT):
                self.belief_after_t_and_obs[i][j] /= normalization_factor

    # Redistributes all particles in the given position when the agent shot and failed.
    def redistribute_particles(self, index_for_redistribution):
        delivery_position = self.convert_index_to_tuple(index_for_redistribution)
        particles_in_delivery_position = self.start_particles[delivery_position[0]][delivery_position[1]]
        if particles_in_delivery_position != self.PARTICLES_QUANTITY:
            # The probability was distributed in various cells and not concentrated in an unique one.
            print("Start particles before redistribution.")
            self.print_formatted_grid(self.start_particles)
            # The adversary was not in that position. Therefore, all particles on that position have to redistributed.
            self.start_particles[delivery_position[0]][delivery_position[1]] = 0
            remaining_particles = self.PARTICLES_QUANTITY - particles_in_delivery_position
            for i in range(0, self.GRID_WIDTH_HEIGHT):
                for j in range(0, self.GRID_WIDTH_HEIGHT):
                    # Corresponding particles for every position are:
                    # Number of particles in that position divided by the remaining particles on the grid.
                    # Multiplied by the particles to redistribute.
                    # Rounded to get only integer results (number of particles is an integer value).
                    # It is done in order to redistribute particles proportionally to the existing ones.
                    corresponding_particles = round((self.start_particles[i][j] / remaining_particles) *
                                                    particles_in_delivery_position)
                    print("Corresponding particles to position", i, j, "=", corresponding_particles)
                    # Particles are added to that position according to the proportion of the already existing particles
                    # there.
                    self.start_particles[i][j] += corresponding_particles
        print("Start particles after redistribution.")
        self.print_formatted_grid(self.start_particles)

    def execute_particle_filtering(self, measurement_index, measurement_color):
        measurement_position = self.convert_index_to_tuple(measurement_index)
        self.update_belief_using_pf()
        self.update_belief_with_obs_using_pf(measurement_position, measurement_color)
        self.resample_particles()

    # Executes the elapse time phase of the Particle Filtering and replaces the previous belief.
    def update_belief_using_pf(self):
        self.elapse_time_using_pf()
        print("Start particles after elapsing time.")
        self.print_formatted_grid(self.start_particles)
        print("Particles after t.")
        self.print_formatted_grid(self.particles_after_t)
        self.start_particles = copy.deepcopy(self.particles_after_t)
        # The grid after elapsing time has to be clean to avoid accumulating more particles than exist.
        self.particles_after_t = copy.deepcopy(self.empty_grid)
        print("-----------")
        print("UPDATED BELIEF USING PF AFTER ELAPSING TIME:")
        print("-----------")
        self.print_formatted_grid(self.start_particles)

    def elapse_time_using_pf(self):
        for i in range(0, self.GRID_WIDTH_HEIGHT):
            for j in range(0, self.GRID_WIDTH_HEIGHT):
                position = (i, j)
                particles_in_position = self.start_particles[position[0]][position[1]]
                # Loop over every particle on that position.
                count = 1
                while count <= particles_in_position:
                    print("Enter while in position", position[0], position[1], "for the", count, "time.")
                    next_position = None
                    cumulative_prob = 0
                    random_value = random.random()  # Random float x, 0.0 <= x < 1.0.
                    for key, value in self.transition_prob[position].items():
                        # Value is the transition probability from the position being analyzed to the key position.
                        cumulative_prob += value
                        if random_value <= cumulative_prob:
                            next_position = key
                            break  # The next position for the current particle was found.
                    self.start_particles[position[0]][position[1]] -= 1  # Particle goes away from that position.
                    # Particle is placed on that position.
                    self.particles_after_t[next_position[0]][next_position[1]] += 1
                    count += 1

    def update_belief_with_obs_using_pf(self, measurement_position, measurement_color):
        self.incorporate_observation_using_pf(measurement_position, measurement_color)
        # The original belief table is updated after elapsing time and incorporating observation.
        self.belief_prev_t = copy.deepcopy(self.belief_after_t_and_obs)
        print("-----------")
        print("UPDATED BELIEF AFTER OBSERVATION USING PF:")
        print("-----------")
        self.print_formatted_grid(self.belief_after_t_and_obs)

    def incorporate_observation_using_pf(self, measurement_position, measurement_color):
        normalization_factor = 0
        for i in range(0, self.GRID_WIDTH_HEIGHT):
            for j in range(0, self.GRID_WIDTH_HEIGHT):
                position = (i, j)
                dist_to_measurement = self.calc_distance(measurement_position, position)
                prob_color_given_dist_to_measurement \
                    = self.sonar_prob_given_dist[dist_to_measurement][measurement_color]
                number_of_particles = self.start_particles[position[0]][position[1]]
                # The number of particles is weighted with the probability of the observed color given the distance.
                self.belief_after_t_and_obs[position[0]][position[1]] = prob_color_given_dist_to_measurement * \
                                                                        number_of_particles
                normalization_factor += self.belief_after_t_and_obs[position[0]][position[1]]
        # Normalization.
        for i in range(0, self.GRID_WIDTH_HEIGHT):
            for j in range(0, self.GRID_WIDTH_HEIGHT):
                self.belief_after_t_and_obs[i][j] /= normalization_factor

    # Particles are created again but using the current distribution, not the uniform one.
    def resample_particles(self):
        self.start_particles = copy.deepcopy(self.empty_grid)
        count = 1
        while count <= self.PARTICLES_QUANTITY:
            position_assigned = None
            cumulative_prob = 0
            random_value = random.random()  # Random float x, 0.0 <= x < 1.0.
            for i in range(0, self.GRID_WIDTH_HEIGHT):
                for j in range(0, self.GRID_WIDTH_HEIGHT):
                    cumulative_prob += self.belief_after_t_and_obs[i][j]
                    if random_value <= cumulative_prob:
                        position_assigned = (i, j)
                        break
                if position_assigned is not None:
                    # A particle is created in the assigned position.
                    self.start_particles[position_assigned[0]][position_assigned[1]] += 1
                    break
            count += 1
        print("-----------")
        print("UPDATED PARTICLES AFTER OBSERVATION AND RESAMPLING:")
        print("-----------")
        self.print_formatted_grid(self.start_particles)

    def calculate_risk_level(self, adv_action, adv_action_param, adv_action_result):
        print("Adversary action:", adv_action, "Adversary action param:", adv_action_param, "Adversary action result:", adv_action_result)
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
                # Adversary shot the agent.
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
            # Adversary observed.
            # Every risk is weighted with the risk level according to the observed color.
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
        # Allows to check which is the movement that achieves the maximum distance to the adversary sight in combination
        # with the maximum posterior possible movements.
        max_distance = 0
        if i - 1 >= 0:  # Up.
            possible_position = (i - 1, j)
            distance_to_adv_sight = self.calc_distance(possible_position, adv_sight)
            distance_to_adv_sight += self.count_possible_movements(possible_position)
            if distance_to_adv_sight > max_distance:
                max_distance = distance_to_adv_sight
                best_position = possible_position
        if j + 1 <= 4:  # Right.
            possible_position = (i, j + 1)
            distance_to_adv_sight = self.calc_distance(possible_position, adv_sight)
            distance_to_adv_sight += self.count_possible_movements(possible_position)
            if distance_to_adv_sight > max_distance:
                max_distance = distance_to_adv_sight
                best_position = possible_position
        if i + 1 <= 4:  # Down.
            possible_position = (i + 1, j)
            distance_to_adv_sight = self.calc_distance(possible_position, adv_sight)
            distance_to_adv_sight += self.count_possible_movements(possible_position)
            if distance_to_adv_sight > max_distance:
                max_distance = distance_to_adv_sight
                best_position = possible_position
        if j - 1 >= 0:  # Left.
            possible_position = (i, j - 1)
            distance_to_adv_sight = self.calc_distance(possible_position, adv_sight)
            distance_to_adv_sight += self.count_possible_movements(possible_position)
            if distance_to_adv_sight > max_distance:
                max_distance = distance_to_adv_sight
                best_position = possible_position
        return self.convert_tuple_to_index(best_position)  # Returns the best position as an index.

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
        if i - 1 >= 0:  # Up.
            possible_movements += 1
        if j + 1 <= 4:  # Right.
            possible_movements += 1
        if i + 1 <= 4:  # Down.
            possible_movements += 1
        if j - 1 >= 0:  # Left.
            possible_movements += 1
        return possible_movements

    def get_max_value_in_table_with_index(self, table):
        np_table = np.array(table)
        max_element_index = np_table.argmax()
        i, j = np.unravel_index(max_element_index, np_table.shape)
        return np_table[i, j], (max_element_index + 1)