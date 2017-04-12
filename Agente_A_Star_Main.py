from Agente_A_Star import AgenteAStar
import random

agent = AgenteAStar()

star_position = random.randint(1, agent.GRID_SIZE)

# r1 = agent.get_action_to_take(1, None, [], star_position)
# print(r1)
# print("========================================")
# print(agent.get_action_to_take(1, agent.RED, [3, 4, None], star_position))
# print("========================================")
# r3 = agent.get_action_to_take(1, agent.RED, [2, 5, agent.GREEN], star_position)
# star_position = r3[1]
# print(r3)
# print("========================================")
# r4 = agent.get_action_to_take(1, agent.RED, [1, 5, 1], star_position)
# star_position = r4[1]
# print(r4)
# print("========================================")
# print(agent.get_action_to_take(1, agent.RED, [3, 3, None], star_position))
# print("========================================")

# print("belief from main", agent.print_formatted_grid(agent.belief_prev_t))
# print(agent.max_probability_in_belief())

sonar = [agent.GREEN, agent.YELLOW, agent.ORANGE, agent.RED]
r = agent.get_action_to_take(1, None, [], star_position)

# for i in range(1, 30):
#     action = r[0]
#     if action == agent.SHOOT:
#         r = agent.get_action_to_take(1, bool(random.getrandbits(1)), [agent.MOVE, random.randint(1, 4), None], star_position)
#     if action == agent.OBSERVE:
#         r = agent.get_action_to_take(1, random.choice(sonar), [agent.MOVE, random.randint(1, 4), None], star_position)
#     elif action == agent.MOVE:
#         star_position = r[1]
#         r = agent.get_action_to_take(1, None, [agent.MOVE, random.randint(1, 4), None], star_position)
#     print(r)
#     print("========================================")

for i in range(1, 30):
    action = r[0]
    if action == agent.SHOOT:
        r = agent.get_action_to_take(1, bool(random.getrandbits(1)), [agent.OBSERVE, random.randint(1, 25), random.choice(sonar)], star_position)
    if action == agent.OBSERVE:
        r = agent.get_action_to_take(1, random.choice(sonar), [agent.OBSERVE, random.randint(1, 25), random.choice(sonar)], star_position)
    elif action == agent.MOVE:
        star_position = r[1]
        r = agent.get_action_to_take(1, None, [agent.OBSERVE, random.randint(1, 25), random.choice(sonar)], star_position)
    print(r)
    print("========================================")
