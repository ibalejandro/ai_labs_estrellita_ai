from Agente_A_Star import AgenteAStar

agent = AgenteAStar()

print(agent.get_action_to_take(1, None, [], 5))
print("========================================")
print(agent.get_action_to_take(1, agent.RED, [3, 4, None], 5))
print("========================================")
print(agent.get_action_to_take(1, agent.RED, [2, 5, agent.GREEN], 5))
print("========================================")
print(agent.get_action_to_take(1, agent.RED, [1, 5, 1], 10))
print("========================================")

# print("belief from main", agent.print_formatted_grid(agent.belief_prev_t))
# print(agent.max_probability_in_belief())