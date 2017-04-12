from Agente_A_Star import AgenteAStar

agent = AgenteAStar()

print(agent.get_action_to_take(1, None, [], 5))
print(agent.get_action_to_take(1, agent.RED, [3, 4, None], 5))
print(agent.get_action_to_take(1, agent.RED, [2, 5, agent.GREEN], 5))
print(agent.get_action_to_take(1, agent.RED, [1, 5, 1], 10))
