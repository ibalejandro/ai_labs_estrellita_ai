from Agente_A_Star import AgenteAStar

agent = AgenteAStar()
agent.update_belief()

print(agent.get_action_to_take(1, None, [], 15))
print(agent.get_action_to_take(1, agent.RED, [1, 20, 0], 15))
