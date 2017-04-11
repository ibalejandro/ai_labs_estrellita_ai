from Agente_A_Star import AgenteAStar

agent = AgenteAStar()
agent.update_belief()
agent.set_measurement_position((0, 4))
agent.update_belief_with_obs(agent.measurement_position, "rojo")

print(agent.get_action_to_take(1, None, [], 15))
