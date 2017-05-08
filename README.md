# Estrellita_AI

Using some techniques learned in the second part of the AI course at Universidad EAFIT, an intelligent agent was implemented to play the "¿Estrellita donde estás?" game,  a game similar to Sea Battle.

## Table of contents
  * [Dependencies.](#dependencies)
  * [Clarifications.](#clarifications)
  * [Usage.](#usage)

## Dependencies
- [Python] 3.6.0

## Clarifications
  - The name of the main class is AgenteAStar.
  - The main class has a method called ``get_action_to_take(current_player, action_result, adversary_action, star_position)``, where the first parameter is the player who has to take an action, the second parameter is the result of the previous action of the current player, the third parameter is the adversary action after the current player acted and the last one is the star position of the currentplayer's star.
 - The agent can take three different actions on the game board (size is 5X5):
 - 1. **Shoot** (action = 1): it can shoot the adversary indicating a position between 1 and 25. If it hits the adversary, a 1 is returned. Otherwise, a 0 is returned.
 - 2. **Observe** (action = 2): it can use a noisy sonar to observe the adversary's location indicating a position between 1 and 25. The result of that action is any of the following colors: "verde" (green), "amarillo" (yellow), "anaranjado" (orange) and "rojo" (red). Given the distance from the measurement position to the actual position of the adversary, the color indicates the probability of the adversary being in the observed cell.
 - 3. **Move** (action = 3): it can move to another position indicating a number between 1 and 4 (1 = go up, 2 = go right, 3 = go down, 4 = go left). The result of that action is _None_.
  - An adversary action looks like this: _[type_of_action, action_parameter, action_result]._ The type of action, action parameter and action result are the same as the ones described above.
  - The winner of the game is the one that first hits its adversary ten times or the one with more points (one hit = one point) after 30 turns per player.
  -  The method ``get_action_to_take(current_player, action_result, adversary_action, star_position)`` returns a list with two items indicating the action that the ``current_player`` decided to take and the parameter for that action, respectively. For example, if the returned value is ``[1, 15]``, the ``current_player`` wants to shoot on the position 5 of the game board.

## Techniques
The agent uses two techniques learned in the second part of the AI course at Universidad EAFIT to improve performance in a strategic way:
  - **Forward Algorithm**: on this particular game, the idea is to keep track of the adversary actions and try to estimate its most likely location turn after turn. The Forward Algorithm is strategic to compute the distribution on the board for the adversary's star position when time passes and new evidence is discovered. As long as the hypothesis space is very large, this algorithm represents an good way to update the agent's belief.
  - **Particle Filtering**: when the agent hits the adversary for the first time, the hypothesis space is reduced to a particular region of the grid. A set of particles is placed on the impacted position, so that the agent's belief can be updated more locally when time passes and new observations are incorporated. Knowing the adversary's transition probabilities is also an advantage to infer the next posible locations of the opponent. Particle filtering is strategic to concentrate the distribution and improve the probability of hitting again on the next turn.

> When the agent hits the adversary for the first time, a transition to use Particle Filtering (local search) is executed.

> The agent calculates the risk level first according to the adversary's action and the returned result. That way, it decides if it´s necessary to move away or not.

> If the risk level is not dangerous for the agent, it decides whether it´s worth to shoot or not. If the probability is not significant, the agent doesn´t shoot and it prefers to observe and improve its belief about the adversary's location.

The following link contains an explanation of the techniques used to implement the intelligent agent, telling why they were used and what is the strategic purpose behind: [YouTube Video].

### Usage
This section is specific for the teacher of the AI course at Universidad EAFIT.

1. Copy the file named **"Agente_A_Star"** to the folder where your Jupyter Notebook file is.

2. In your Jupyter Notebook file, add the following line to your imports on the top.

    ```ssh
    from Agente_A_Star import AgenteAStar
    ```

3. Then, instantiate the variable which is going to be the agent. It is crucial to instantiate it before the game starts, because that one instance is going to be used during the whole game. Like this:

    ```ssh
    agente = AgenteAStar()

    Begin of the game
    ... Loop ...
    End of the game
    ```

4. To use the created instance in order to get the action to take from the intelligent agent during the game, invoke the ``get_action_to_take(current_player, action_result, adversary_action, star_position)`` method passing the current player, the result of its previous action, the last adversary action and the current player's star position as parameters. Like this:

    ```ssh
    [tipoAccion, parametroAccion] = agente.get_action_to_take(jugadorActual, resultado_accion[jugadorActual-1], accion_oponente[jugadorActual%2], tableros[jugadorActual-1].PosEstrellita())
    ```

5. Execute your Jupyter Notebook file and see the game progression. It's like **magic**!

> **Important note:** the star position that comes from the skeleton seems to be zero-based (i.e. between 0 and 24) and in fact it should be one-based (i.e. between 1 and 25). To avoid modifying the skeleton, the given star position was adjusted internally adding 1 to it.

[Python]: <https://www.python.org/downloads/>
[YouTube Video]: <https://youtu.be/e2s5pg0WLII>
