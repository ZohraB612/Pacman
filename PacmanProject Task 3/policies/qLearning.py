from policies.Policy import *
from Agent import AgentObs

# Galkin, O. (2021). Q-Learning For House Navigation Task

"""
QLearing parameters
"""
class QLearningParams:
    def __init__(self):
        self.alpha = 0.99
        self.gamma = 0.8
        self.eps = 0.05

"""
Qlearning Policy algorithm
"""
class QLearning(Policy):

    def __init__(self, env, params=QLearningParams(), verbose=False):
        # We create a 3D numpy array with the q-values for each state and action combinations
        # We use np.zeroes to initialise all our q-values to zero
        self.q_values = np.zeros((AgentObs.get_max_state_nums(env), len(Actions.All)))
        self.alpha = params.alpha
        self.gamma = params.gamma
        # exploration prob rate
        self.eps = params.eps
        self.verbose = verbose

    """
    Epsilon greedy action selection
    """
    def get_action(self, obs, goal = None):
        state = obs.get_state()
        p = np.random.random()
        # Exploration
        if p < self.eps:
            #print("Taking Random direction")
            return self.get_random_action()
        # Exploitation
        # pos = obs.agent_pos
        best_qvalue = np.max(self.q_values[state, :])
        best_action_id = np.argmax(self.q_values[state, :])
        best_action = Actions.id2action(best_action_id)

        # take the second best action
        if obs.is_ghost_there(best_action):
            best_action_id = self.q_values[state, :].argsort()[-2]
            best_action = Actions.id2action(best_action_id)

        if self.verbose:
            print("--------Current State:")
            obs.print_state(state)
            print(self.q_values[state, :])
            print("------")

        return best_action

    #Updates the Q values tablle
    # obs - the current state the agent ( position )
    # action - the action the agent takes from the current state
    # reward - points that the agent received from taking the action
    # next_obs - the next state of the agent ( position)
    def update(self, obs, action, reward, next_obs):
        state = obs.get_state()
        old_value = self.q_values[state, Actions.action2id[action]]

        next_state = next_obs.get_state()
        next_state_value = np.max(self.q_values[next_state, :])

        # calculate new_q value
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_state_value)

        if self.verbose:
            print("State ", state)
            print("Old Q Values: ", self.q_values[state, :])

        # set the value back into the table
        self.q_values[state, Actions.action2id[action]] = new_value
        if self.verbose:
            print("New Q values: ", self.q_values[state, :])
