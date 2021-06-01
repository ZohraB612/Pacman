from HER import HindsightReplayMem
from HER import DQN
import numpy as np
import torch
torch.manual_seed(0)
import os
from Actions import Actions

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# AdamStelmaszczyk (2020). Github: https://github.com/AdamStelmaszczyk/dqn
# Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P.,
# McGrew, B., Tobin, J., Abbeel, P., and Zaremba, W. (2017). ‘Hindisght Experience Replay’.
#[Online]. Available at: https://arxiv.org/abs/1707.01495 [Accessed 2nd April 2021]
# TianhongDai (2021). Github : https://github.com/TianhongDai/hindsight-experience-replay
# Hemilpanchiwala (2021). Github: https://github.com/hemilpanchiwala/Hindsight-Experience-Replay
# Kwea123 (2018). Github: https://github.com/kwea123/hindsight_experience_replay


class DQN_HER_params:
    def __init__(self):
        self.learning_rate = 0.0001
        self.input_dims = 4
        self.gamma = 0.99
        self.epsilon = 1
        self.batch_size = 64
        self.memory_size = 10000
        self.replace_network_count = 50
        self.input_dims = 4
        self.n_actions = len(Actions.All)
        self.decay_epsilon = 1e-5
        self.min_epsilon = 0.1


class DQNAgentWithHER(object):
    def __init__(self, params=DQN_HER_params(), checkpoint_dir='/tmp/ddqn/'):
        self.learning_rate = params.learning_rate
        self.n_actions = params.n_actions
        self.input_dims = params.input_dims
        self.gamma = params.gamma
        self.epsilon = params.epsilon
        self.batch_size = params.batch_size
        self.memory_size = params.memory_size
        self.replace_network_count = params.replace_network_count
        self.decay_epsilon = params.decay_epsilon
        self.min_epsilon = params.min_epsilon
        self.checkpoint_dir = checkpoint_dir
        self.action_indices = [i for i in range(self.n_actions)]
        self.learn_steps_count = 0

        self.q_eval = DQN.DeepQNetwork(learning_rate=self.learning_rate, n_actions=self.n_actions,
                                       input_dims=2 * self.input_dims, name='q_eval',
                                       checkpoint_dir=checkpoint_dir)

        self.q_next = DQN.DeepQNetwork(learning_rate=self.learning_rate, n_actions=self.n_actions,
                                       input_dims=2 * self.input_dims, name='q_next',
                                       checkpoint_dir=checkpoint_dir)

        self.experience_replay_memory = HERmemory.HindsightExperienceReplayMemory(memory_size=self.memory_size,
                                                                                  input_dims=self.input_dims,
                                                                                  n_actions=self.n_actions)

    def decrement_epsilon(self):
        
        """
        Epsilon is decayed until it reaches its minimum value
        """
        
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay_epsilon
        else:
            self.epsilon = self.min_epsilon

    def update(self, state, action, reward, next_state, done, goal):
        """
        Saves the experience to the hindsight experience replay memory
        """
        self.experience_replay_memory.add_experience(state=state, action=action,
                                                     reward=reward, next_state=next_state,
                                                     done=done, goal=goal)

    def get_sample_experience(self):
        
        """
        Provides a sample experience from HER memory
        """
        
        # set value network
        state, action, reward, next_state, done, goal = self.experience_replay_memory.get_random_experience(
            self.batch_size)

        t_state = torch.tensor(state).to(self.q_eval.device)
        t_action = torch.tensor(action, dtype=torch.long).to(self.q_eval.device)
        t_reward = torch.tensor(reward).to(self.q_eval.device)
        t_next_state = torch.tensor(next_state).to(self.q_eval.device)
        t_done = torch.tensor(done).to(self.q_eval.device)
        t_goal = torch.tensor(goal).to(self.q_eval.device)

        return t_state, t_action, t_reward, t_next_state, t_done, t_goal

    def replace_target_network(self):
        
        """
        Updates the parameters after replace_network_count steps
        """
        
        if self.learn_steps_count % self.replace_network_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def get_action(self, obs, goal):
        
        """
        E-greedy policy is used to choose an action
        """
        
        state = obs.get_state_vector()
        if np.random.random() > self.epsilon:
            concat_state_goal = np.concatenate([state, goal])
            state = torch.tensor([concat_state_goal], dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)

            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.n_actions)

        return action

    def learn(self):
        if self.experience_replay_memory.counter < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        state, action, reward, next_state, done, goal = self.get_sample_experience()
        # spaces the batches evenly
        batches = torch.tensor(np.arange(self.batch_size), dtype=torch.long)

        concat_state_goal = torch.cat((state, goal), 1)
        concat_next_state_goal = torch.cat((next_state, goal), 1)

        q_pred = self.q_eval.forward(concat_state_goal)
        q_pred = q_pred[batches, action]
        q_next = self.q_next.forward(concat_next_state_goal).max(dim=1)[0]

        q_next[done] = 0.0
        q_target = reward + self.gamma * q_next

        # Next we perform backpropagation and calculate the loss
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.decrement_epsilon()
        self.learn_steps_count += 1

    def save_model(self):
        """
        Saves the values of q_eval and q_next at the checkpoint
        """
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_model(self):
        """
        Loads the values of q_eval and q_next at the checkpoint
        """
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
