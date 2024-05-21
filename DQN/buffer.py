from collections import deque
import random

class Buffer:
    def __init__(self,n_agents,buffer_size,batch_size):
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.replay_buffers = []
        for agent_idx in n_agents:
            self.replay_buffers.append(deque(maxlen=buffer_size))

    def store(self,transition):
        i = 0
        for agent_idx in self.n_agents:
            # print(agent_idx)
            # obs = transition[0][agent_idx]
            # actions = transition[1][agent_idx]
            # rewards = transition[2][agent_idx]
            # dones = transition[3][agent_idx]
            # new_obs = transition[4][agent_idx]

            obs = transition.get((agent_idx,'obs'))
            actions = transition.get((agent_idx,'actions'))
            rewards = transition.get((agent_idx,'rewards'))
            dones = transition.get((agent_idx,'dones'))
            new_obs = transition.get((agent_idx,'new_obs'))

            agent_transition = (obs, actions, rewards, dones, new_obs)
            self.replay_buffers[i].append(agent_transition)
            i+=1

    def sample(self):
        samples = []
        for agent_idx in range(len(self.n_agents)):
            samples.append(random.sample(self.replay_buffers[agent_idx], self.batch_size))
        return samples
