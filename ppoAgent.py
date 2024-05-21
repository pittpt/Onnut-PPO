import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from ppoNetwork import FeedForwardNN
import random

class maPPO:
	def __init__(self, env, n_agents, num_actions, input_dims, **hyperparameters):
		# Make sure the environment is compatible with our code
		assert(type(env.observation_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Init multi agent networks
		self.env = env
		self.actors = {} # {ts: <FeedForwardNN>}
		self.critics = {}
		self.actor_optims = {}
		self.critic_optims = {}
		self.n_agents = n_agents
		self.device = 'cpu'
		# self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		for agent_idx in range(len(self.n_agents)):
			# Initialize actor and critic networks
			self.actors[n_agents[agent_idx]] = FeedForwardNN(input_dims[agent_idx], num_actions[agent_idx]).to(self.device) # e.g.  FeedForwardNN(21,2)
			self.critics[n_agents[agent_idx]] = FeedForwardNN(input_dims[agent_idx], 1).to(self.device)  # Assuming critic outputs a single value
            
			# Initialize optimizers for actor and critic
			self.actor_optims[n_agents[agent_idx]] = torch.optim.Adam(self.actors[n_agents[agent_idx]].parameters(), lr=self.lr)
			self.critic_optims[n_agents[agent_idx]] = torch.optim.Adam(self.critics[n_agents[agent_idx]].parameters(), lr=self.lr)
		
		print(self.device)
		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

	def learn(self, total_timesteps):
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		
		t_so_far = 0 # Timesteps simulated so far (total)
		i_so_far = 0 # Iterations ran so far

		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			print('Learn >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)
			print('Total timesteps: ', t_so_far)

			# Increment the number of iterations
			i_so_far += 1

			# # Logging timesteps so far and iterations so far
			# self.logger['t_so_far'] = t_so_far
			# self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = {agent_id: batch_rtgs[agent_id] - V[agent_id].detach() for agent_id in V} # {ts: A_K}			# ALG STEP 5

			# Normalize
			A_k = {agent_id: (A_k[agent_id] - A_k[agent_id].mean().detach()) / (A_k[agent_id].std().detach() + 1e-10) for agent_id in A_k}  # {ts: A_K (normal)}

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V , curr_log_probs = self.evaluate(batch_obs, batch_acts) # {ts: Tensor[]}
				assert V[self.n_agents[0]].shape == batch_rtgs[self.n_agents[0]].shape, f"Mismatch in V and batch_rtgs shapes for agent {self.n_agents[0]}"
				assert V[self.n_agents[1]].shape == batch_rtgs[self.n_agents[1]].shape, f"Mismatch in V and batch_rtgs shapes for agent {self.n_agents[1]}"

				for agent_id in self.n_agents:

					# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
					ratios = torch.exp(curr_log_probs[agent_id] - batch_log_probs[agent_id]) # batch_log_probs -> {ts: Tensor[]}

					# Calculate surrogate losses.
					surr1 = ratios * A_k[agent_id]
					surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k[agent_id]

					# Calculate actor and critic losses.
					actor_loss = (-torch.min(surr1, surr2)).mean()
					critic_loss = nn.MSELoss()(V[agent_id], batch_rtgs[agent_id])

					# Calculate gradients and perform backward propagation for actor network
					self.actor_optims[agent_id].zero_grad()
					actor_loss.backward(retain_graph=True)
					self.actor_optims[agent_id].step()

					# Calculate gradients and perform backward propagation for critic network
					self.critic_optims[agent_id].zero_grad()
					critic_loss.backward()
					self.critic_optims[agent_id].step()

					# Log actor loss
					self.logger['actor_losses'].append(f"{agent_id}: {actor_loss.item()}")

			# Print a summary of our training so far
			# self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				self.save_models()
			

	def compute_kl_divergence(self, old_log_probs, new_log_probs):
		# Calculate KL divergence between old and new log probabilities
		old_log_probs = old_log_probs.unsqueeze(1)
		new_log_probs = new_log_probs.unsqueeze(1)
		kl_div = (torch.exp(old_log_probs) * (old_log_probs - new_log_probs)).sum(axis=1).mean()
		return kl_div
	
	def rollout(self):
		# Batch data. For more details, check function header.
		batch_obs = [] #all are list of [{ts1: obs, ts2: obs2},{},...] TENSOR
		batch_acts = [] # list of action lists [[1,2],[0,2]....] TENSOR
		batch_log_probs = [] #list of probs [[s,ss],[x,xx]....] TENSOR
		batch_rews = [] # list of list of dict [[{} , ...] , [{} , ...] , ...]
		batch_rtgs = [] # list of rewards from each ts {ts: Tensor[rewards]}
		batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			print('batch >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

			random_seed = t
			random.seed(random_seed)

			ep_rews = [] # rewards collected per episode
			obs = self.env.reset(random_seed)
			done = [False]*3

			# Run an EPISODE
			for ep_t in range(self.max_timesteps_per_episode):

				t += 1 # Batch increment

				# Track observations in this batch
				batch_obs.append(obs)
				action, log_prob = self.get_action(obs) # List of actions [1,2] and log prob [x,xx]
				print('-------Episode step{}-------'.format(ep_t))
				print(action)

				actions = dict(zip(self.n_agents, action))
				obs, rew, done, _ = self.env.step(actions) 

				# Track recent reward, action, and action log probability
				ep_rews.append(rew) # [{}]
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				done = list(done.values())
				if any(done):
					print(done,'DONE')
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1) 
			batch_rews.append(ep_rews)

			self.env.save_score()

		# Reshape data as tensors in the shape specified in function description, before returning

		# [{}] -> {ts: Tensor[]}
		batch_obs = {agent_id: torch.tensor(np.stack([timestep[agent_id] for timestep in batch_obs], axis=0), dtype=torch.float) for agent_id in self.n_agents}
		# [[]] -> {ts: Tensor[]}
		batch_acts = {agent_id: torch.tensor([timestep[index] for timestep in batch_acts], dtype=torch.long) for index, agent_id in enumerate(self.n_agents)}
		# [[Tensor()]] -> {ts: Tensor[]}
		batch_log_probs = {agent_id: torch.stack([timestep[index] for timestep in batch_log_probs], dim=0) for index, agent_id in enumerate(self.n_agents)}

		batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

		# # Log the episodic returns and episodic lengths in this batch.
		# self.logger['batch_rews'] = batch_rews
		# self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	# batch_rews = [ [ {} ] ]
	# return {ts: Tensor[rtgs]}
	def compute_rtgs(self, batch_rews):
	
		agent_rtgs = {agent_id: [] for agent_id in batch_rews[0][0].keys()}
		
		 # Iterate through each episode || REVERSE OR NOT
		for ep_rews in reversed(batch_rews):  # ep_rews is a list of dictionaries at each timestep
			# Initialize a dictionary to hold discounted rewards so far for each agent
			discounted_rewards = {agent_id: 0 for agent_id in ep_rews[0].keys()}

			# Reverse iterate through the episode rewards to compute discounted rewards
			for timestep_rewards in reversed(ep_rews):
				for agent_id, rew in timestep_rewards.items():
					# Update the discounted reward for each agent
					discounted_rewards[agent_id] = rew + (discounted_rewards[agent_id] * self.gamma)
					# Prepend the current discounted reward to the agent's RTGs list
					agent_rtgs[agent_id].insert(0, discounted_rewards[agent_id])

		# Convert the RTGs for each agent into tensors
		for agent_id in agent_rtgs.keys():
			agent_rtgs[agent_id] = torch.tensor(agent_rtgs[agent_id], dtype=torch.float)

		return agent_rtgs

	# obs = {ts: []}
	# return actions = [[]] | log_probs = [[]]
	def get_action(self, obs):

		actions = []
		log_probs = []
		
		for agent_idx, agent in self.actors.items():
			# Query the actor network for a mean action
			logit = agent(obs[agent_idx])

			dist = torch.distributions.Categorical(logits=logit)

			# Sample an action from the distribution
			action = dist.sample()

			# Calculate the log probability for that action
			log_prob = dist.log_prob(action)

			# Store the sampled action and the log probability
			actions.append(action.item())
			log_probs.append(log_prob.detach())

		# Return the sampled action and the log probability of that action in our distribution
		return actions, log_probs

	# batch, acts = {ts: Tensor[]} 
	# Return V, log_probs = {ts: Tensor[]}
	def evaluate(self, batch_obs, batch_acts):
		
		V = {}
		log_probs = {}

		for agent_id, agent in self.actors.items():
			obs = batch_obs[agent_id].to(self.device) # Tensor[obs,...]
			acts = batch_acts[agent_id].to(self.device)
			# acts = Tensor[0,1,1,0,1,0,1,....] for that agent

			# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
			V[agent_id] = self.critics[agent_id](obs).squeeze() # List of critics [v from obs1, v from obs2, ...]

			# Calculate the log probabilities of batch actions using most recent actor network.
			# This segment of code is similar to that in get_action()
			logit = agent(obs)
			dist = torch.distributions.Categorical(logits=logit)
			log_probs[agent_id] = dist.log_prob(acts)

		# Return the value vector V of each observation in the batch {ts: Tensor[V]}
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 2880                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 960            # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 2                              # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def save_models(self, filepath='./multiagent_checkpoint.pth'):
		filepath='./{}/multiagent_checkpoint.pth'.format(self.env.name)
		checkpoint = {
			'actors': {agent_id: self.actors[agent_id].state_dict() for agent_id in self.n_agents},
			'critics': {agent_id: self.critics[agent_id].state_dict() for agent_id in self.n_agents},
			'actor_optims': {agent_id: self.actor_optims[agent_id].state_dict() for agent_id in self.n_agents},
			'critic_optims': {agent_id: self.critic_optims[agent_id].state_dict() for agent_id in self.n_agents},
		}
		print('Saved: ', checkpoint, ' to ', filepath)
		torch.save(checkpoint, filepath)

	def load_models(self, filepath='./multiagent_checkpoint.pth'):
		checkpoint = torch.load(filepath)
		print('Loaded: ', checkpoint, ' from ', filepath)
		for agent_id in self.n_agents:
			self.actors[agent_id].load_state_dict(checkpoint['actors'][agent_id])
			self.critics[agent_id].load_state_dict(checkpoint['critics'][agent_id])
			self.actor_optims[agent_id].load_state_dict(checkpoint['actor_optims'][agent_id])
			self.critic_optims[agent_id].load_state_dict(checkpoint['critic_optims'][agent_id])

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []                      