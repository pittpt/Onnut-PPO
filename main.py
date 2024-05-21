import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from env_Onnut import SumoEnvironment
import gym
import msgpack_numpy as m
from ppoAgent import maPPO

m.patch()

OUTPUT_FOLDER = 'PPO-main-batchx3' 
LOAD_FOLDER='PPO-main-batchx3' #for loading checkpoints

EPOCHS = 10
MAX_EP_STEPS = 960
EPISODES_PER_BATCH = 3
RUN = 350

UPDATE_PER_ITR = 3
GAMMA = 0.95
LR = 0.005

#==== max_green ====#
max_green_ONNUT = 90
max_green_VIRTUAL = 90

env = SumoEnvironment(net_file='onnut.net.xml',
                            single_agent=False,
                            out_csv_name='{}/onnut-ppo'.format(OUTPUT_FOLDER),
                            use_gui=False, 
                            num_seconds=68400,
                            yellow_time=0,
                            min_green=15,
                            max_green_onnut= max_green_ONNUT,
                            max_green_virtual= max_green_VIRTUAL,
                            run=RUN,
                            name=OUTPUT_FOLDER
                            )
# Extract env info
action_splits = []
for ts in env.ts_ids:
    if isinstance(env.action_spaces(ts), gym.spaces.Discrete):
        action_splits.append([env.action_spaces(ts).n]) # action_space.n means Discrete(3)
    elif isinstance(env.action_spaces(ts), gym.spaces.MultiDiscrete):
        action_splits.append(env.action_spaces(ts).nvec) # action_space.nvec means (3): [3,2]

n_actions_each = [sum(a) for a in action_splits]
input_dims = [] 
for ts in env.ts_ids:
    input_dims.append(env.observation_spaces(ts).shape[0])

hyperparameters = {'timesteps_per_batch': MAX_EP_STEPS * EPISODES_PER_BATCH, 'max_timesteps_per_episode': MAX_EP_STEPS, 'gamma': GAMMA, 'n_updates_per_iteration': UPDATE_PER_ITR,
							'lr': LR} #more adjustments can be done in hyperparameters function in ppoAgent.py

agents = maPPO(env, n_agents=list(env.traffic_signals.keys()), num_actions=n_actions_each, 
               input_dims=input_dims, **hyperparameters)

agents.load_models('./{}/multiagent_checkpoint.pth'.format(LOAD_FOLDER)) #comment out if this is the first time running

agents.learn(EPOCHS * MAX_EP_STEPS) #learn specifies total number of steps instead of number of epochs, so we do epochs x timesteps

env.close()