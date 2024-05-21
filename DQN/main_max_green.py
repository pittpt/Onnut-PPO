import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np

from buffer import Buffer
from madqn import maDQN
from env_Onnut import SumoEnvironment
import gym
import time
import random
random.seed(0)

import msgpack
import msgpack_numpy as m
m.patch()

ENV_NAME = 'onnut'

N_EPISODES = 10
MAX_STEPS = 960
GAMMA=0.95
BATCH_SIZE = 60
BUFFER_SIZE=int(1000000)
EPSILON_START= 1
EPSILON_END=0.01
EPSILON_DECAY= 10
TARGET_UPDATE_FREQ = 60
LR = 0.01
PRINT_INTERVAL = 1
TRAIN_INTERVAL = 60
MOVING_AVERAGE = 960
best_score = 1
action_splits = []
max_green_ONNUT_list = [90,120,150,180]
max_green_VIRTUAL_list = [30,60,90,120,150]

for max_green_ONNUT in max_green_ONNUT_list :
    for max_green_VIRTUAL in max_green_VIRTUAL_list :

        #============ For PP run on cloud ============#
        #skip the couple of hyperparam that already run (90,30)
        if (max_green_ONNUT == 90) & (max_green_VIRTUAL == 30) :
            continue

        # elif (max_green_ONNUT == 90) & (max_green_VIRTUAL == 60) :
        #     continue

        env = SumoEnvironment(net_file='onnut.net.xml',
                                    single_agent=False,
                                    out_csv_name='outputs_max_green/onnut-dqn',
                                    use_gui=True,
                                    num_seconds=68400,
                                    yellow_time=0,
                                    min_green=15,
                                    max_green_onnut= max_green_ONNUT,
                                    max_green_virtual= max_green_VIRTUAL)

        for ts in env.ts_ids:
            if isinstance(env.action_spaces(ts), gym.spaces.Discrete):
                action_splits.append([env.action_spaces(ts).n]) # action_space.n means Discrete(3)
            elif isinstance(env.action_spaces(ts), gym.spaces.MultiDiscrete):
                action_splits.append(env.action_spaces(ts).nvec) # action_space.nvec means (3)

        n_actions_each = [sum(a) for a in action_splits]

        input_dims = []
        for ts in env.ts_ids:
            input_dims.append(env.observation_spaces(ts).shape[0])

        maDQN_agents = maDQN(n_agents=list(env.traffic_signals.keys()), num_actions=n_actions_each, input_dims= input_dims,
                                learning_rate=LR,
                                gamma=GAMMA, env_name=ENV_NAME)

        replay_buffer = Buffer(n_agents=list(env.traffic_signals.keys()),buffer_size=BUFFER_SIZE,batch_size=BATCH_SIZE)

        score_history = []
        transition = {}
        # Main Training Loop
        for step in range(N_EPISODES):
            obs =  env.reset()

            done = [False]*3
            episode_reward = 0
            episode_step = 0
            start_time = time.time()
            epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

            if step >=TARGET_UPDATE_FREQ and step % TARGET_UPDATE_FREQ == 0:
                maDQN_agents.target_update()

            while not any(done):
                rnd_sample = random.random()
                if rnd_sample <= epsilon:
                    # actions = np.random.randint(0,N_ACTIONS,size=N_AGENTS)
                    action_onnut = random.randint(0,1)
                    action_virtual = random.randint(0,2)
                    actions = [action_onnut,action_virtual]
                else:
                    actions = maDQN_agents.choose_actions(obs)
                # print('-------------------')
                # print(actions)

                actions = dict(zip(env.traffic_signals.keys(), actions))
                new_obs, rewards, done, _  = env.step(actions)
                # transition = (obs, actions, rewards, dones, new_obs)
                for ts in env.ts_ids:
                    transition[(ts, 'obs')] = obs[ts]
                    transition[(ts, 'actions')] = actions[ts]
                    transition[(ts, 'rewards')] = rewards[ts]
                    transition[(ts, 'dones')] = done[ts]
                    transition[(ts, 'new_obs')] = new_obs[ts]
                    replay_buffer.store(transition)

                if episode_step >= MAX_STEPS:
                    done = [True] * 3
                else:
                    done = list(done.values())
                episode_step += 1

                obs = new_obs
                episode_reward += sum(rewards.values())

                if step >= TRAIN_INTERVAL and  step % TRAIN_INTERVAL == 0:
                    batch = replay_buffer.sample()
                    maDQN_agents.learn(batch)

            # env.save_score()
            env.save_score_max_green()
            score_history.append(episode_reward)

            # Logging
            if step >= PRINT_INTERVAL and step % PRINT_INTERVAL == 0:
                avg_score = np.mean(score_history[-MOVING_AVERAGE:])
                print('Step:', step)
                print('Average Score: {:.2f}'.format(avg_score))
                np.save(ENV_NAME+'score_history.npy',np.array(score_history))
                if avg_score > best_score:
                    best_score = avg_score
                    maDQN_agents.save_checkpoint(best_score)
                    
            elapsed_time = env.getTime((time.time()-start_time))
            print('episode {}'.format(step), ' takes time : ',elapsed_time)
        
        env.close()
