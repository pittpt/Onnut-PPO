import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np

from buffer import Buffer
from madqn import maDQN
from env_Onnut import SumoEnvironment
import gym
import time
import random

import msgpack
import msgpack_numpy as m
m.patch()

#KubeMQ Publish 

from kubemq.events.lowlevel.event import Event
from kubemq.events.lowlevel.sender import Sender

#KubeMQ Subscribe
from kubemq.events.subscriber import Subscriber
from kubemq.tools.listener_cancellation_token import ListenerCancellationToken
from kubemq.subscription.subscribe_type import SubscribeType
from kubemq.subscription.events_store_type import EventsStoreType
from kubemq.subscription.subscribe_request import SubscribeRequest

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

def send_single_event(Message):
    sender = Sender("202.28.193.102:50000")
    event = Event(
        #metadata="EventMetaData",
        body=(Message).encode('UTF-8'),
        store=False,
        channel="IotcloudServe.RL",
        client_id="RL_pipeline"
    )
    try:
        send = sender.send_event(event)
        print(send)
    except Exception as err:
        print(
        "'error sending:'%s'" % (
            err
                    )
        )

def handle_incoming_events(event):
    if event:
        body = str(event.body)
        body = body[2:len(body)-1]
        print("Subscriber Received Event: Channel: %s, Body: %s, tags: %s"
        %(  event.channel,
            body,
            event.tags
        ))
        
        ENV_NAME = 'onnut'

        N_EPISODES = 1000
        MAX_STEPS = 960
        GAMMA=0.95
        BATCH_SIZE = 60
        BUFFER_SIZE=int(1000000)
        EPSILON_START= 1
        EPSILON_END=0.01
        EPSILON_DECAY= 10
        TARGET_UPDATE_FREQ = 30
        # TARGET_UPDATE_FREQ = 2
        LR = 0.01
        PRINT_INTERVAL = 1
        TRAIN_INTERVAL = 10
        # TRAIN_INTERVAL = 2
        MOVING_AVERAGE = 960
        best_score = 1
        action_splits = []

        #==== max_green ====#
        max_green_ONNUT = 90
        max_green_VIRTUAL = 90

        env = SumoEnvironment(net_file='onnut.net.xml',
                                    single_agent=False,
                                    out_csv_name='outputs/onnut-dqn',
                                    use_gui=False,
                                    num_seconds=68400,
                                    yellow_time=0,
                                    min_green=15,
                                    max_green_onnut= max_green_ONNUT,
                                    max_green_virtual= max_green_VIRTUAL
                                    )

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

            #======== for confidence interval ========#
            random_seed = step
            random.seed(random_seed)
            #========================================#

            obs =  env.reset(random_seed)

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

            env.save_score()
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

            elapsed_time = env.getTime((time.time() - start_time))
            print('episode', step, ' takes time', elapsed_time)

        env.close()
def handle_incoming_error(error_msg):
        print("received error:%s'" % (
            error_msg
        ))

if __name__ == "__main__":
    print("Subscribing to event on channel example")
    cancel_token=ListenerCancellationToken()


    # Subscribe to events without store
    subscriber = Subscriber("202.28.193.102:50000")
    subscribe_request = SubscribeRequest(
        channel="IotcloudServe.SUMO",
        client_id="RL_pipeline",
        events_store_type=EventsStoreType.Undefined,
        events_store_type_value=0,
        group="",
        subscribe_type=SubscribeType.Events
    )
    subscriber.subscribe_to_events(subscribe_request, handle_incoming_events, handle_incoming_error, cancel_token)
    app.run(host="localhost", port=52000)
