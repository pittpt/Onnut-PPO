from env_Onnut import SumoEnvironment
from ql_agent import QLAgent
from epsilon_greedy import EpsilonGreedy
import random

if __name__ == '__main__':

    alpha = 0.01
    gamma = 0.95
    decay = 1
    runs = 1

    env = SumoEnvironment(net_file='onnut_fix2_20Apr.net.xml',
                          single_agent=False,
                          out_csv_name='outputs/onnut-ql',
                          use_gui=True,
                          num_seconds=68400,
                          yellow_time=0,
                          min_green=15)

    for run in range(1, runs+1):
        initial_states = env.reset()
        print(initial_states)
        ql_agents = {ts: QLAgent(
                                # starting_state=env.encode(initial_states[ts], ts),
                                 starting_state= tuple(initial_states[ts]),
                                 state_space=env.observation_spaces(ts),
                                 action_space=env.action_spaces(ts),
                                 alpha=alpha,
                                 gamma=gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)) for ts in env.ts_ids}
        infos = []
        done = {'__all__': False}
        while not done['__all__']:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(action=actions)
            # print(s)
            
            for agent_id in s.keys():
                # ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
                ql_agents[agent_id].learn(next_state=tuple(s[agent_id]), reward=r[agent_id])
        env.save_csv('outputs/QL', run)
        env.close()


