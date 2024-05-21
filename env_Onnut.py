import os
import sys
from pathlib import Path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd
from traffic_signal_Onnut import TrafficSignal
import random
from gym import spaces
from torch.utils.tensorboard import SummaryWriter

class SumoEnvironment(MultiAgentEnv):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    """

    def __init__(self, net_file, out_csv_name=None, use_gui=False, num_seconds=68400 #7pm
                 , time_to_teleport=900, delta_time=15, yellow_time=0, min_green=15
                 , max_green_onnut=135, max_green_virtual=30, single_agent=False, run=0
                 , name=None):
        self._net = net_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.begin_time = 54000
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        # max_green_onnut = 135,
        # max_green_virtual =  30,
        self.max_green_onnut = max_green_onnut
        self.max_green_virtual = max_green_virtual
        self.max_green =  {'cluster_1088409501_272206263_5136790697_70702637':max_green_onnut,'gneJ42':max_green_virtual}
        self.yellow_time = yellow_time
        # self.random_number = random_number

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # start only to retrieve information

        self.single_agent = single_agent
        self.ts_ids = ['cluster_1088409501_272206263_5136790697_70702637','gneJ42']
        self.ts_junction = {'cluster_1088409501_272206263_5136790697_70702637':'ONNUT','gneJ42':'VIRTUAL'}
        self.traffic_signals = {ts: TrafficSignal(self,
                                                  ts,
                                                  self.delta_time,
                                                  self.yellow_time,
                                                  self.min_green,
                                                  self.max_green[ts],
                                                  self.begin_time,
                                                  self.ts_junction[ts]) for ts in self.ts_ids}

        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}
        self.teleport_numbers = 0
        self.reward_range = (-float('inf'), float('inf'))
        self.start = run
        self.run = run
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.name=name
        self.writer = SummaryWriter('tensorBoard/{}'.format(name))

        traci.close()

    def save_score(self):
        self.save_csv(self.out_csv_name, self.run)
        self.run += 1

    def save_score_max_green(self):
        self.save_csv_max_green(self.out_csv_name, self.run, self.max_green_onnut, self.max_green_virtual)
        self.run += 1

    def reset(self, random_seed):
        if self.run > self.start:
            traci.close()
            # self.save_csv(self.out_csv_name, self.run)
        # self.run += 1
        self.metrics = []

        traci.start([self._sumo_binary,
                     '-n', self._net,
                     '-c', "onnut_ake.sumocfg",
                     '--time-to-teleport', str(self.time_to_teleport),
                     '--start', 'true',
                     '--quit-on-end','true',
                     "--no-internal-links",'true',
                     "--ignore-junction-blocker",'-1',
                     '--no-warnings', 'true',
                     '--seed', str(random_seed),
                     ])

        self.traffic_signals = {ts: TrafficSignal(self,
                                                  ts,
                                                  self.delta_time,
                                                  self.yellow_time,
                                                  self.min_green,
                                                  self.max_green[ts],
                                                  self.begin_time,
                                                  self.ts_junction[ts]) for ts in self.ts_ids}


        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    # def reset(self):
    #     if self.run > self.start:
    #         traci.close()
    #         # self.save_csv(self.out_csv_name, self.run)
    #     # self.run += 1
    #     self.metrics = []
    #     random_seed = random.seed(self.run)

    #     traci.start([self._sumo_binary,
    #                     '-n', self._net,
    #                     '-c', "onnut_ake.sumocfg",
    #                     '--time-to-teleport', str(self.time_to_teleport),
    #                     '--start', 'true',
    #                     '--quit-on-end','true',
    #                     "--no-internal-links",'true',
    #                     "--ignore-junction-blocker",'-1',
    #                     '--no-warnings', 'true',
    #                     '--seed', str(random_seed),
    #                     ])

    #     self.traffic_signals = {ts: TrafficSignal(self,
    #                                                 ts,
    #                                                 self.delta_time,
    #                                                 self.yellow_time,
    #                                                 self.min_green,
    #                                                 self.max_green[ts],
    #                                                 self.begin_time,
    #                                                 self.ts_junction[ts]) for ts in self.ts_ids}


    #     if self.single_agent:
    #         return self._compute_observations()[self.ts_ids[0]]
    #     else:
    #         return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()

    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
                if self.sim_step % 15 == 0:
                    info = self._compute_step_info()
                    self.metrics.append(info)
        else:
            self._apply_actions(action)

            time_to_act = False

            #---- Reset reward for new timestep ----#
            self.rewards[self.ts_ids[0]] = 0
            self.rewards[self.ts_ids[1]] = 0

            self.teleport_numbers = 0

            # i = 0
            while not time_to_act:
                self._sumo_step()

                rewards = self._compute_rewards()
                for k, v in rewards.items():
                    temp = self.rewards.get(k)
                    if temp == None:
                        temp = 0
                    self.rewards[k] = temp+v

                teleport_number = self._compute_teleports()
                if teleport_number == None :
                    teleport_number = 0
                self.teleport_numbers += teleport_number

                # for k, v in teleport_number.items():
                #     temp = self.teleport_numbers.get(k)
                #     if temp == None:
                #         temp = 0
                #     self.teleport_numbers[k] = temp+v

                for ts in self.ts_ids:
                    self.traffic_signals[ts].update()
                    if self.traffic_signals[ts].time_to_act:
                        time_to_act = True

                if self.sim_step % 15 == 0:
                    info = self._compute_step_info()
                    self.metrics.append(info)

        observations = self._compute_observations()
        # rewards = self._compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time}
        done.update({ts_id: False for ts_id in self.ts_ids})

        if self.single_agent:
            return observations[self.ts_ids[0]], self.rewards[self.ts_ids[0]], done['__all__'], {}
        else:
            return observations, self.rewards, done, {}

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """   
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                self.traffic_signals[ts].set_next_phase(action)
    
    def _compute_observations(self):
        self.observations.update({ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        # return {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        return {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids}

    def _compute_teleports(self):
        # return {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        return self.traffic_signals[self.ts_ids[0]].compute_teleport()
        # {ts: self.traffic_signals[ts].compute_teleport() for ts in self.ts_ids}

    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space
    
    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space
    
    def observation_spaces(self, ts_id):
        return self.traffic_signals[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        traci.simulationStep()

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'onnut_action': self.traffic_signals[self.ts_ids[0]].current_phase,
            'virtual_action': self.traffic_signals[self.ts_ids[1]].current_phase,        
            # 'reward_onnut': sum(self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act),
            'reward_onnut' : self.rewards[self.ts_ids[0]],
            'reward_virtual' : self.rewards[self.ts_ids[1]],
            'total_travel_time_onnut' : self.traffic_signals[self.ts_ids[0]].get_travel_time(),
            'total_travel_time_virtual' : self.traffic_signals[self.ts_ids[1]].get_travel_time(),
            # 'total_travel_time_all' : self.traffic_signals[self.ts_ids[0]].get_travel_time_all()
            'teleport_number' : self.teleport_numbers #doesn't care about teleport number
        }

    def close(self):
        traci.close()
        self.writer.close()

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            self.displayTensor(df,run)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)
    
    def displayTensor(self,df,run):
        self.writer.add_scalar('Reward: Onnut', df.reward_onnut.sum(), run) #number of vehicles
        self.writer.add_scalar('Reward: Virtual', df.reward_virtual.sum(), run)

        self.writer.add_scalar('Travel time: Onnut', df.total_travel_time_onnut.mean(), run)
        self.writer.add_scalar('Travel time: Virtual', df.total_travel_time_virtual.mean(), run)
        
    def save_csv_max_green(self, out_csv_name, run, max_green_onnut, max_green_virtual):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_onnut{}'.format(max_green_onnut) +'_virtual{}'.format(max_green_virtual)+ '_run{}'.format(run) + '.csv', index=False)

    def getTime(self,time):
        time=time%(24*3600)
        hours=time//3600
        time%=3600
        minutes=time//60
        time%=60
        seconds=time
        periods=[('hours',int(hours)),('minutes',int(minutes)),('seconds',int(seconds))]
        time_string=':'.join('{}'.format(value) for name,value in periods)
        return time_string



    
