import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
from gym import spaces
import gym
gym.logger.set_level(40)

class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green, begin_time, junction):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.current_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None

        self.phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0].phases
        self.num_phases = len(self.phases)  # Number of green phases

        #============ Indicate junction =========#
        self.junction = junction  #Using for difference in junction
        #========================================#


        #self.observation_space = spaces.Box(low=np.zeros(self.num_phases + 1 + self.get_observation_places()), high=np.ones(self.num_phases + 1 +self.get_observation_places()), dtype=np.float32)

        if self.junction == "ONNUT" :
            self.observation_space = spaces.Box(low=np.zeros(self.num_phases + 1 + 18), high=np.ones(self.num_phases + 1 + 18), dtype=np.float32)
        elif self.junction == "VIRTUAL" :
            self.observation_space = spaces.Box(low=np.zeros(self.num_phases + 1 + 15), high=np.ones(self.num_phases + 1 + 15), dtype=np.float32)    

        print('Observation space of ', junction,'is :', self.observation_space)
        # print('>>>>>>>>>>>>>>>>>>>>>>>>')
        self.action_space = spaces.Discrete(self.num_phases)
        print('Action space of ', junction,'is :', self.action_space)

        if self.junction == 'ONNUT' :
            #SB,WB,NB
            # ONNUT_UPSTREAM_DETECTOR_ID = 4,5,6,9,11,12
            # ONNUT_DOWNSTREAM_DETECTOR_ID = 3,8,10

            #Dict with key: indicate up/downstream , value: list of detectorID

            self.SB_detectorID_dict = {
            'UPSTREAM' :   [ "S_ONT_04_0","S_ONT_04_1","S_ONT_04_2",
                             "S_ONT_09_0","S_ONT_09_1","S_ONT_09_2"] ,

            'DOWNSTREAM' : [ "S_ONT_03_0","S_ONT_03_1","S_ONT_03_2",
                             "S_ONT_07_0","S_ONT_07_1","S_ONT_07_2"]
                                    }

            self.WB_detectorID_dict = {
            'UPSTREAM' : ["S_ONT_05_0","S_ONT_05_1"],
            'DOWNSTREAM' : ["S_ONT_18_0","S_ONT_18_1"]
                                    }

            self.NB_detectorID_dict = {
            'UPSTREAM' : [
                    "S_ONT_06_0","S_ONT_06_1","S_ONT_06_2","S_ONT_06_3",
                    "S_ONT_11_0","S_ONT_11_1","S_ONT_11_2","S_ONT_11_3",
                    "S_ONT_12_0","S_ONT_12_1","S_ONT_12_2","S_ONT_12_3"
                                        ],
            'DOWNSTREAM' : [
                    "S_ONT_08_0",
                    "S_ONT_10_0","S_ONT_10_1","S_ONT_10_2"
                                            ]
                                    }

            self.loopID = [
                'Induction_Loop_1','Induction_Loop_2','Induction_Loop_3',
                'Induction_Loop_4','Induction_Loop_5','Induction_Loop_6',
                'Induction_Loop_7','Induction_Loop_8','Induction_Loop_9','Induction_Loop_10' 
                        ]

            self.edgeID_for_MOE = [
                #NB
                '824116560#0','824116560#1','824116560#2','824116560#3','824816455','220429932#0','824116561-AddedOffRampEdge','113135465#5',
                '750035412#1-AddedOffRampEdge','824816456','113135465#0','113135465#2',
                #SB
                '751454884#3','751454884#2','751454884#0',
                '751454885#0','751454885#2','751454885#3','751454885#5',
                #WB
                '824456410#0','824456410#2','824456410#2.41',
                '156591171#2','-824456410#2','156591171#0'
                                ]

            # self.edgeID_all = traci.edge.getIDList()
                
        elif self.junction == "VIRTUAL" :

            # SB, WB, BigC
            #Dict with key: indicate up/downstream , value: list of detectorID
            #VIRTUAL_UPSTREAM_DETECTOR_ID = 2,13,14,17
            #VIRTUAL_DOWNSTREAM_DETECTOR_ID = 1,15,16

            self.SB_detectorID_dict = {
            'UPSTREAM' : [
                    "S_ONT_13_0","S_ONT_13_1",
                    "S_ONT_14_0","S_ONT_14_1",
                                            ],
            'DOWNSTREAM' : [
                    "S_ONT_15_0","S_ONT_15_1",
                    "S_ONT_16_0","S_ONT_16_1"
                                            ]
                                        }
            self.WB_detectorID_dict = {
            'UPSTREAM' : [
                    "S_ONT_02_0","S_ONT_02_1",
                    "S_ONT_13_0","S_ONT_13_1",
                    "S_ONT_14_0","S_ONT_14_1",
                    "S_ONT_17_0"
                                            ],
            'DOWNSTREAM' : [
                    "S_ONT_01_0","S_ONT_01_1",
                    "S_ONT_15_0","S_ONT_15_1",
                    "S_ONT_16_0","S_ONT_16_1"
                                            ]
                                        }
            self.BIGC_detectorID_dict = {
            'UPSTREAM' : ["S_ONT_17_0"],
            'DOWNSTREAM' : []
                                        }

            self.loopID = [
                "Virtual_loop_1","Virtual_loop_2","Virtual_loop_3",
                "Virtual_loop_4","Virtual_loop_5"]
            
            self.edgeID_for_MOE = [
                #SB
                '824456410#2','824456410#2.41','824456409#0','-gneE25',
                '-824456410#2','156591171#0','-824456409#0','-gneE24',
                #WB
                'gneE34','824456409#5','824456409#6','113135397#1','113135397#3',
                'gneE33','-824456409#5','-113135397#0','-113135397#2','-113135397#4',
                #BigC
                'gneE32','gneE29'
                                ]
            # self.edgeID_all = []

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    @property
    def time_to_act(self):
        # print(self.next_action_time , self.env.sim_step,self.next_action_time == self.env.sim_step)
        return self.next_action_time == self.env.sim_step

    def update(self):
        self.time_since_last_phase_change += 1

    def set_next_phase(self, new_phase):
        self.new_phase = new_phase
        # print('*************************************************')
        # print('tls id : ', self.id)
        # print('current phase : ', self.phase)
        # print('new phase :', self.new_phase)
        # print('current sim time : ',self.env.sim_step)
        # print('self.time_since_last_phase_change',self.time_since_last_phase_change)

        if self.phase == self.new_phase and self.time_since_last_phase_change < self.min_green:
            # print('current phase and agent\'s action phase are equal but duration of current phase is less than min green time')
            self.next_action_time = self.env.sim_step +  self.delta_time

        elif self.phase == self.new_phase and self.time_since_last_phase_change >= self.min_green:
            # print('current phase and agent\'s action phase are equal, and still less than max green time')
            self.next_action_time = self.env.sim_step + self.delta_time

        elif self.phase == self.new_phase and self.time_since_last_phase_change > self.max_green:
            # print('duration of current phase is over max_green now')
            if self.new_phase +1 >=self.num_phases:
                self.current_phase = 0
            else:
                self.current_phase = self.new_phase + 1

            traci.trafficlight.setPhase(self.id, self.current_phase)
            self.next_action_time = self.env.sim_step + self.delta_time
            self.time_since_last_phase_change = 0

        elif self.phase != self.new_phase and self.time_since_last_phase_change < self.min_green:
            # print('current phase and agent\'s action phase are not equal but duration of current phase is less than min green time')
            self.next_action_time = self.env.sim_step +  self.delta_time

        elif self.phase != self.new_phase and self.time_since_last_phase_change >= self.min_green:
            # print('current phase and agent\'s action phase are not equal, and duration is greater than min green time')
            self.current_phase = self.new_phase
            traci.trafficlight.setPhase(self.id, self.current_phase)
            self.next_action_time = self.env.sim_step + self.delta_time
            self.time_since_last_phase_change = 0
        #######################################################

        # print(self.next_action_time)


    def compute_observation(self):
        phase_id = [1 if self.current_phase == i else 0 for i in range(self.num_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green else 1]
        UPSTREAM_OBS = []
        DOWNSTREAM_OBS = []
        if self.junction == "ONNUT" :
            occu_SB_UP,occu_WB_UP,occu_NB_UP = self.get_occupancy_average_percent(indicate="UPSTREAM")
            flow_SB_UP,flow_WB_UP,flow_NB_UP = self.get_flow_sum(indicate="UPSTREAM")
            unjam_SB_UP,unjam_WB_UP,unjam_NB_UP = self.get_flow_sum(indicate="UPSTREAM")

            UPSTREAM_OBS = [occu_SB_UP,flow_SB_UP,unjam_SB_UP,
                            occu_WB_UP,flow_WB_UP,unjam_WB_UP,
                            occu_NB_UP,flow_NB_UP,unjam_NB_UP]

            occu_SB_DOWN,occu_WB_DOWN,occu_NB_DOWN = self.get_occupancy_average_percent(indicate="DOWNSTREAM")
            flow_SB_DOWN,flow_WB_DOWN,flow_NB_DOWN = self.get_flow_sum(indicate="DOWNSTREAM")
            unjam_SB_DOWN,unjam_WB_DOWN,unjam_NB_DOWN = self.get_flow_sum(indicate="DOWNSTREAM")

            DOWNSTREAM_OBS = [occu_SB_DOWN,flow_SB_DOWN,unjam_SB_DOWN,
                              occu_WB_DOWN,flow_WB_DOWN,unjam_WB_DOWN,
                              occu_NB_DOWN,flow_NB_DOWN,unjam_NB_DOWN]

        elif self.junction == "VIRTUAL" :
            occu_SB_UP,occu_WB_UP,occu_BIGC_UP = self.get_occupancy_average_percent(indicate="UPSTREAM")
            flow_SB_UP,flow_WB_UP,flow_BIGC_UP = self.get_flow_sum(indicate="UPSTREAM")
            unjam_SB_UP,unjam_WB_UP,unjam_BIGC_UP = self.get_flow_sum(indicate="UPSTREAM")

            UPSTREAM_OBS = [occu_SB_UP,flow_SB_UP,unjam_SB_UP,
                            occu_WB_UP,flow_WB_UP,unjam_WB_UP,
                            occu_BIGC_UP,flow_BIGC_UP,unjam_BIGC_UP]

            occu_SB_DOWN,occu_WB_DOWN,occu_BIGC_DOWN = self.get_occupancy_average_percent(indicate="DOWNSTREAM")
            flow_SB_DOWN,flow_WB_DOWN,flow_BIGC_DOWN = self.get_flow_sum(indicate="DOWNSTREAM")
            unjam_SB_DOWN,unjam_WB_DOWN,unjam_BIGC_DOWN = self.get_flow_sum(indicate="DOWNSTREAM")

            DOWNSTREAM_OBS = [occu_SB_DOWN,flow_SB_DOWN,unjam_SB_DOWN,
                            occu_WB_DOWN,flow_WB_DOWN,unjam_WB_DOWN]   


        observation = np.array(phase_id + min_green + UPSTREAM_OBS + DOWNSTREAM_OBS , dtype=np.float32)
        # print(len(observation))
        # print('------------------------------------------------------------------')
        return observation
            
    def compute_reward(self):
        self.last_reward = self._throughput_reward()
        return self.last_reward

    def compute_teleport(self):
        self.last_teleport = traci.simulation.getEndingTeleportNumber()
        return self.last_teleport


    def _throughput_reward(self):
        ####################### detectors for each intersection ######################################################
        ONNUT_loopcoil = ['Induction_Loop_1','Induction_Loop_2','Induction_Loop_3',
                          'Induction_Loop_4','Induction_Loop_5','Induction_Loop_6',
                          'Induction_Loop_7','Induction_Loop_8','Induction_Loop_9','Induction_Loop_10']

        VIRTUAL_loopcoil = [
            "Virtual_loop_1","Virtual_loop_2","Virtual_loop_3",
            "Virtual_loop_4","Virtual_loop_5"]

        if self.junction == 'VIRTUAL':
            self.throughput = self.get_throughput(VIRTUAL_loopcoil)

        elif self.junction == 'ONNUT':
            self.throughput = self.get_throughput(ONNUT_loopcoil)

        return self.throughput

    def get_throughput(self,loopcoilIDs):
        throughput = 0
        for id in loopcoilIDs:

            laneID = traci.inductionloop.getLaneID(id)
            edgeID = traci.lane.getEdgeID(laneID)
            speed = traci.edge.getLastStepMeanSpeed(edgeID)
            if speed > 0:
                throughput += traci.inductionloop.getLastStepVehicleNumber(id)

        # # throughput = sum([traci.inductionloop.getLastStepVehicleNumber(i) for i in loopcoilIDs if traci.inductionloop.getLastStepMeanSpeed(i) > 0])
        return throughput
    ########################################################################
    #
    # getting attention places for each intersection
    #
    def get_observation_places(self):

        Jvirtual = {
            'SB_UPSTREAM' : [
                            "S_ONT_13_0","S_ONT_13_1",
                            "S_ONT_14_0","S_ONT_14_1",
                            ],
            'SB_DOWNSTREAM' : [
                            "S_ONT_15_0","S_ONT_15_1",
                            "S_ONT_16_0","S_ONT_16_1"
                            ],
            'WB_UPSTREAM' : [
                "S_ONT_02_0","S_ONT_02_1",
                "S_ONT_13_0","S_ONT_13_1",
                "S_ONT_14_0","S_ONT_14_1",
                "S_ONT_17_0"
            ],
            'WB_DOWNSTREAM' : [
                "S_ONT_01_0","S_ONT_01_1",
                "S_ONT_15_0","S_ONT_15_1",
                "S_ONT_16_0","S_ONT_16_1"
            ],

            'BIGC_UPSTREAM' : ["S_ONT_17_0"],
            # 'BIGC_DOWNSTREAM' : []
        }

        JOnnut = {
                'SB_UPSTREAM' : [
                    "S_ONT_04_0","S_ONT_04_1","S_ONT_04_2",
                    "S_ONT_09_0","S_ONT_09_1","S_ONT_09_2"
                ],
                'SB_DOWNSTREAM' : ["S_ONT_03_0","S_ONT_03_1","S_ONT_03_2",
                                   "S_ONT_07_0","S_ONT_07_1","S_ONT_07_2"],

                'WB_UPSTREAM' : ["S_ONT_05_0","S_ONT_05_1"],
                'WB_DOWNSTREAM' : ["S_ONT_18_0","S_ONT_18_1"],

                'NB_UPSTREAM' : [
                    "S_ONT_06_0","S_ONT_06_1","S_ONT_06_2","S_ONT_06_3",
                    "S_ONT_11_0","S_ONT_11_1","S_ONT_11_2","S_ONT_11_3",
                    "S_ONT_12_0","S_ONT_12_1","S_ONT_12_2","S_ONT_12_3"
                                ],
                'NB_DOWNSTREAM' : [
                    "S_ONT_08_0",
                    "S_ONT_10_0","S_ONT_10_1","S_ONT_10_2"
                ]
                }

        MAP = [Jvirtual, JOnnut]

        state_places = None
        if self.junction == 'ONNUT':
            state_places = MAP[0]
        elif self.junction == 'VIRTUAL':
            state_places = MAP[1]

        return len(state_places)

    def get_flow_sum(self,indicate):
        #     Speed (metres per sec) = flow (vehicle per sec) / density (veh per metre), Ajarn chaodit
        #         flow= int(densityPerLane) * float(meanSpeed)#flow per lane
        #     print('LastStepVehicleNumber', sum([traci.lanearea.getLastStepVehicleNumber(e) for e in detector_id]))
        #     print('length', sum([traci.lanearea.getLength(i) for i in detector_id]))
        #     density = sum([traci.lanearea.getLastStepVehicleNumber(e) for e in detector_id])/\
        #     sum([traci.lanearea.getLength(i) for i in detector_id])
        #     print('density', density)
        flow_SB = 0
        flow_WB = 0
        flow_OtherBound = 0
        if indicate == "UPSTREAM" :
            flow_SB = sum(([traci.lanearea.getLastStepVehicleNumber(e)*traci.lanearea.getLastStepMeanSpeed(e)/traci.lanearea.getLength(e)
                            for e in self.SB_detectorID_dict['UPSTREAM'] if traci.lanearea.getLastStepMeanSpeed(e) >= 0]))
            flow_WB = sum(([traci.lanearea.getLastStepVehicleNumber(e)*traci.lanearea.getLastStepMeanSpeed(e)/traci.lanearea.getLength(e)
                            for e in self.WB_detectorID_dict['UPSTREAM'] if traci.lanearea.getLastStepMeanSpeed(e) >= 0]))

            if self.junction == "ONNUT" :
                #OtherBound = NB in ONNUT
                flow_OtherBound = sum(([traci.lanearea.getLastStepVehicleNumber(e)*traci.lanearea.getLastStepMeanSpeed(e)/traci.lanearea.getLength(e)
                                        for e in self.NB_detectorID_dict['UPSTREAM'] if traci.lanearea.getLastStepMeanSpeed(e) >= 0]))

            elif self.junction == "VIRTUAL" :
                #OtherBound = BIGC in onnut
                flow_OtherBound = sum(([traci.lanearea.getLastStepVehicleNumber(e)*traci.lanearea.getLastStepMeanSpeed(e)/traci.lanearea.getLength(e)
                                        for e in self.BIGC_detectorID_dict['UPSTREAM'] if traci.lanearea.getLastStepMeanSpeed(e) >= 0]))

        elif indicate == "DOWNSTREAM" :

            flow_SB = sum(([traci.lanearea.getLastStepVehicleNumber(e)*traci.lanearea.getLastStepMeanSpeed(e)/traci.lanearea.getLength(e)
                            for e in self.SB_detectorID_dict["DOWNSTREAM"] if traci.lanearea.getLastStepMeanSpeed(e) >= 0]))
            flow_WB = sum(([traci.lanearea.getLastStepVehicleNumber(e)*traci.lanearea.getLastStepMeanSpeed(e)/traci.lanearea.getLength(e)
                            for e in self.WB_detectorID_dict["DOWNSTREAM"] if traci.lanearea.getLastStepMeanSpeed(e) >= 0]))

            if self.junction == "ONNUT" :
                #OtherBound = NB in ONNUT
                flow_OtherBound = sum(([traci.lanearea.getLastStepVehicleNumber(e)*traci.lanearea.getLastStepMeanSpeed(e)/traci.lanearea.getLength(e)
                                        for e in self.NB_detectorID_dict["DOWNSTREAM"] if traci.lanearea.getLastStepMeanSpeed(e) >= 0]))

            elif self.junction == "VIRTUAL" :
                #OtherBound = BIGC in onnut
                flow_OtherBound = sum(([traci.lanearea.getLastStepVehicleNumber(e)*traci.lanearea.getLastStepMeanSpeed(e)/traci.lanearea.getLength(e)
                                        for e in self.BIGC_detectorID_dict["DOWNSTREAM"] if traci.lanearea.getLastStepMeanSpeed(e) >= 0]))

        return flow_SB, flow_WB, flow_OtherBound #OtherBound refer to different bound in onnut - virtual


    def get_unjamlength_meters(self,indicate):
        unjam_SB = 0
        unjam_WB = 0
        unjam_OtherBound = 0
        if indicate == "UPSTREAM" :
            detector_length = sum(traci.lanearea.getLength(d) for d in self.SB_detectorID_dict["UPSTREAM"])
            unjam_SB = detector_length - (sum([traci.lanearea.getJamLengthMeters(e) for e in self.SB_detectorID_dict["UPSTREAM"]])) #/detector_length

            detector_length = sum(traci.lanearea.getLength(d) for d in self.WB_detectorID_dict["UPSTREAM"])
            unjam_WB = detector_length - (sum([traci.lanearea.getJamLengthMeters(e) for e in self.WB_detectorID_dict["UPSTREAM"]])) #/detector_length

            if self.id == "ONNUT" :
                #OtherBound = NB in ONNUT
                detector_length = sum(traci.lanearea.getLength(d) for d in self.NB_detectorID_dict["UPSTREAM"])
                unjam_OtherBound = detector_length - (sum([traci.lanearea.getJamLengthMeters(e) for e in self.NB_detectorID_dict["UPSTREAM"]])) #/detector_length

            elif self.id == "VIRTUAL" :
                #OtherBound = BIGC in onnut
                detector_length = sum(traci.lanearea.getLength(d) for d in self.BIGC_detectorID_dict["UPSTREAM"])
                unjam_OtherBound = detector_length - (sum([traci.lanearea.getJamLengthMeters(e) for e in self.BIGC_detectorID_dict["UPSTREAM"]])) #/detector_length

        elif indicate == "DOWNSTREAM" :

            detector_length = sum(traci.lanearea.getLength(d) for d in self.SB_detectorID_dict["DOWNSTREAM"])
            unjam_SB = detector_length - (sum([traci.lanearea.getJamLengthMeters(e) for e in self.SB_detectorID_dict["DOWNSTREAM"]])) #/detector_length

            detector_length = sum(traci.lanearea.getLength(d) for d in self.WB_detectorID_dict["DOWNSTREAM"])
            unjam_WB = detector_length - (sum([traci.lanearea.getJamLengthMeters(e) for e in self.WB_detectorID_dict["DOWNSTREAM"]])) #/detector_length

            if self.id == "ONNUT" :
                #OtherBound = NB in ONNUT
                detector_length = sum(traci.lanearea.getLength(d) for d in self.NB_detectorID_dict["DOWNSTREAM"])
                unjam_OtherBound = detector_length - (sum([traci.lanearea.getJamLengthMeters(e) for e in self.NB_detectorID_dict["DOWNSTREAM"]])) #/detector_length

            elif self.id == "VIRTUAL" :
                #OtherBound = BIGC in onnut
                detector_length = sum(traci.lanearea.getLength(d) for d in self.BIGC_detectorID_dict["DOWNSTREAM"])
                unjam_OtherBound = detector_length - (sum([traci.lanearea.getJamLengthMeters(e) for e in self.BIGC_detectorID_dict["DOWNSTREAM"]])) #/detector_length

        return unjam_SB, unjam_WB, unjam_OtherBound #OtherBound refer to different bound in onnut - virtual

    def get_occupancy_average_percent(self,indicate):
        #get occupancy average for all detector in detector_id and scale by (Vehicle Length + MinimumGap)/MinimumGap
        #Vehicle Length = 4.62 MinimumGap = 2.37
        occu_SB = 0
        occu_WB = 0
        occu_OtherBound = 0
        if indicate == "UPSTREAM" :
            occu_SB = (sum([traci.lanearea.getLastStepOccupancy(e) for e in self.SB_detectorID_dict['UPSTREAM']])
                       /len(self.SB_detectorID_dict["UPSTREAM"]))*((4.62+2.37)/4.62)
            occu_WB = (sum([traci.lanearea.getLastStepOccupancy(e) for e in self.WB_detectorID_dict['UPSTREAM']])
                       /len(self.WB_detectorID_dict["UPSTREAM"]))*((4.62+2.37)/4.62)
            if self.junction == "ONNUT" :
                #OtherBound = NB in ONNUT
                occu_OtherBound = (sum([traci.lanearea.getLastStepOccupancy(e) for e in self.NB_detectorID_dict['UPSTREAM']])
                                   /len(self.NB_detectorID_dict["UPSTREAM"]))*((4.62+2.37)/4.62)

            elif self.junction == "VIRTUAL" :
                #OtherBound = BIGC in onnut
                occu_OtherBound = (sum([traci.lanearea.getLastStepOccupancy(e) for e in self.BIGC_detectorID_dict['UPSTREAM']])
                                   /len(self.BIGC_detectorID_dict["UPSTREAM"]))*((4.62+2.37)/4.62)

        elif indicate == "DOWNSTREAM" :

            occu_SB = (sum([traci.lanearea.getLastStepOccupancy(e) for e in self.SB_detectorID_dict["DOWNSTREAM"]])
                       /len(self.SB_detectorID_dict["UPSTREAM"]))*((4.62+2.37)/4.62)

            if len(self.WB_detectorID_dict["DOWNSTREAM"]) == 0 :
                occu_WB = 0
            else :
                occu_WB = (sum([traci.lanearea.getLastStepOccupancy(e) for e in self.WB_detectorID_dict["DOWNSTREAM"]])
                           /len(self.WB_detectorID_dict["DOWNSTREAM"]))*((4.62+2.37)/4.62)
            if self.junction == "ONNUT" :
                #OtherBound = NB in ONNUT
                occu_OtherBound = (sum([traci.lanearea.getLastStepOccupancy(e) for e in self.NB_detectorID_dict["DOWNSTREAM"]])
                                   /len(self.NB_detectorID_dict["DOWNSTREAM"]))*((4.62+2.37)/4.62)

            elif self.junction == "VIRTUAL" :
                #OtherBound = BIGC in onnut
                if len(self.BIGC_detectorID_dict["DOWNSTREAM"]) == 0 :
                    occu_OtherBound = 0
                else :
                    occu_OtherBound = (sum([traci.lanearea.getLastStepOccupancy(e) for e in self.BIGC_detectorID_dict["DOWNSTREAM"]])
                                       /len(self.BIGC_detectorID_dict["DOWNSTREAM"]))*((4.62+2.37)/4.62)

        return occu_SB, occu_WB, occu_OtherBound #OtherBound refer to different bound in onnut - virtual

    def get_travel_time(self) :
        # sum(traci.edge.getTraveltime(edgeID) for edgeID in self.edgeID_for_MOE)/len(self.edgeID_for_MOE)
        if ((sum(traci.edge.getLastStepVehicleNumber(edgeID) for edgeID in self.edgeID_for_MOE)) != 0) : 
            travel_time = sum(traci.edge.getTraveltime(edgeID)*traci.edge.getLastStepVehicleNumber(edgeID) for edgeID in self.edgeID_for_MOE) \
                /(sum(traci.edge.getLastStepVehicleNumber(edgeID) for edgeID in self.edgeID_for_MOE) *len(self.edgeID_for_MOE))
        else : 
            travel_time = 0
        return travel_time
    
    # def get_travel_time_all(self) :
    #     if ((sum(traci.edge.getLastStepVehicleNumber(edgeID) for edgeID in self.edgeID_all)) != 0) :
    #         travel_time_all = sum(traci.edge.getTraveltime(edgeID)*traci.edge.getLastStepVehicleNumber(edgeID) for edgeID in self.edgeID_all) \
    #             /(sum(traci.edge.getLastStepVehicleNumber(edgeID) for edgeID in self.edgeID_all) *len(self.edgeID_all))
    #     else :
    #         travel_time_all = 0
    #     return travel_time_all