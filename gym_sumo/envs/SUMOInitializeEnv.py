from .sumo_env import SUMOEnv
import os
import traci

class SUMOEnv_Initializer(SUMOEnv):
	def __init__(self,mode='gui'):
		super(SUMOEnv_Initializer, self).__init__(mode=mode, simulation_end=3600)
