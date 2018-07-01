from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
import traci
import traci.constants as tc
from scipy.misc import imread
from gym import spaces
from string import Template
import numpy as np
import math
import time
from cv2 import imread,imshow,resize
import cv2

import os, sys, subprocess

STATE_W = 128
STATE_H = 128


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	def __init__(self,mode='gui',simulation_end=3600):

		self.simulation_end = simulation_end
		self.mode = mode
		self._seed()
		self.traci = self.initSimulator(True,8870)
		self.sumo_step = 0
		self.collisions =0
		self.episode = 0
		self.flag = True

		self.action_space = spaces.Box(low=np.array([-1]), high= np.array([+1])) # acceleration
		self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 1), dtype=np.uint8)
		self.sumo_running = False
		self.viewer = None	
		#self.dt = self.traci.simulation.getDeltaT()/1000.0

		## INITIALIZE EGO CAR
		self.egoCarID = 'veh0'
		self.speed = 0
		self.max_speed = 20.1168		# m/s 
		self.observation = self._reset()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self,accel):
		r = 0
		self.sumo_step +=1
		self.takeAction(accel)
		self.traci.simulationStep()

		# Get reward and check for terminal state
		reward, terminal, terminalType = self._reward()
		r += reward

		braking = self.isTrafficBraking()
		# if egoCar.isTrafficWaiting(): waitingTime += 1

		self.observation = self._observation()

		info = {braking, terminalType}

		if(self.episode%5==0  and self.flag):
			self.flag = False
			print("Collision Rate : {0} ".format(self.collisions/5))
			self.collisions = 0

		return self.observation, reward, terminal, {}

	def _reward(self):
		terminal = False
		terminalType = 'None'

		try :
			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))
			distance_ego = (np.asarray([np.linalg.norm(position_ego - self.endPos)]))[0]
		
		except:
			print("self.traci couldn't find car")
			return -1.0, True, 'Car not found'
			distance_ego = 0

		# Step cost
		reward = -0.1 

		#Cooperation Reward
		traffic_waiting = self.isTrafficWaiting()
		traffic_braking = self.isTrafficBraking()

		if(traffic_waiting and traffic_braking):
			reward += -0.05
		elif(traffic_braking or traffic_waiting):
			reward += -0.025
		elif(not traffic_waiting and not traffic_braking):
			reward += + 0.05

		# Collision check
		teleportIDList = self.traci.simulation.getStartingTeleportIDList()
		if teleportIDList:
			collision = True
			self.collisions +=1
			reward += -1.0 
			terminal = True
			terminalType = 'Collided!!!'
			self.episode += 1
			if(self.episode%5 == 0):
				self.flag = True

			#print("Reward for this episode is :{0}".format(reward))

		else: # Goal check
			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))
			#print(position_ego)
			distance_ego = np.linalg.norm(position_ego - self.endPos)
			if position_ego[0] <= self.endPos[0]:
				reward += 1.0 
				terminal = True
				terminalType = 'Survived'
				self.episode +=1
				if(self.episode%5 == 0):
					self.flag = True
		

		return reward, terminal, terminalType


	def _observation(self):
		if self.mode == "gui":
			self.traci.gui.screenshot(self.traci.gui.DEFAULT_VIEW,os.path.join(os.path.dirname(os.path.realpath(__file__)),'sumo.png'))

		image = imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),'sumo.png'),0)
		image = resize(image,(STATE_W,STATE_H))
		image = image.reshape((STATE_W,STATE_H,1))
		#imshow('Display window',image)


		return image

	def _reset(self):
		try:
			self.traci.vehicle.remove(self.egoCarID)
		except:
			pass

		self.addEgoCar()            # Add the ego car to the scene
		self.setGoalPosition()      # Set the goal position
		self.speed = 0
		self.traci.simulationStep() 		# Take a simulation step to initialize car
		
		self.observation = self._observation()
		return self.observation


	def _render(self, mode='gui', close=False):

		if self.mode == "gui":
			img = imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),'sumo.png'), 1)
			if mode == 'rgb_array':
				return img
			elif mode == 'human':
				from gym.envs.classic_control import rendering
				if self.viewer is None:
					self.viewer = rendering.SimpleImageViewer()
				self.viewer.imshow(img)
		else:
			raise NotImplementedError("Only rendering in GUI mode is supported")


	def initSimulator(self,withGUI,portnum):
		# Path to the sumo binary
		if withGUI:
			sumoBinary = "/usr/bin/sumo-gui"
		else:
			sumoBinary = "/usr/bin/sumo"

		sumoConfig = "/home/salonidash7/gym-sumo/gym_sumo/envs/sumo_configs/0221.sumocfg"

		# Call the sumo simulator
		sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(portnum), \
			"--time-to-teleport", str(-1), "--collision.check-junctions", str(True), \
			"--no-step-log", str(True), "--no-warnings", str(True)], stdout=sys.stdout, stderr=sys.stderr)

		# Initialize the simulation
		traci.init(portnum)
		return traci

	def closeSimulator(traci):
		traci.close()
		sys.stdout.flush()
	
	def setGoalPosition(self):
		self.endPos= [79.0, 113.65]

		
	def addEgoCar(self):																

		vehicles=self.traci.vehicle.getIDList()

		## PRUNE IF TRAFFIC HAS BUILT UP TOO MUCH
		# if more cars than setnum, p(keep) = setnum/total
		setnum = 20
		if len(vehicles)>0:
			keep_frac = float(setnum)/len(vehicles)
		for i in range(len(vehicles)):
			if vehicles[i] != self.egoCarID:
				if np.random.uniform(0,1,1)>keep_frac:
					self.traci.vehicle.remove(vehicles[i])

		## DELAY ALLOWS CARS TO DISTRIBUTE 
		for j in range(np.random.randint(40,50)):#np.random.randint(0,10)):
			self.traci.simulationStep()

		## STARTING LOCATION
		# depart = -1   (immediate departure time)
		# pos    = -2   (random position)
		# speed  = -2   (random speed)
		
		self.traci.vehicle.addFull(self.egoCarID, 'routeEgo', depart=None, departPos='84.0', departSpeed='0', departLane='0', typeID='vType0')
	

		self.traci.vehicle.setSpeedMode(self.egoCarID, int('00000',2))


	def isTrafficBraking(self):
		""" Check if any car is braking
		"""
		for carID in self.traci.vehicle.getIDList():
			if carID != self.egoCarID:
				brakingState = self.traci.vehicle.getSignals(carID)
				if brakingState == 8:
					return True
		return False

	def isTrafficWaiting(self):
		""" Check if any car is waiting
		"""
		for carID in self.traci.vehicle.getIDList():
			if carID != self.egoCarID:
				speed = self.traci.vehicle.getSpeed(carID)
				if speed <= 1e-1:
					return True
		return False

	def takeAction(self, accel):
		# New speed
		dt = self.traci.simulation.getDeltaT()/1000.0
		self.speed = self.speed + (dt)*accel
		#print(self.traci.simulation.getDeltaT()/1000.0)
		if self.speed > self.max_speed:
			# Exceeded lane speed limit
			self.speed = self.max_speed 
		elif self.speed < 0 :
			self.speed = 0

		#print("\n Speed is : {0}".format(self.speed))
		self.traci.vehicle.slowDown(self.egoCarID, self.speed,int(dt*1000))



	

