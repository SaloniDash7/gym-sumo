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
from collections import namedtuple

import os, sys, subprocess


class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	def __init__(self,mode='gui',simulation_end=36000):

		self.simulation_end = simulation_end
		self.mode = mode
		self._seed()
		self.traci = self.initSimulator(True,8870)
		self.sumo_step = 0
		self.collisions =0
		self.episode = 0
		self.flag = True

		## INITIALIZE EGO CAR
		self.egoCarID = 'veh0'
		self.speed = 0
		self.max_speed = 20.1168		# m/s 

		self.sumo_running = False
		self.viewer = None	

		self.observation = self._reset()

		self.action_space = spaces.Box(low=np.array([-1,-1]), high= np.array([+1,+1])) # acceleration, steering angle
		self.observation_space = spaces.Box(low=0, high=1, shape=(np.shape(self.observation)))


	
	def _observation(self):
		return self.getFeatures()


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



	def _reward(self):
		terminal = False
		terminalType = 'None'

		try :
			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))
			distance_ego = (np.asarray([np.linalg.norm(position_ego - self.endPos)]))[0]
		
		except:
			print("self.traci couldn't find car")
			self._reset()
			return -1.0, True, 'Car not found'
			distance_ego = 0

		# Step cost
		reward = -0.05

		# Encourage speed along lane center, discourage lateral speed
		
		c_angle = self.traci.vehicle.getAngle(self.egoCarID) 
		c_v = self.traci.vehicle.getSpeed(self.egoCarID)
		c_vx = c_v*np.sin(np.deg2rad(c_angle))
		c_vy = c_v*np.cos(np.deg2rad(c_angle))

		reward += c_vy
		reward -= c_vx
		reward -= c_angle*0.1
		
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
			print(self.collisions)
			reward += -100.0
			terminal = True
			terminalType = 'Collided!!!'
			self.episode += 1
			if(self.episode%100 == 0):
				self.flag = True



		else: # Goal check
			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))
			distance_ego = np.linalg.norm(position_ego - self.endPos)
			if position_ego[0] <= self.endPos[0]:
				reward += 50.0 
				terminal = True
				terminalType = 'Survived'
				self.episode +=1
				if(self.episode%100 == 0):
					self.flag = True
		

		return reward, terminal, terminalType



		
	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self,action_values):
		r = 0
		self.sumo_step +=1
		accel,angle = action_values[0], action_values[1]
		self.takeAction(accel,angle)
		self.traci.simulationStep()

		# Get reward and check for terminal state
		reward, terminal, terminalType = self._reward()
		r += reward

		braking = self.isTrafficBraking()
		# if egoCar.isTrafficWaiting(): waitingTime += 1

		self.observation = self._observation()

		info = {braking, terminalType}

		if(self.episode%100==0  and self.flag):
			self.flag = False
			print("Collision Rate : {0} ".format(self.collisions/100))
			self.collisions = 0

		return self.observation, reward, terminal, {}



	def initSimulator(self,withGUI,portnum):
		# Path to the sumo binary
		if withGUI:
			sumoBinary = "/usr/bin/sumo-gui"
		else:
			sumoBinary = "/usr/bin/sumo"

		sumoConfig = "/home/salonidash7/gym-sumo/gym_sumo/envs/sumo_configs/0002.sumocfg"

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
		self.endPos= [79.0, 114.00]

		
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

	def takeAction(self, accel, angle):
		dt = self.traci.simulation.getDeltaT()/1000.0
		c_angle = self.traci.vehicle.getAngle(self.egoCarID) 
		c_v = self.traci.vehicle.getSpeed(self.egoCarID)
		c_vx = c_v*np.sin(np.deg2rad(c_angle))
		c_vy = c_v*np.cos(np.deg2rad(c_angle))
		c_vx = c_vx + accel*np.sin(np.deg2rad(c_angle))*dt
		c_vy = c_vy + accel*np.cos(np.deg2rad(c_angle))*dt

		ego_x, ego_y = self.traci.vehicle.getPosition(self.egoCarID)

		x = ego_x + c_vx*dt
		y = ego_y + c_vy*dt
		edge_id = traci.vehicle.getRoadID(self.egoCarID)
		lane_index = traci.vehicle.getLaneIndex(self.egoCarID)
		
		if(angle>=0):
			angle = angle*90
		else:
			angle = angle*90 + 360

		self.traci.vehicle.moveToXY(vehID=self.egoCarID,edgeID=edge_id,lane=lane_index,x=x,y=y,angle=angle,keepRoute=1)

	def getFeatures(self):
		""" Main file for ego car features at an intersection.
		"""

		carDistanceStart, carDistanceStop, carDistanceNumBins = 0, 80, 40
		## LOCAL (101, 90ish)
		carDistanceYStart, carDistanceYStop, carDistanceYNumBins = -5, 40, 18 # -4, 24, relative to ego car
		carDistanceXStart, carDistanceXStop, carDistanceXNumBins = -80, 80, 26
		TTCStart, TTCStop, TTCNumBins = 0, 6, 30    # ttc
		carSpeedStart, carSpeedStop, carSpeedNumBins = 0, 20, 10 # 20  
		carAngleStart, carAngleStop, carAngleNumBins = -180, 180, 10 #36

		ego_x, ego_y = self.traci.vehicle.getPosition(self.egoCarID)
		ego_angle = self.traci.vehicle.getAngle(self.egoCarID)
		ego_v = self.traci.vehicle.getSpeed(self.egoCarID)

	
		discrete_features = np.zeros((carDistanceYNumBins, carDistanceXNumBins,3))

		# ego car
		pos_x_binary = self.getBinnedFeature(0, carDistanceXStart, carDistanceXStop, carDistanceXNumBins)
		pos_y_binary = self.getBinnedFeature(0, carDistanceYStart, carDistanceYStop, carDistanceYNumBins)
		x = np.argmax(pos_x_binary)
		y = np.argmax(pos_y_binary)
		discrete_features[y,x,:] = [0.0, ego_v/20.0, 1]

		for carID in self.traci.vehicle.getIDList(): 
			if carID==self.egoCarID:
				continue
			c_x,c_y = self.traci.vehicle.getPosition(carID)
			angle = self.traci.vehicle.getAngle(carID) 
			c_v = self.traci.vehicle.getSpeed(carID)
			c_vx = c_v*np.sin(np.deg2rad(angle))
			c_vy = c_v*np.cos(np.deg2rad(angle))
			zx,zv = c_x,c_vx
			
			p_x = c_x-ego_x
			p_y = c_y-ego_y
			
			if(c_vx!=0):
				c_ttc_x = float(np.abs(p_x/c_vx))
			else:
				c_ttc_x=10.0

			carframe_angle = wrapPi(angle-ego_angle)
			
			c_vec = np.asarray([p_x, p_y, 1])
			rot_mat = np.asarray([[ np.cos(np.deg2rad(ego_angle)), np.sin(np.deg2rad(ego_angle)), 0],
								  [-np.sin(np.deg2rad(ego_angle)), np.cos(np.deg2rad(ego_angle)), 0],
								  [                             0,                             0, 1]])
			rot_c = np.dot(c_vec,rot_mat) 

			carframe_x = rot_c[0] 
			carframe_y = rot_c[1] 
			

			pos_x_binary = self.getBinnedFeature(carframe_x, carDistanceXStart, carDistanceXStop, carDistanceXNumBins)
			pos_y_binary = self.getBinnedFeature(carframe_y, carDistanceYStart, carDistanceYStop, carDistanceYNumBins)
			
			x = np.argmax(pos_x_binary)
			y = np.argmax(pos_y_binary)

			discrete_features[y,x,:] = [carframe_angle/90.0, c_v/20.0, 1]
			
		
		return discrete_features

	def getBinnedFeature(self, val, start, stop, numBins):
		""" Creating binary features.
		"""
		bins = np.linspace(start, stop, numBins)
		binaryFeatures = np.zeros(numBins)

		if val == 'unknown':
			return binaryFeatures

		# Check extremes
		if val <= bins[0]:
			binaryFeatures[0] = 1
		elif val > bins[-1]:
			binaryFeatures[-1] = 1

		# Check intermediate values
		for i in range(len(bins) - 1):
			if val > bins[i] and val <= bins[i+1]:
				binaryFeatures[i+1] = 1

		return binaryFeatures

def wrapPi(angle):
	# makes a number -pi to pi
	while angle <= -180:
		angle += 360
	while angle > 180:
		angle -= 360
	return angle

