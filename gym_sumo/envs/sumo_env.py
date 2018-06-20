from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
import traci
import traci.constants as tc
from scipy.misc import imread
from gym import spaces
from string import Template
import os, sys
import numpy as np
import math
import time

class Car:
	""" A class struct that stores the car features.
	"""
	def __init__(self, carID, position = None, distance = None, speed = None, angle = None, signal = None, length = None):
		self.carID = carID
		self.position = position
		self.distance = distance
		self.speed = speed
		self.angle = angle
		self.signal = signal
		self.length = length


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
sys.exit("please declare environment variable 'SUMO_HOME'")

class SUMO_Env(Env):
	