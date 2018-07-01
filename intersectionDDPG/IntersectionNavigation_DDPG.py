import numpy as np
import sys
sys.path.append('../')
import gym
import gym_sumo
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential, Model
from keras.layers import Dropout,Dense, Activation, Flatten, Input, Concatenate, Conv2D,MaxPooling2D, Reshape
from keras.optimizers import Adam

STATE_W = 128
STATE_H = 128

env = gym.make('SumoGUI-v0')
nb_actions = env.action_space.shape[0]
print(nb_actions)

actor = Sequential()
actor.add(Reshape((STATE_W,STATE_H,1),input_shape=(1,STATE_W,STATE_H,1)))
actor.add(Conv2D(64, (3, 3), activation='relu', input_shape=(STATE_W,STATE_H,1)))
actor.add(Conv2D(64, (3, 3), activation='relu'))
actor.add(MaxPooling2D(pool_size=(2,2)))
actor.add(Dropout(0.25))
actor.add(Conv2D(32, (3, 3), activation='relu'))
actor.add(Conv2D(32, (3, 3), activation='relu'))
actor.add(MaxPooling2D(pool_size=(2,2)))
actor.add(Dropout(0.25))
actor.add(Flatten())
actor.add(Dense(2048, activation='relu'))
actor.add(Dropout(0.5))
actor.add(Dense(2048, activation='relu'))
actor.add(Dropout(0.5))
actor.add(Dense(nb_actions, activation='sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,STATE_W,STATE_H,1), name='observation_input')
processed_input = Reshape((STATE_W,STATE_H,1),input_shape = (1,STATE_W,STATE_H,1))(observation_input)
processed_input = Conv2D(64, (3, 3), activation='relu', input_shape=(STATE_W,STATE_H,1))(processed_input)
processed_input = Conv2D(64, (3, 3), activation='relu')(processed_input)
processed_input = MaxPooling2D(pool_size=(2,2))(processed_input)
processed_input = Dropout(0.25)(processed_input)
processed_input = Conv2D(32, (3, 3), activation='relu')(processed_input)
processed_input = Conv2D(32, (3, 3), activation='relu')(processed_input)
processed_input = MaxPooling2D(pool_size=(2,2))(processed_input)
processed_input = Dropout(0.25)(processed_input)
observation_input_flat = Flatten()(processed_input)
x = Concatenate()([action_input,observation_input_flat])
x = Dense(2048, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)
critic = Model(inputs = [action_input,observation_input], outputs = x)
print(critic.summary())

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=5000, nb_steps_warmup_actor=5000,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

agent.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=300)