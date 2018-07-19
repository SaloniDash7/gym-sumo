import numpy as np
import sys
sys.path.append('../')
import gym
import gym_sumo
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import ModelIntervalCheckpoint
import warnings
warnings.filterwarnings('ignore')
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dropout,Dense, Activation, Flatten, Input, Concatenate, Conv2D,MaxPooling2D, Reshape
from keras.optimizers import Adam
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--action',help='"train_from_scratch" or "resume_training", or "test"')
args = parser.parse_args()

env = gym.make('SumoGUI-v0')
nb_actions = env.action_space.shape[0]
print(nb_actions)


actor = Sequential()
#actor.add(Input(shape=(1,18,26,3),name='observation_input'))
actor.add(Flatten(input_shape=(1,18,26,3)))
actor.add(Dense(512, activation='relu'))
#actor.add(Dropout(0.1))
actor.add(Dense(512, activation='relu'))
#actor.add(Dropout(0.1))
actor.add(Dense(256, activation='relu'))
#actor.add(Dropout(0.1))
actor.add(Dense(64, activation='relu'))
actor.add(Dense(nb_actions, activation='tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,18,26,3), name='observation_input')
observation_input_flat = Flatten()(observation_input)
x = Concatenate()([action_input,observation_input_flat])
x = Dense(512, activation='relu')(x)
#x = Dropout(0.1)(x)
x = Dense(512, activation='relu')(x)
#x = Dropout(0.1)(x)
x = Dense(256, activation='relu')(x)
#x = Dropout(0.1)(x)
x = Dense(64, activation='relu')(x)
x = Dense(1,activation='tanh')(x)
critic = Model(inputs = [action_input,observation_input], outputs = x)
print(critic.summary())

if(args.action=='resume_training' or args.action=='test'):
	actor.load_weights('./CheckPoints/_actor')
	critic.load_weights('./CheckPoints/_critic')
	print('Weights loaded..... \n')

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=['mae'])

if(args.action=='test'):
	agent.test(env,nb_episodes=1000,verbose=2,visualize=False,nb_max_episode_steps=300)

tbCallback = TensorBoard(log_dir='./Graph/',write_grads=True,write_graph=True,histogram_freq=0)
ckptCallback = ModelIntervalCheckpoint(filepath='./CheckPoints/',interval=1000)
agent.fit(env, nb_steps=250000, visualize=False, verbose=2, nb_max_episode_steps=150,callbacks=[tbCallback,ckptCallback])
