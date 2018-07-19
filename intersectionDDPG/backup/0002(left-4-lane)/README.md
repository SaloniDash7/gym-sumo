Rewards:
	Step cost : -0.05
	Non Zero Speed reward : + 0.01
	Traffic Braking and Traffic Waiting : -0.05
	Traffic Braking or Traffic Waiting : -0.025
	Traffic not Braking and Traffic not Waiting : +0.05
	Collision Cost : -10
	Goal Reward : +5

Actor:
	actor = Sequential()
	#actor.add(Input(shape=(1,18,26,3),name='observation_input'))
	actor.add(Flatten(input_shape=(1,18,26,3)))
	actor.add(Dense(512, activation='relu'))
	actor.add(Dense(512, activation='relu'))
	actor.add(Dense(256, activation='relu'))
	actor.add(Dense(64, activation='relu'))
	actor.add(Dense(nb_actions, activation='tanh'))

Critic:
	(SAME)

Optimizer: Adam(lr=0.0001,clipnorm=1.) for both

(State space : [speed,angle,1]

54/3000 collisions
(1.8%)
