from gym.envs.registration import register

register(
    id='SumoGUI-v0',
    entry_point='gym_sumo.envs:SUMOEnv_Initializer',
)
