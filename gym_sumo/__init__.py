from gym.envs.registration import register

register(
    id='Sumo-gui-v0',
    entry_point='gym_sumo.envs:SUMOEnv',
)
