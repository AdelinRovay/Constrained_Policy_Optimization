from gym.envs.registration import register

register(
    id='CustomTerrainAnt-v0',
    entry_point='envs.custom_ant_env:CustomTerrainAntEnv',

    # reward_threshold=6000.0,
    # xml="ant_custom_terrain.xml",
)
register(
    id='CustomTerrainHumanoid-v0',
    entry_point='envs.custom_humanoid_env:CustomTerrainHumanoidEnv'
    # max_episode_steps=1000,
)
