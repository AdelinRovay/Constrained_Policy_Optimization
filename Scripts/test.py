import gym
import envs
from PIL import Image
import numpy as np
import time
#from envs import custom_ant_ev
from gym_recorder import Recorder

env = gym.make('CustomTerrainAnt-v0')#,model_path='assets/ant_custom_terrain.xml', frame_skip=5, observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(111,), dtype=np.float32))
# env = gym.make('FrozenLake-v1')#, render_mode='ansi')
xenv = Recorder(env, episode_num=10)

# frames=[]
for ep in range(10):
    observation = env.reset()
    done=False
    while not done :
        action = env.action_space.sample()
        observation, reward, done,truncated,info = env.step(action)
        env.render(mode='human')
        # time.sleep(0.1)
        



    # image = Image.fromarray(px)
    # image.save('E:/mujoco_project/frame.png')
    

