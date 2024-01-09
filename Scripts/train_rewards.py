import os
import gym
import numpy as np
# from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse
import stable_baselines3
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
#from stable_baselines3.common.envs import DummyVecEnv
#from stable_baselines3.multi_object.common.vec_env import SubprocVecEnv
#from stable_baselines3.multi_object.sac import MultiObjectiveSAC
# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools, algorithms
#@title Import packages for plotting and creating graphics
import time
import imageio
import itertools
# import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List
import pickle

 
def compute_speed_reward(observation):
    velocity = observation[VELOCITY_KEY]
    speed_reward = -np.linalg.norm(velocity)# Negative because you want to maximize speed
    print("__speed__",speed_reward)
    return speed_reward

def compute_energy_spent(observation):
    action = observation[ACTION_KEY]
    energy_spent = -np.sum(np.square(action))  # Negative because you want to minimize energy spent
    print("__energy__",energy_spent)
    return energy_spent

    
speed_avg = []
energy_avg = []
objective_speed = []
objective_energy = []
no_of_steps = 100

def custom_reward_human(callback_locals, callback_globals, env):
    global speed_avg, energy_avg, objective_speed, objective_energy, no_of_steps
    rewards = callback_locals['rewards']
    infos = callback_locals['infos']  # Extract the info dictionary
    # print("infos: ",infos)
    # print()
    total_speed = 0
    total_energy = 0
    total_reward = []

    # Use the provided environment (env) instead of calling env.reset()
    env.reset()

    action = callback_locals['actions']
    _, obs, reward, done, info = env.step(action[0])
    
    # Extract the relevant information from the info dictionary
    # print(info)
    total_speed = info['reward_forward']#(info['reward_forward']**2 + info['reward_ctrl']**2 + info['reward_contact']**2 + info['reward_survive']**2)**0.5
    total_energy = info['reward_energy']#np.sum(np.square(action[0]))
    # print('total_speed: ',total_speed)
    # print('total_energy: ',total_energy)
    speed_avg.append(total_speed)
    energy_avg.append(total_energy)

    if len(speed_avg) == no_of_steps:
        objective_speed.append(sum(speed_avg) / no_of_steps)
        objective_energy.append(sum(energy_avg) / no_of_steps)
        with open('logs_RL_human_SAC_data.pkl', 'wb') as f:
            logs = {'speed':objective_speed,'energy':objective_energy}
            pickle.dump(logs,f)
        speed_avg = []
        energy_avg = []

    # print("rewards: ", rewards)
    # print("total_speed: ", total_speed)
    # print("total_energy: ", total_energy)

    total_reward.append(rewards + 10 * total_speed - total_energy * (0.1 / 6))
    # print("__reward3__", 10 * total_speed, ' ', total_energy * (0.1 / 6), ' ', total_reward)

    return total_reward

def custom_reward(callback_locals, callback_globals, env):
    global speed_avg, energy_avg, objective_speed, objective_energy, no_of_steps
    rewards = callback_locals['rewards']
    infos = callback_locals['infos']  # Extract the info dictionary
    # print("infos: ",infos)
    # print()
    total_speed = 0
    total_energy = 0
    total_reward = []

    # Use the provided environment (env) instead of calling env.reset()
    env.reset()

    action = callback_locals['actions']
    _, obs, reward, done, info = env.step(action[0])
    
    # Extract the relevant information from the info dictionary
    # print(info)
    total_speed = (info['reward_forward']**2 + info['reward_ctrl']**2 + info['reward_contact']**2 + info['reward_survive']**2)**0.5
    total_energy = np.sum(np.square(action[0]))
    
    speed_avg.append(total_speed)
    energy_avg.append(total_energy)

    if len(speed_avg) == no_of_steps:
        objective_speed.append(sum(speed_avg) / no_of_steps)
        objective_energy.append(sum(energy_avg) / no_of_steps)
        with open('logs_RL_human_SAC_x.pkl', 'wb') as f:
            logs = {'speed':objective_speed,'energy':objective_energy}
            pickle.dump(logs,f)
        speed_avg = []
        energy_avg = []

    # print("rewards: ", rewards)
    # print("total_speed: ", total_speed)
    # print("total_energy: ", total_energy)

    total_reward.append(rewards + 10 * total_speed - total_energy * (0.1 / 6))
    # print("__reward3__", 10 * total_speed, ' ', total_energy * (0.1 / 6), ' ', total_reward)

    return total_reward


# def custom_reward(callback_locals, callback_globals, env):
#     global speed_avg, energy_avg, objective_speed, objective_energy, no_of_steps
#     rewards = callback_locals['rewards']

#     total_speed = 0
#     total_energy = 0
#     total_reward = []
    
#     # Use the provided environment (env) instead of calling env.reset()
#     env.reset() 

#     action = callback_locals['actions']
#     obs, reward, done, truncated, info = env.step(action[0])
#     print("info: ", info)
#     total_speed += (info['reward_forward']**2 + info['reward_ctrl']**2)**0.5
#     total_energy += np.sum(np.square(action[0]))
#     # total_speed = (info[''])
#     # total_speed = (info['x_velocity']**2 + info['y_velocity']**2)**0.5
#     # total_energy = np.sum(np.square(action))
#     speed_avg.append(total_speed)
#     energy_avg.append(total_energy)

#     if len(speed_avg) == no_of_steps:
#         objective_speed.append(sum(speed_avg) / no_of_steps)
#         objective_energy.append(sum(energy_avg) / no_of_steps)
#         speed_avg = []
#         energy_avg = []

#     print("rewards: ", rewards)
#     print("total_speed: ", total_speed)
#     print("total_energy: ", total_energy)

#     total_reward.append(rewards + 10 * total_speed - total_energy * (0.1 / 6))
#     print("__reward3__", 10 * total_speed, ' ', total_energy * (0.1 / 6), ' ', total_reward)

#     return total_reward

# def custom_reward(callback_locals, callback_globals):
#     global speed_avg,energy_avg,objective_speed,objective_energy,no_of_steps
#     #print("__reward__",callback_locals)
#     # observations = callback_locals['self'].env.get_attr('state')[0]
#     rewards = callback_locals['rewards']
#     #print("__reward__",rewards)
#     # total_reward = []
#     # for observation in observations:
#     #     speed_reward = compute_speed_reward(observation)
#     #     energy_reward = compute_energy_spent(observation)
#     #     total_reward.append(rewards + speed_reward + energy_reward)
#     action = callback_locals['actions']
#     #print("__reward2__",np.shape(action))
#     # print("----------------",(env.step(action)))
#     total_speed = 0
#     total_energy = 0
#     total_reward = []
#     env.reset()
#     # for _ in range(100):
#     action = callback_locals['actions']
#     _, obs, reward, done, info = env.step(action[0])

#     total_speed = (info['x_velocity']**2 + info['y_velocity']**2)**0.5
#     total_energy = np.sum(np.square(action))
#     speed_avg.append(total_speed)
#     energy_avg.append(total_energy)
#     if len(speed_avg) == no_of_steps:
#         objective_speed.append(sum(speed_avg)/no_of_steps)
#         objective_energy.append(sum(energy_avg)/no_of_steps)
#         speed_avg = []
#         energy_avg = []
#     print("rewards: ",rewards)
#     print("total_speed: ",total_speed)
#     print("total_energy: ",total_energy)
#     total_reward.append(rewards + 10*total_speed - total_energy*(0.1/6))
#     print("__reward3__",10*total_speed,' ',total_energy*(0.1/6),' ',total_reward)
#     return total_reward

