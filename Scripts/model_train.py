# import mujoco_py
# import glfw
# from train_rewards import *
# import os
# import gym
# import numpy as np
# from stable_baselines3 import SAC
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from envs.custom_ant_env import CustomTerrainAntEnv
# from stable_baselines3.common.vec_env import DummyVecEnv
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import pickle
# from deap import base, creator, tools, algorithms

# log_dir = "E:/mujoco_project/logs"
# model_dir = "E:/mujoco_project/model"
# # def train(env, sb3_algo):
# #     num_envs = 4  # You can adjust this based on your hardware capabilities
# #     env = gym.make('CustomTerrainAnt-v0') #SubprocVecEnv([lambda: gym.make('CustomTerrainAnt-v0')] * num_envs)
    
# #     if sb3_algo == 'SAC':
# #         model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

# #     TIMESTEPS = 25000
# #     iters = 0
# #     max_iters = 1000000
# #     while iters<max_iters:
# #         iters += 1

# #         model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=custom_reward)

# #         # Save the model periodically, e.g., every 1000 timesteps
# #         print("iters: ",iters)
# #         if iters % 1000 == 0:
# #             print("model_path: ",f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")
# #             model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")

# def train(env, sb3_algo, total_timesteps=1000):
#     num_envs = 4  # You can adjust this based on your hardware capabilities
#     env = gym.make(env)  # Or use the provided environment

#     if sb3_algo == 'SAC':
#         model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

#     TIMESTEPS = total_timesteps
#     iters = 0
#     max_iters = 10
#     while iters < max_iters:
#         iters += 1
#         with tqdm(total=TIMESTEPS, desc=f'Iteration {iters}/{max_iters}', unit='timesteps') as pbar:

#             model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=lambda loc, glob: custom_reward(loc, glob, env))
#             pbar.update(TIMESTEPS)
#         # Save the model periodically, e.g., every 100 timesteps
#         # print("iters: ", iters)
#         if iters % 100 == 0:
#             print("model_path: ", f"{model_dir}/{sb3_algo}_{TIMESTEPS * iters}")
#             model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS * iters}")
        
#         logs_path = 'logs_RL_SAC.pkl'
#         with open(logs_path, 'rb') as f:
#             logs = pickle.load(f)
#             objective_speed.extend(logs['speed'])
#             objective_energy.extend(logs['energy'])

#     pareto_front = tools.sortNondominated(zip(objective_speed, objective_energy), len(objective_speed), first_front_only=True)[0]

#     # Print or analyze the Pareto front solutions
#     print("Pareto front solutions:")
#     for ind in pareto_front:
#         print(ind)

#     # Plotting
#     generations = range(len(objective_speed))
#     plt.figure(figsize=(10, 6))
#     plt.scatter(objective_speed, objective_energy, c=generations, cmap='viridis', label='Generation')
#     plt.colorbar(label='Generation')
#     plt.xlabel('Speed')
#     plt.ylabel('Energy')
#     plt.title('Energy vs Speed Color-coded by Generation')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
# if __name__ == "__main__":

#     # Set the algorithm
#     sb3_algo = 'SAC'

#     # Set the total timesteps
#     total_timesteps = 10

#     # Create the environment
#     env_name = 'CustomTerrainAnt-v0'

#     # Call the train function
#     train(env_name, sb3_algo, total_timesteps)


import mujoco_py
import glfw
from train_rewards import *
import os
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.custom_ant_env import CustomTerrainAntEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from deap import base, creator, tools, algorithms

log_dir = "E:/mujoco_project/logs"
model_dir = "E:/mujoco_project/model"

def train(env, sb3_algo, total_timesteps=1000):
    num_envs = 4
    env = gym.make(env)
    video_recorder = VideoRecorder(env, path='./video', enabled=True)# ,format='mp4')
    if sb3_algo == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    TIMESTEPS = total_timesteps
    iters = 0
    max_iters = 600
    while iters < max_iters:
        iters += 1
        with tqdm(total=TIMESTEPS, desc=f'Iteration {iters}/{max_iters}', unit='timesteps') as pbar:
            for _ in range(TIMESTEPS):
                model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=lambda loc, glob: custom_reward(loc, glob, env))
                video_recorder = VideoRecorder(env, path='./video', enabled=True)# ,format='mp4')
                video_recorder.capture_frame()
                pbar.update(TIMESTEPS)
                # Save the model periodically
        if iters % 100 == 0:
            print("model_path: ", f"{model_dir}/{sb3_algo}_{TIMESTEPS * iters}")
            model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS * iters}")
        
    # Load the logs after training completes
    logs_path = 'logs_RL_SAC.pkl'
    with open(logs_path, 'rb') as f:
        logs = pickle.load(f)
        objective_speed.extend(logs['speed'])
        objective_energy.extend(logs['energy'])
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    individuals = [creator.Individual([speed, energy]) for speed, energy in zip(objective_speed, objective_energy)]

# Sort individuals based on non-domination
    pareto_front = tools.sortNondominated(individuals, len(objective_speed), first_front_only=True)[0]

    # Print or analyze the Pareto front solutions
    print("Pareto front solutions:")
    for ind in pareto_front:
        print(ind.fitness.values)

    # Plotting
    generations = range(len(objective_speed))
    plt.figure(figsize=(10, 6))
    plt.scatter(objective_speed, objective_energy, c=generations, cmap='viridis', label='Generation')
    plt.colorbar(label='Generation')
    plt.xlabel('Speed')
    plt.ylabel('Energy')
    plt.title('Energy vs Speed Color-coded by Generation')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig('pareto_front_plot.png')

    # Display the plot
    plt.show()

if __name__ == "__main__":
    sb3_algo = 'SAC'
    total_timesteps = 500
    env_name = 'CustomTerrainAnt-v0'
    train(env_name, sb3_algo, total_timesteps)
