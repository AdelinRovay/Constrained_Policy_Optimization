import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from os import path
import gym
import os

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class CustomTerrainHumanoidEnv(mujoco_env.MuJocoPyEnv, utils.EzPickle):
    metadata = {
    'render_modes': ['human', 'rgb_array', 'depth_array'],
    'video.frames_per_second': 50,
    'render_fps': 67
    }
    def __init__(self):
        self.observation_space = gym.spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(129,), 
        dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
        low=-1, 
        high=1, 
        shape=(8,), 
        dtype=np.float32
        )
        # self.obj_dim = 2
        

        print(type(self.observation_space))
        # Define observation space: This is just an example, you might need to adjust the size
        # depending on the actual size of the observations returned by the _get_obs method
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

        xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid_custom_terrain.xml')
        print(xml_path)
        print(CustomTerrainHumanoidEnv.metadata)
        #mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        # print(self.metadata)
        mujoco_env.MuJocoPyEnv.__init__(self, xml_path, 5, observation_space=self.observation_space)
        # self.sim = self._get_sim()
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        ).astype(np.float32)


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        #qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
    # def _get_obs(self):
    #     data = self.sim.data
    #     return np.concatenate([data.qpos.flat[2:],
    #                            data.qvel.flat,
    #                            data.cinert.flat,
    #                            data.cvel.flat,
    #                            data.qfrc_actuator.flat,
    #                            data.cfrc_ext.flat])

    # def step(self, a):
    #     truncated = False
    #     xposbefore = self.get_body_com("torso")[0]
    #     self.do_simulation(a, self.frame_skip)
    #     xposafter = self.get_body_com("torso")[0]
    #     forward_reward = (xposafter - xposbefore) / self.dt
    #     ctrl_cost = 0.5 * np.square(a).sum()
    #     contact_cost = (
    #         0.5
    #         * 1e-3
    #         * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
    #     )
    #     survive_reward = 1.0
    #     reward = forward_reward - ctrl_cost - contact_cost + survive_reward
    #     state = self.state_vector()
    #     notdone = (
    #         np.isfinite(state).all() 
    #     )
    #     done = not notdone
    #     ob = self._get_obs()
    #     return (
    #         ob,
    #         reward,
    #         done,
    #         truncated,
    #         dict(
    #             reward_forward=forward_reward,
    #             reward_ctrl=-ctrl_cost,
    #             reward_contact=-contact_cost,
    #             reward_survive=survive_reward,
    #         ),
    #     )

    def step(self, a):
        truncated=False
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        ob = self._get_obs()
        data = self.sim.data

        alive_bonus = 3.0
        reward_run = 1.25 * (pos_after - pos_before) / self.dt + alive_bonus
        reward_energy = 3.0 - 4.0 * np.square(data.ctrl).sum() + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        return ob, 0., done,truncated, dict(
                reward_forward=reward_run,
                reward_energy=reward_energy,
                reward_survive=alive_bonus,)
        #{'obj': np.array([reward_run, reward_energy])}

    # def reset_model(self):
    #     c = 0.01
    #     self.set_state(
    #         self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
    #         self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
    #     )
    #     self.obj_dim = 2  # You might need to adjust this based on your specific use case
    #     return self._get_obs()

    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = 1
    #     self.viewer.cam.distance = self.model.stat.extent * 1.0
    #     self.viewer.cam.lookat[2] = 2.0
    #     self.viewer.cam.elevation = -20
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.05
        self.viewer.cam.elevation = -90

    def render(self, mode='human', width=500, height=500):
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)  # specify width and height if needed
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            return np.asarray(data)[:, :, :3]  # Just RGB, no alpha
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            return self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
        elif mode == 'human':
            self._get_viewer(mode).render()
        else:
            raise ValueError("Unknown render mode: {}".format(mode))






# import numpy as np
# from gym.envs.mujoco import mujoco_env
# from gym import utils
# from os import path

# def mass_center(model, sim):
#     mass = np.expand_dims(model.body_mass, 1)
#     xpos = sim.data.xipos
#     return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

# class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     def __init__(self):
#         self.obj_dim = 2
#         mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/humanoid.xml"), frame_skip = 5)
#         utils.EzPickle.__init__(self)

#     def _get_obs(self):
#         data = self.sim.data
#         return np.concatenate([data.qpos.flat[2:],
#                                data.qvel.flat,
#                                data.cinert.flat,
#                                data.cvel.flat,
#                                data.qfrc_actuator.flat,
#                                data.cfrc_ext.flat])

#     def step(self, a):
#         pos_before = mass_center(self.model, self.sim)
#         self.do_simulation(a, self.frame_skip)
#         pos_after = mass_center(self.model, self.sim)
#         ob = self._get_obs()
#         data = self.sim.data
        
#         alive_bonus = 3.0
#         reward_run = 1.25 * (pos_after - pos_before) / self.dt + alive_bonus
#         reward_energy = 3.0 - 4.0 * np.square(data.ctrl).sum() + alive_bonus
#         qpos = self.sim.data.qpos
#         done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        
#         return ob, 0., done, {'obj': np.array([reward_run, reward_energy])}

#     def reset_model(self):
#         c = 0.01
#         self.set_state(
#             self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
#             self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
#         )
#         return self._get_obs()

#     def viewer_setup(self):
#         self.viewer.cam.trackbodyid = 1
#         self.viewer.cam.distance = self.model.stat.extent * 1.0
#         self.viewer.cam.lookat[2] = 2.0
#         self.viewer.cam.elevation = -20