import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import gym 


class CustomTerrainAntEnv(mujoco_env.MuJocoPyEnv, utils.EzPickle):
    metadata = {
    'render_modes': ['human', 'rgb_array', 'depth_array'],
    'video.frames_per_second': 50,
    'render_fps': 20
    }

    def __init__(self):#, model_path='assets/ant_custom_terrain.xml', frame_skip=5, observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(111,), dtype=np.float32)):
        # super().__init__()
        # self.model_path = model_path
        # self.frame_skip = frame_skip
        # self.observation_space = observation_space
        # Define action space: Assuming 8-dimensional continuous actions with range [-1, 1]
        #self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(111,), 
        dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
        low=-1, 
        high=1, 
        shape=(8,), 
        dtype=np.float32
        )
        print(type(self.observation_space))


        xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'ant_custom_terrain.xml')
        print(xml_path)
        print(CustomTerrainAntEnv.metadata)

        mujoco_env.MuJocoPyEnv.__init__(self, xml_path, 5, observation_space=self.observation_space)
        # self.sim = self._get_sim()
        utils.EzPickle.__init__(self)

        
    # class CustomTerrainAntEnv(utils.EzPickle, mujoco_env.MuJocoPyEnv):
    #     metadata = {
    #         'render_modes': ['human', 'rgb_array', 'depth_array'],
    #         'video.frames_per_second': 50,
    #         'render_fps': 20
    #     }

    #     def __init__(self):
    #         xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'ant_custom_terrain.xml')
    #         super().__init__(xml_path, 5) # Calling parent initializer

    #         # Define action and observation spaces
    #         self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
    #         self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(111,), dtype=np.float32)

    #         utils.EzPickle.__init__(self)
        

    def step(self, a):
        truncated = False
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5
            * 1e-3
            * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = (
            np.isfinite(state).all() 
        )
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            truncated,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

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
