import gym
from Viewer import Viewer
from gym import spaces
from gym.utils import seeding


import logging
logger = logging.getLogger(__name__)


class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
        self.width = 16
        self.height = 9
        self._cell_size = 10

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.height * self._cell_size, self.width * self._cell_size, 1)

        self.viewer = Viewer(width=self.width, height=self.height, cell_size=self._cell_size)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.viewer.move_agent(action)
        self.state = self.viewer.get_state()
        done = self.viewer.is_on_goal()
        reward = 1 if done else 0
        return self.state, reward, done, {}

    def _reset(self):
        self.viewer.reset_agent()
        self.state = self.viewer.get_state()
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        return self.viewer.render()

    def set_grid_size(self, width, height):
        self.width = width
        self.height = height
        self.viewer = Viewer(height=self.height, width=self.width, cell_size=self._cell_size)
        self.reset()
