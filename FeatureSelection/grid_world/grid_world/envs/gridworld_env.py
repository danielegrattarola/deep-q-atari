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
        self.height = 9
        self.width = 16
        self._cell_size = 10

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.height * self._cell_size, self.width * self._cell_size, 1)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.viewer.move_character(action)
        self.state = self.viewer.get_state()
        done = self.viewer.is_on_goal()
        reward = 1 if done else 0
        return self.state, reward, done, {}

    def _reset(self, height=9, width=16):
        self.viewer = Viewer(height=height, width=width, cell_size=self._cell_size)
        self.state = self.viewer.get_state()
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        return self.viewer.render()
