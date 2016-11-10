import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='GridWorld-v0',
    entry_point='grid_world.grid_world.envs:GridWorldEnv',
    timestep_limit=1000,
    reward_threshold=1.0
)
