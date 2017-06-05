import argparse
import evaluation
import gym
import joblib
from DQAgent import DQAgent
from Logger import Logger


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true',
                    help='run in debug mode (no output files)')
parser.add_argument('-e', '--environment', type=str,
                    help='name of the OpenAI Gym environment to use '
                         '(default: MsPacman-v0) \nDeepMind paper: MsPacman-v0,'
                         ' BeamRider-v0, Breakout-v0, Enduro-v0, Pong-v0, '
                         'Qbert-v0, Seaquest-v0, SpaceInvaders-v0',
                    default='Breakout-v0')
args = parser.parse_args()

logger = Logger(debug=args.debug, append=args.environment)

env = gym.make(args.environment)
network_input_shape = (4, 110, 84)  # Dimension ordering: 'th'
DQA = DQAgent(env.action_space.n,
              network_input_shape)

episodes = evaluation.collect_episode(env, DQA)
joblib.dump(episodes, logger.path + 'evaluation_dataset.pickle')
