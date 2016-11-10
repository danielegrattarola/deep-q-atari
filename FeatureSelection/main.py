import gym, random, argparse
from grid_world.grid_world.envs.gridworld_env import GridWorldEnv

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', action='store_true', help='display video output')
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
args = parser.parse_args()

# Example run
env = gym.make('GridWorld-v0')
env.set_grid_size(16, 9) # Optional

action_space = env.action_space.n

state = env.reset()
reward = 0
done = False

# Start episode
frame_counter = 0
while not done:
    frame_counter += 1
    # Select an action
    action = random.randrange(0, action_space)
    # Execute the action, get next state and reward
    state, reward, done, info = env.step(action)
    if args.video:
        raw_input()
        env.render()
# End episode

