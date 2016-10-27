from __future__ import print_function
import gym
from builtins import range
import time
import numpy as np
from joblib import Parallel, delayed
from PIL import Image

from Logger import Logger

def preprocess_observation(obs):
	image = Image.fromarray(obs, 'RGB').convert('L').resize((84, 110))  # Convert to gray-scale and resize according to PIL coordinates
	return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0])  # Convert to array and return


def get_next_state(current, obs):
	# Next state is composed by the last 3 images of the previous state and the new observation
	return np.append(current[1:], [obs], axis=0)


def evaluate(DQA, args, logger):
    evaluation_csv = 'evaluation_info.csv'
    logger.to_csv(evaluation_csv, 'length,score')
    env = gym.make(args.environment)
    scores = list()
    frame_counter = 0

    while frame_counter < args.validation_frames:
        remaining_random_actions = args.initial_random_actions
    	observation = preprocess_observation(env.reset())
        frame_counter += 1
    	current_state = np.array([observation, observation, observation, observation])  # Initialize the first state with the same 4 images
        t = 0
        episode = 0
        score = 0

        # Start episode
        while True:
            action = DQA.get_action(np.asarray([current_state]), testing=True, force_random=remaining_random_actions > 0)
            observation, reward, done, info = env.step(action)
            observation = preprocess_observation(observation)
            current_state = get_next_state(current_state, observation)

            remaining_random_actions -= 1 if remaining_random_actions > 0 else 0

            score += reward
            t += 1
            frame_counter += 1

            # End episode
            if done or t > args.max_episode_length:
                episode += 1
                print('Episode %d end\n---------------\nFrame counter: %d\n' % (episode, frame_counter))
                print('Length: %d\n, Score: %f\n\n' % (t, score))
                logger.to_csv(evaluation_csv, [t, score])  # Save episode data in the evaluation csv
                break

        scores.append(score)

    return np.max(scores)


def _eval_and_render(mdp, policy, nbEpisodes=1, metric='discounted',
                     initialState=None, render=True):
    """
    This function evaluate a policy on the specified metric by executing
    multiple episode and visualize its performance
    Params:
        policy (object): a policy object (method drawAction is expected)
        nbEpisodes (int): the number of episodes to execute
        metric (string): the evaluation metric ['discounted', 'average']
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
        step (float): average number of step before finish
        stepConfidence (float):  95% confidence level for step average
    """
    values, steps = _eval_and_render_vectorial(mdp, policy, nbEpisodes, metric, initialState, render)

    return values.mean(), 2 * values.std() / np.sqrt(nbEpisodes), \
           steps.mean(), 2 * steps.std() / np.sqrt(nbEpisodes)


def _eval_and_render_vectorial(mdp, policy, nbEpisodes=1, metric='discounted',
                               initialState=None, render=True):
    """
    This function evaluate a policy on the specified metric by executing
    multiple episode and visualize its performance
    Params:
        policy (object): a policy object (method drawAction is expected)
        nbEpisodes (int): the number of episodes to execute
        metric (string): the evaluation metric ['discounted', 'average']
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
        step (float): average number of step before finish
        stepConfidence (float):  95% confidence level for step average
    """
    fps = mdp.metadata.get('video.frames_per_second') or 100
    values = np.zeros(nbEpisodes)
    steps = np.zeros(nbEpisodes)
    gamma = mdp.gamma
    if metric == 'average':
        gamma = 1
    for e in range(nbEpisodes):
        epPerformance = 0.0
        df = 1
        t = 0
        H = np.inf
        done = False
        if render:
            mdp.render(mode='human')
        if hasattr(mdp, 'horizon'):
            H = mdp.horizon
        mdp.reset()
        state = mdp._reset(initialState)
        while (t < H) and (not done):
            if policy:
                action = policy.get_action(state)
            else:
                action = mdp.action_space.sample()
            state, r, done, _ = mdp.step(action)
            epPerformance += df * r
            df *= gamma
            t += 1

            if render:
                mdp.render()
                time.sleep(1.0 / fps)
        # if(t>= H):
        #    print("Horizon!!")
        if gamma == 1:
            epPerformance /= t
        print("\tperformance", epPerformance)
        values[e] = epPerformance
        steps[e] = t

    return values, steps


def _parallel_eval(mdp, policy, nbEpisodes, metric, initialState, n_jobs, nEpisodesPerJob):
    # TODO using joblib
    # return _eval_and_render(mdp, policy, nbEpisodes, metric,
    #                         initialState, False)
    how_many = int(round(nbEpisodes / nEpisodesPerJob))
    out = Parallel(
        n_jobs=n_jobs, verbose=2,
    )(
        delayed(_eval_and_render)(gym.make(mdp.spec.id), policy, nEpisodesPerJob, metric, initialState)
        for _ in range(how_many))

    # out is a list of quadruplet: mean J, 95% conf lev J, mean steps, 95% conf lev steps
    # (confidence level should be 0 or NaN)
    values, steps = np.array(out)
    return values.mean(), 2 * values.std() / np.sqrt(nbEpisodes), \
           steps.mean(), 2 * steps.std() / np.sqrt(nbEpisodes)


def evaluate_policy(mdp, policy, nbEpisodes=1,
                    metric='discounted', initialState=None, render=False,
                    n_jobs=-1, nEpisodesPerJob=10):
    """
    This function evaluate a policy on the given environment w.r.t.
    the specified metric by executing multiple episode.
    Params:
        policy (object): a policy object (method drawAction is expected)
        nbEpisodes (int): the number of episodes to execute
        metric (string): the evaluation metric ['discounted', 'average']
        initialState (np.array, None): initial state where to start the episode.
                                If None the initial state is selected by the mdp.
        render (bool): flag indicating whether to visualize the behavior of
                        the policy
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
    """
    assert metric in ['discounted', 'average'], "unsupported metric for evaluation"
    if render:
        return _eval_and_render(mdp, policy, nbEpisodes, metric, initialState, True)
    else:
        return _parallel_eval(mdp, policy, nbEpisodes, metric, initialState, n_jobs, nEpisodesPerJob)


def collectEpisode(mdp, policy=None):
    """
    This function can be used to collect a dataset running an episode
    from the environment using a given policy.

    Params:
        policy (object): an object that can be evaluated in order to get
                         an action

    Returns:
        - a dataset composed of:
            - a flag indicating the end of an episode
            - state
            - action
            - reward
            - next state
            - a flag indicating whether the reached state is absorbing
    """
    done = False
    t = 0
    H = np.inf
    data = list()
    action = None
    if hasattr(mdp, 'horizon'):
        H = mdp.horizon
    state = mdp.reset()
    while (t < H) and (not done):
        if policy:
            action = policy.get_action(state)
        else:
            action = mdp.action_space.sample()
        nextState, reward, done, _ = mdp.step(action)

        # TODO: should look the dimension of the action
        action = np.reshape(action, (1))

        if not done:
            if t < H:
                # newEl = [0] + state.tolist() + action.tolist() + [reward] + \
                #         nextState.tolist() + [0]
                newEl = [[0], state, action, [reward], nextState, [0]]
            else:
                # newEl = [1] + state.tolist() + action.tolist() + [reward] + \
                #         nextState.tolist() + [0]
                newEl = [[1], state, action, [reward], nextState, [0]]
        else:
            # newEl = [1] + state.tolist() + action.tolist() + \
            #         [reward] + nextState.tolist() + [1]
            newEl = [[1], state, action, [reward], nextState, [1]]

        # assert len(newEl.shape) == 1
        # data.append(newEl.tolist())
        data.append(newEl)
        state = nextState
        t += 1

    return data


def collectEpisodes(mdp, policy=None, nbEpisodes=100, n_jobs=-1):
    out = Parallel(
        n_jobs=n_jobs, verbose=2,
    )(
        delayed(collectEpisode)(gym.make(mdp.spec.id), policy)
        for _ in range(nbEpisodes))

    # out is a list of np.array, each one representing an episode
    # merge the results
    data = np.concatenate(out, axis=0)
    return data
