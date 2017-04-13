from __future__ import print_function
import gym
import numpy as np
import time
import utils
from joblib import Parallel, delayed

max_mean_score = 0


def evaluate(DQA, args, logger):
    global max_mean_score

    evaluation_csv = 'evaluation.csv'
    logger.to_csv(evaluation_csv, 'length,score')
    env = gym.make(args.environment)
    scores = list()
    frame_counter = 0

    while frame_counter < args.validation_frames:
        remaining_random_actions = args.initial_random_actions
        obs = utils.preprocess_observation(env.reset())

        frame_counter += 1
        # Initialize the first state with the same 4 images
        current_state = np.array([obs, obs, obs, obs])
        t = 0
        episode = 0
        score = 0

        # Start episode
        while True:
            # Render the game if video output is not suppressed
            if args.video:
                env.render()

            action = DQA.get_action(np.asarray([current_state]),
                                    testing=True,
                                    force_random=remaining_random_actions > 0)
            obs, reward, done, info = env.step(action)
            obs = utils.preprocess_observation(obs)
            current_state = utils.get_next_state(current_state, obs)

            if remaining_random_actions > 0:
                remaining_random_actions -= 1

            score += reward
            t += 1
            frame_counter += 1

            # End episode
            if done or t > args.max_episode_length:
                episode += 1
                print('Episode %d end\n---------------\nFrame counter: %d\n' % 
                      (episode, frame_counter))
                print('Length: %d\n, Score: %f\n\n' % (t, score))
                # Save episode data in the evaluation csv
                logger.to_csv(evaluation_csv, [t, score])
                break
                
        scores.append([t, score])

    scores = np.asarray(scores)
    max_indices = np.argwhere(scores[:, 1] == np.max(scores[:, 1])).ravel()
    max_idx = np.random.choice(max_indices)

    # Save best model
    if max_mean_score < np.mean(scores):
        max_mean_score = np.mean(scores)
        DQA.DQN.save(append='_best')

    return scores[max_idx, :].ravel()


def _eval_and_render(mdp, policy, n_episodes=1, metric='discounted',
                     initial_state=None, render=True):
    """
    This function evaluate a policy on the specified metric by executing
    multiple episode and visualize its performance
    Params:
        policy (object): a policy object (method get_action is expected)
        n_episodes (int): the number of episodes to execute
        metric (string): the evaluation metric ['discounted', 'average']
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
        step (float): average number of step before finish
        step_confidence (float):  95% confidence level for step average
    """
    values, steps = _eval_and_render_vectorial(mdp, policy, n_episodes, metric, 
                                               initial_state, render)

    return values.mean(), 2 * values.std() / np.sqrt(n_episodes), \
           steps.mean(), 2 * steps.std() / np.sqrt(n_episodes)


def _eval_and_render_vectorial(mdp, policy, n_episodes=1, metric='discounted',
                               initial_state=None, render=True):
    """
    This function evaluate a policy on the specified metric by executing
    multiple episode and visualize its performance
    Params:
        policy (object): a policy object (method get_action is expected)
        n_episodes (int): the number of episodes to execute
        metric (string): the evaluation metric ['discounted', 'average']
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
        step (float): average number of step before finish
        step_confidence (float):  95% confidence level for step average
    """
    fps = mdp.metadata.get('video.frames_per_second') or 100
    values = np.zeros(n_episodes)
    steps = np.zeros(n_episodes)
    gamma = mdp.gamma
    if metric == 'average':
        gamma = 1
    for e in range(n_episodes):
        ep_performance = 0.0
        df = 1
        t = 0
        H = np.inf
        done = False
        if render:
            mdp.render(mode='human')
        if hasattr(mdp, 'horizon'):
            H = mdp.horizon
        mdp.reset_agent()
        state = mdp._reset(initial_state)
        while (t < H) and (not done):
            if policy:
                action = policy.get_action(state)
            else:
                action = mdp.action_space.sample()
            state, r, done, _ = mdp.step(action)
            ep_performance += df * r
            df *= gamma
            t += 1

            if render:
                mdp.render()
                time.sleep(1.0 / fps)
        # if(t>= H):
        #    print("Horizon!!")
        if gamma == 1:
            ep_performance /= t
        print("\tperformance", ep_performance)
        values[e] = ep_performance
        steps[e] = t

    return values, steps


def _parallel_eval(mdp, policy, n_episodes, metric, initial_state, n_jobs, 
                   n_episodes_per_job):
    # return _eval_and_render(mdp, policy, n_episodes, metric,
    #                         initial_state, False)
    how_many = int(round(n_episodes / n_episodes_per_job))
    out = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(_eval_and_render)(gym.make(mdp.spec.id), policy,
                                  n_episodes_per_job, metric, initial_state)
        for _ in range(how_many))

    # Out is a list of quadruplet: mean J, 95% conf lev J, mean steps, 
    # 95% conf lev steps (confidence level should be 0 or NaN)
    values, steps = np.array(out)
    return values.mean(), 2 * values.std() / np.sqrt(n_episodes),\
           steps.mean(), 2 * steps.std() / np.sqrt(n_episodes)


def evaluate_policy(mdp, policy, n_episodes=1,
                    metric='discounted', initial_state=None, render=False,
                    n_jobs=-1, n_episodes_per_job=10):
    """
    This function evaluate a policy on the given environment w.r.t.
    the specified metric by executing multiple episode.
    Params:
        policy (object): a policy object (method drawAction is expected)
        n_episodes (int): the number of episodes to execute
        metric (string): the evaluation metric ['discounted', 'average']
        initial_state (np.array, None): initial state where to start the episode.
            If None the initial state is selected by the mdp.
        render (bool): flag indicating whether to visualize the behavior of
            the policy
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
    """
    assert metric in ['discounted', 'average'], "Unsupported metric"
    if render:
        return _eval_and_render(mdp, policy, n_episodes, metric, initial_state,
                                True)
    else:
        return _parallel_eval(mdp, policy, n_episodes, metric, initial_state,
                              n_jobs, n_episodes_per_job)


def collect_episode(mdp, policy=None):
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

        action = np.reshape(action, 1)

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


def collect_episodes(mdp, policy=None, n_episodes=100, n_jobs=-1):
    out = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(collect_episode)(gym.make(mdp.spec.id), policy) 
        for _ in range(n_episodes))

    # Output is a list of np.array, each one representing an episode
    data = np.concatenate(out, axis=0)  # Merge the results
    return data
