# Deep Q-learning for Atari Games
This is an implementation in Keras and OpenAI Gym of the Deep Q-Learning
algorithm (often referred to as Deep Q-Network, or DQN) by Mnih et al.
on the well known Atari games.
  
Rather than a pre-packaged tool to simply see the agent playing the game,
this is a model that needs to be trained and fine tuned by hand and has
more of an educational value.
This code tries to replicate the experimental setup described in
[the original DeepMind paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

A similar project on Deep Q-Learning applied to videogames can be found
on [this repo of mine](https://github.com/danielegrattarola/deep-q-snake).

## Acknowledgments
Make sure to cite the paper by Mnih et al. if you use this code for
your research:
```
@article{mnih2015human,
    title={Human-level control through deep reinforcement learning},
    author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A and Veness, Joel and Bellemare, Marc G and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K and Ostrovski, Georg and others},
    journal={Nature},
    volume={518},
    number={7540},
    pages={529--533},
    year={2015},
    publisher={Nature Research}
}
```

and remember to give me a shoutout by linking to my
[Github profile](https://github.com/danielegrattarola),
[blog](https://danielegrattarola.github.io), or
[Twitter](https://twitter.com/riceasphait).

A big thank you to Carlo D'Eramo
([@carloderamo](https://github.com/carloderamo)), who helped me
develop this code and spent a good amount of time debugging in the
beginning.

## Setup
To run the script you'll need the following dependencies:
- [Keras](http://keras.io/#installation)
- [OpenAI Gym](https://gym.openai.com/)  
- [PIL](http://www.pythonware.com/products/pil/)
- [h5py](http://packages.ubuntu.com/trusty/python-h5py)

which should all be available through Pip.
  
No additional setup is needed, so simply clone the repo:
```sh
git clone https://gitlab.com/danielegrattarola/deep-q-atari.git
cd deep-q-atari
```  
  
## Usage
A default training session can be run by typing:
```sh
python atari.py -t
```  
which will train the model with the same parameters as described in 
[this Nature article](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html), 
on the `MsPacmanDeterministic-v4` environment.
  
By running:
```sh
python atari.py -h
```  
you'll see the options list. The possible options are:
`-t, train`: train the agent;  
`-l, --load`: load the neural network weights from the given path;  
`-v, --video`: show video output;  
`-d, --debug`: run in debug mode (no output files);
`--eval`: evaluate the agent;
`-e, --environment`: name of the OpenAI Gym environment to use 
(default: MsPacman-v0);
`--minibatch-size`: number of sample to train the DQN at each update;  
`--replay-memory-size`: number of samples stored in the replay memory;  
`--target-network-update-freq`: frequency (number of frames) with which
the target DQN is updated;
`--avg-val-computation-freq`: frequency (number of DQN updates) with which the 
average reward and Q value are computed;  
`--discount-factor`: discount factor for the environment;  
`--update-freq`: frequency (number of steps) with which to train the DQN;  
`--learning-rate`: learning rate for the DQN;
`--epsilon`: initial exploration rate for the agent;  
`--min-epsilon`: final exploration rate for the agent;  
`--epsilon-decrease`: rate at which to linearly decrease epsilon;  
`--replay-start-size`: minimum number of transitions (with fully random policy) 
to store in the replay memory before starting training;  
`--initial-random-actions`: number of random actions to be performed by the 
agent at the beginning of each episode;  
`--dropout`: dropout rate for the DQN;  
`--max-episodes`: maximum number of episodes that the agent can experience 
before quitting;  
`--max-episode-length`: maximum number of steps in an episode;  
`--max-frames-number`: maximum number of frames for a run;  
`--test-freq`: frequency (number of episodes) with which to test the agent's 
performance;  
`--validation-frames`: number of frames to test the model like in table 3 of the
 paper  
`--test-states`: number of states on which to compute the average Q value;  
  
The possible environments on which the agent can be trained are all the 
environments in the Atari gym package.
A typical usage of this script on an headless server (e.g. EC2 instance) would 
look like this:
```sh
python atari.py -t -e BreakoutDeterministic-v4
```
If you want to see the actual game being played by the agent, simply add the `-v`
flag to the above command (note that this will obviously slow the collection of
samples.
  
## Output
You'll find some csv files in the output folder of the run (`output/runYYYMMDD-hhmmss`)
 which will contain raw data for the analysis of the agent's performance.
  
More specifically, the following files will be produced as output:  
1. **training_info.csv**: will contain the episode length and cumulative 
(non-clipped) reward of each training episode;    
2. **evaluation_info.csv**: will contain the episode length and cumulative 
(non-clipped) reward of each evaluation episode;  
3. **training_history.csv**: will contain the average loss and accuracy for each
 training step, as returned by the `fit` method of Keras;  
4. **test_score_mean_q_info.csv**: will contain the average score and Q-value 
(computed over a number of held out random states defined by the `--test-states`
 flag) calculated at intervals of N DQN updates (where N is set by the
 `--avg-val-computation-freq` flag);  
5. **log.txt**: a text file with various information about the parameters of the 
run and the progress of the model;  
6. **model_DQN.h5, model_DQN_target.h5**: files containing the weights of the 
DQN and target DQN (both files will be saved when the script quits or is killed 
with `ctrl+c`). You can pass any of these files as argument with the `--load` 
flag to initialize a new DQN with these weights (Note: the DQN architecture must
be unchanged for this to work);