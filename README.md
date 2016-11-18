# Deep Q-learning for Atari Games
This is an implementation in Keras and OpenAI Gym of deep Q-learning applied to the well known environments of the Atari games.  
The project was conducted with the machine learning research team of Politecnico di Milano led by Prof. Restelli.  
  
Rather than a pre-packaged tool to simply see the agent playing the game, this is a model that needs to be trained and fine tuned by hand and has more of an educational value.  
This work is closely inspired by [the original DeepMind paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) by Mnih et al. on deep Q-learning, and the variations are mostly focused on the Q-newtork architecture in order to analyse the difference in performance.  
  
A similar work on deep Q-learning applied to videogames can be found on [my Gitlab page](https://gitlab.com/danielegrattarola/deep-q-snake).  

### Installation
To run the script you'll need [Keras](http://keras.io/#installation) and [OpenAI Gym](https://gym.openai.com/) installed on your system.  
Other dependencies include [PIL](http://www.pythonware.com/products/pil/) and [h5py](http://packages.ubuntu.com/trusty/python-h5py), which should be available through Pip.  
  
To install the script simply download the source code:
```sh
git clone https://gitlab.com/danielegrattarola/deep-q-atari.git
cd deep-q-atari
```  
  
### Usage
A default training session can be run by typing:
```sh
python atari.py -t
```  
which will train the model with the same parameters as described in [this DeepMind paper](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html), on the MsPacman environment.  
  
By running:
```sh
python atari.py -h
```  
you'll see the options list. Possible options are:  
`-t, train`: train the agent;  
`-l, --load`: load the neural network weights from the given path;  
`-v, --novideo`: suppress video output (useful to train on headless servers);  
`-d, --debug`: run in debug mode (no output files);  
`-e, --environment`: name of the OpenAI Gym environment to use (default: MsPacman-v0).;  
`--minibatch-size`: number of transitions to train the DQN on;  
`--replay-memory-size`: number of samples stored in the replay memory;  
`--target-network-update-freq`: frequency (number of DQN updates) with which the target DQN is updated;  
`--avg-val-computation-freq`: frequency (number of DQN updates) with which the average reward and Q value are computed;  
`--discount-factor`: discount factor for the environment;  
`--update-freq`: frequency (number of steps) with which to train the DQN;  
`--learning-rate`: learning rate for the DQN;  
`--epsilon`: initial exploration rate for the agent;  
`--min-epsilon`: final exploration rate for the agent;  
`--epsilon-decrease`: rate at which to linearly decrease epsilon;  
`--replay-start-size`: minimum number of transitions (with fully random policy) to store in the replay memory before starting training;  
`--initial-random-actions`: number of random actions to be performed by the agent at the beginning of each episode;  
`--dropout`: dropout rate for the DQN;  
`--max-episodes`: maximum number of episodes that the agent can experience before quitting;  
`--max-episode-length`: maximum number of steps in an episode;  
`--test-freq`: frequency (number of episodes) with which to test the agent's performance;  
`--test-states`: number of states on which to compute the average Q value;  
  
The possible environments on which the agent can be trained are all the environments in the Atari gym package, which include: MsPacman-v0, BeamRider-v0, Breakout-v0, Enduro-v0, Pong-v0, Qbert-v0, Seaquest-v0, SpaceInvaders-v0, etc..  
A typical usage of this script on an dedicated headless VPS (e.g. EC2 instance) would look like this:
```sh
python atari.py -t -v -e Breakout-v0 
```  
  
### Output
Running the script with any combination of options (except `-d`) will output some data collections to help you interpret the data.  
You'll find the pertinent csv files in the output folder of the run (output/runYYYMMDD-hhmmss) which will contain raw data for the analysis of the agent's performance.  
  
More specifically, the following files will be output:  
1. **test_info.csv**: will contain the episode length and cumulative (non-clipped) reward of each test episode;  
2. **training_info.csv**: will contain the episode length and cumulative (non-clipped) reward of each training episode;  
3. **training_history.csv**: will contain the average loss and accuracy for each training session, as output by the `fit` method of Keras;  
4. **average_values_training.csv**: will contain the average reward and Q-value (computed over a held out set of random states, whose number is set by the `--test-states` flag) over a period of N DQN updates (where N is set by the `--avg-val-computation-freq` flag);  
5. **log.txt**: a logfile with various information about the parameters of the run and the progress of the model;  
6. **model_DQN.h5, model_DQN_target.h5**: files containing the latest weights of the DQN and target DQN (both files will be saved when the script quits or is killed with `ctrl+c`). Pass any of these as argument to the `--load` flag to initialize a new DQN with these weights (Note: the DQN architecture must be the same for this to work);  
