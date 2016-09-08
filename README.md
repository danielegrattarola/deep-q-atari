# Deep Q-learning for Atari Games
This is an implementation in Keras and OpenAI Gym of deep Q-learning applied to the well known environments of the Atari games. The project was conducted with the Machine Learning research team of Politecnico di Milano led by Prof. Restelli.    

Rather than a pre-packaged tool to simply see the agent playing the game, this is a model that needs to be trained and fine tuned by hand and has more of an educational value.   
This work is closely inspired by [the original DeepMind paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) by Mnih et al., and the variations are mostly focused on the Q-newtork architecture in order to analyse the difference in performance.   
 
A similar work on deep Q-learning applied to videogames can be found on [my Gitlab page](https://gitlab.com/danielegrattarola/deep-q-snake).

### Installation
To run the script you'll need Keras and OpenAI Gym installed on your system: see [here](http://keras.io/#installation) and [here](https://gym.openai.com/) for detailed instructions on how to install the libraries; you might want to enable GPU support in order to speed up the convolution, but since this a rather simple model it is not strictly necessary.    
Other dependencies include PIL and [h5py](http://packages.ubuntu.com/trusty/python-h5py), which should be available through Pip.   

To run the script simply download the source code:
```sh
git clone https://gitlab.com/danielegrattarola/deep-q-atari.git
cd deep-q-atari
```
and run: 
```sh
python atari.py -t
```
to run a training session on a new model.   

### Usage
By running:
```sh
python atari.py -h
```
you'll see the options list. Possible options are:
- **t, train**: train the Q-network. 
- **l, load path/to/file.h5**: initialize the Q-network using the weights stored in the given HDF5 file.
- **v, no-video**: suppress video output (useful to train on headless servers).
- **d, debug**: do not print anything to file and do not create the output folder.  

### Output
Running the script with any combination of options will output some useful data collections to help you interpret the data.     
You'll find the pertinent csv files in the output folder of the run (output/runYYYMMDD-hhmmss) which will contain useful raw data for the analysis of the agent's performance.




