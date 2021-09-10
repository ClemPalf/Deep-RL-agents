# Udacity - Deep Reinforcement Learning Nanodegree (Collaboration and Competition)

### Project Details

This is the third project of the Deep Reinforcement Learning Nanodegree. I trained a Multi DDPG Agent to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#Tennis) environment.  This project is influenced by the previous one: https://github.com/escribano89/reacher-ddpg and the DDPG implementations from the Udacity's repository https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is **considered solved**, when the average (over 100 episodes) of those scores is at least **+0.5**.

## Requirements
In order to prepare the environment, follow the next steps after downloading this repository:
* Create a new environment:
	* __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	* __Windows__: 
	```bash
	conda create --name dqn python=3.6 
	activate drlnd
	```
* Min install of OpenAI gym
	* If using __Windows__, 
		* download [swig for windows](http://www.swig.org/Doc1.3/Windows.html) and add it the PATH of windows
		* install [ Microsoft Visual C++ Build Tools ](https://visualstudio.microsoft.com/es/downloads/).
	* then run these commands
	```bash
	pip install gym
	pip install gym[classic_control]
	pip install gym[box2d]
	```
* Install the dependencies under the folder [python/] (https://github.com/udacity/deep-reinforcement-learning/tree/master/python).
```bash
	cd python
	pip install .
```
* Create an IPython kernel for the `drlnd` environment
```bash
	python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

* Download the environment from one of the links below.  You need only select the environment that matches your operating system:
     - Links
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

* Unzip the downloaded file and move it inside the project's root directory
* Change the kernel of your environment to `drlnd`
* Open the **params.py** file and change the path to the unity environment appropriately (UNITY_EXE_PATH=PATH_OF_THE_TENNIS_EXE)

## Getting started

If you want to test the trained agents, execute the **test.py** file. 

If you want to train the agents, execute the **train.py** file. After reaching the goal, the networks weights will be stored in the project's root folder.


## Resources

* report.pdf: A document that describes the details of the implementation and future proposals.
* madddpg: implemented agent using the MADDPG algorithm (contains ddpg agents)
* ddpg: ddpg agent
* actor: the actor NN model
* critic: the critic NN model
* actor_critic: The actor-critic model.
* unity_env: a class for handling the unity environment
* replay_buffer: a class for handling the experience replay
* ou_noise: a class for handling the initial exploration noise
* test.py: Entry point for testing the agents using the trained networks
* train.py: Entry point for training the agents using MADDPG algorithm
* *.pth files: Our model's weights ***(Solved in less than 1100 episodes)***

## Trace of the training

![Training](https://github.com/escribano89/tennis-maddpg/blob/main/score.PNG)

![Training](https://github.com/escribano89/tennis-maddpg/blob/main/trace.PNG)

## Video

You can find an example of the trained agents [here](https://youtu.be/ii6CPP9cpIM)

[![Navigation](https://img.youtube.com/vi/ii6CPP9cpIM/0.jpg)](https://youtu.be/ii6CPP9cpIM)
