# MADDPG to play Tennis

## Introduction


<p align="center">
  <img src="https://github.com/ClemPalf/Deep-RL-agents/blob/main/MADDPG/images/illustration.gif"/>
</p>



In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

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

## Instructions

### Navigating The Source Code
There are three important files which contains the source code to run and train our agent:
- `Tennis.ipynb` contains the instructions to explore the environent, train the agent to be smart and run the smart agent.
- `MADDPG_agent.py` contains the source code which describes how the agent works
- `models.py` contains the source code of the deep learning models for the actor and critic.

### Explore the Environment
In order to understand how to create a smart agent, we must recognize the environment first. Please, open `Continious Control.ipynb` using Jupyter Notebook and follow these steps:
- Step 1 : Start the Environment
- Step 2 : Examine the State and Action Spaces
- Step 3 : Take Random Actions in the Environment
- TRAIN : Train the agent using the algo described in Report.md

## License
This repository is under the **MIT license**.
