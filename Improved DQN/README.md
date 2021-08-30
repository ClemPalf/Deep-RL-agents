# Improved DQN
This repository contains an implementation of a Deep-Q-Network algorithm with the following improvements:  
- Double DQN
- Prioritized experience replay    
- Duelling DQN


The target of the agent is to collect bananas in a square world which is similar to the [banana collector of Unity](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).<br/>

<p align="center">
  <img src="https://github.com/ClemPalf/Deep-RL-agents/blob/main/Improved%20DQN/images/banana.gif"/>
</p>
  
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

<p align="center">
  <img src="https://github.com/ClemPalf/Deep-RL-agents/blob/main/Improved%20DQN/images/actions.png"/>
</p>

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.



## Requirements
To run the project, follow the next steps:
* Create a new environment:
	* __Linux__ or __Mac__: 
	```bash
	conda create --name dqn python=3.6
	source activate dqn
	```
	* __Windows__: 
	```bash
	conda create --name dqn python=3.6 
	activate dqn
	```
* Perform a minimal install of OpenAI gym
	* If using __Windows__, 
		* download swig for windows and add it the PATH of windows
		* install Microsoft Visual C++ Build Tools
	* then run these commands
	```bash
	pip install gym
	pip install gym[classic_control]
	pip install gym[box2d]
	```
* Install the dependencies under the folder python/
```bash
	cd python
	pip install -r requirements.txt .
```
* Create an IPython kernel for the `dqn` environment
```bash
	python -m ipykernel install --user --name dqn --display-name "dqn"
```
* Download the Unity Environment (thanks to Udacity) which matches your operating system
	* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
	* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
	* [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
	* [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

* Start jupyter notebook from the root of this python codes
```bash
jupyter notebook
```
* Once started, change the kernel through the menu `Kernel`>`Change kernel`>`dqn`
* If necessary, inside the ipynb files, change the path to the unity environment appropriately
