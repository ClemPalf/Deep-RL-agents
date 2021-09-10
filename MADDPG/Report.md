# Precis

MADDPG is a multi-agent variant of DDPG, a model-free, off-policy, policy gradient-based algorithm that uses two separate deep neural networks (one actor, one critic) to both explore the stocastic environment and, separately, learn the best policy to achieve maximum reward. DDPG has been shown to be quite effective at continuous control tasks and here the multi-agent version is applied to this continuous control task.

## Specifications: 

2 identical DDPG agents were created. 
An agent is composed of 4 different deep neural networks. DDPG with experience replay uses 2 networks for each actor and critic element, one as source network which makes the predictions and other as target which after every set interatctions with the environment updates the source networks.  

The critic (value) network maps (state, action) pairs -> Q-values.  
The actor (policy) network maps states -> Actions.  

Each network is composed of two hidden layers of 256-128 units, with ReLU activation functions on the hidden layers and tanh on the output layers. 
The actor network was getting fed the concatenated states from both agents.  
The critic network was getting fed the concatenated states from both agents, and their actions.

Exeriences (state, reward, action, next_state) were stored into a replay buffer of size 1,000,000. (Each agent has its own buffer, yet, they store the states and action from both agent).  
Every timstep, a batch of 128 elements were sampled from the replay buffer to train the actor and critic networks.  
  
To explore the environment, noises were added to the action values using the Ornstein-Uhlenbeck process. This noise was slowly decay with an noise schedule. More precisly:  
noise_schedule = lambda episode: max(NOISE_END, NOISE_START - episode * ((NOISE_START - NOISE_END) / NOISE_DECAY))  
With: NOISE_START = 6, NOISE_END = 6, and NOISE_DECAY = 500


   
The following table contains the rest of the hyperparameters used within this implemetation:
| Parameter     | Description                 | Value    |
| ------------- |-----------------------------|----------|
| GAMMA         | Discount factor             | 0.99     |
| LR_ACTOR      | Learning rate of the actor  | 1e-3     |
| LR_CRITIC     | Learning rate of the critic | 1e-3     |
| WEIGHT_DECAY  | L2 weight decay             | 0        |
| TAU           | Soft update coefficient     | 6e-2     |


## Performance: 
Using the aformentionned configuration, the environment was solved in 635 episodes.
<p align="center">
  <img src="https://github.com/ClemPalf/Deep-RL-agents/blob/main/MADDPG/images/Results.png"/>
</p>

## Recommandations: 
To increase the speed of learning, one could select the batch of experiences to train in a more targeted way.  
Prioritized Experience Replay is a type of experience replay in reinforcement learning where we select more frequently replay transitions with high expected learning progress, as measured by the magnitude of their temporal-difference (TD) error.  

