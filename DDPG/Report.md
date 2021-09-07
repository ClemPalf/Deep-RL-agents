# Precis

This DDPG agent was initially built following the original [Continuous control with deep reinforcement learning paper](https://arxiv.org/abs/1509.02971).  
The main idea consists in implementing actor_critic methods to solve an environment with continuous action spaces.  

## Specifications: 

DDPG with experience replay uses 2 networks for each actor and critic element, one as source network which makes the predictions and other as target which after every set interatctions with the environment updates the source networks.

The critic (value) network maps (state, action) pairs -> Q-values.  
The actor (policy) network maps states -> Actions. 

Exeriences (state, reward, action, next_state) get stored into a replay buffer of size 1,000,000.  
Every 10 timsteps, we sample a batch of 128 elements from the replay buffer to train the actor and critic networks 10 times. This methods tends to make the training more stable.  

First, the critic is trained using the idea of Double Q training (the action value within the TD target is calculated by selecting the best action with the local net, but estimated using the target net). To avoid divergence, the weights are clipped at 1. Secondly, the actor is trained using the negative of the critic prediction.  
Additionnaly, the target networks were soft-updated using the following formula:  θ_target = τ*θ_local + (1 - τ)*θ_target with τ = 1e-3   
  
To explore the environment, noises were added to the action values using the Ornstein-Uhlenbeck process. This noise was slowly decay with an epislon value decreasing of 1e-6 at every training step.


- The actor network is composed of 4 fully-connected layers with 256, 128, 64, and 4 units.
- The critic network is composed of 3 fully-connected layers with 256, 128, and 1 units. Note that for the first layer, only the state is used as input, the output of this layer is then concatenated with the action before passing through the others.  
   
The following table contains the rest of the hyperparameters used within this implemetation:
| Parameter     | Description                 | Value    |
| ------------- |-----------------------------|----------|
| GAMMA         | Discount factor             | 0.99     |
| LR_ACTOR      | Learning rate of the actor  | 2e-4     |
| LR_CRITIC     | Learning rate of the critic | 2e-4     |
| WEIGHT_DECAY  | L2 weight decay             | 0        |



## Performance: 
Using the aformentionned configuration, the environment was solved in 129 episodes.
<p align="center">
  <img src="https://github.com/ClemPalf/Deep-RL-agents/blob/main/DDPG/images/Results.png"/>
</p>

## Recommandations: 
To increase the speed of learning, one could select the batch of experiences to train in a more targeted way.  
Prioritized Experience Replay is a type of experience replay in reinforcement learning where we select more frequently replay transitions with high expected learning progress, as measured by the magnitude of their temporal-difference (TD) error.  
To learn more about it, follow this [link](https://arxiv.org/abs/1511.05952).












