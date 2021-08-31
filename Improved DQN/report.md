# Precis

This DQN agent was initially built following the original [DeepMind paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).  
The main idea consists in implementing the Sarsamax algorithm with a neural net as state value function approximator.  
To solve our selected environment (as describe within the README.md), 2 main improvements were made to the original DQN algorithm. 

## Improvements:
1) Double-Q-Learning  
The popular Q-learning algorithm is known to overestimate action values under certain conditions. The idea of [Double-Q-Learning](https://arxiv.org/abs/1509.06461) has been proven very effective to solve this problem and increase the learning speed.
The main idea is that when updating the local neural net parameters, the action value within the TD target is calculated by selecting the best action with the local net, but estimated using a different (often called "target") net.
 The following lines of code within dqn_agent.py implement this idea:  
 ```
next_best_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
Q_targets_next = self.qnetwork_target(next_states).gather(1, next_best_actions)
```


2) Dualling network  
The idea behind this technique is to slightly change the neural architecture, so that the output is a combination of the state value and the state-dependent action value (advantage value). 
The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm.  
You can find the implementation within model.py.

## Specifications: 

The behaviour policy during training was e-greedy. E was slowly decayed at each time step (eps_decay=0.995) down to 0.01.  
  
The local and target neural net are both updated every 4 time step. More precisly, the target net was soft-updated using the following formula:  θ_target = τ*θ_local + (1 - τ)*θ_target with τ = 1e-3  
  
A buffer of size 100000 was used to store the experiences and train the agent by experience replay.  
  
The neural network architecture is composed of two branches:  
- The state branche, composed of 3 fully-connected layers with 128, 64, and 1 units.
- The advantage branhce, composed of 3 fully-connected layers with 128, 64, and 4 (number of possible actions) units.
  
All relevant other relevant hyperparameters can be found at the beginning of dqn_agent.py.

## Performance: 
Using the aformentionned configuration, the environment was solved in 413 episodes.
<p align="center">
  <img src="https://github.com/ClemPalf/Deep-RL-agents/blob/main/Improved%20DQN/images/Results.png"/>
</p>

## Recommandations: 
To increase the speed of learning, one could select the batch of experiences to train in a more targeted way.  
Prioritized Experience Replay is a type of experience replay in reinforcement learning where we select more frequently replay transitions with high expected learning progress, as measured by the magnitude of their temporal-difference (TD) error.  
To learn more about it, follow this [link](https://arxiv.org/abs/1511.05952).


