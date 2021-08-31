# Precis

This DQN agent was initially built following the original [DeepMind paper].  
The main idea consists in implementing the Sarsamax algorithm with a neural net as state value function approximator.  
To solve our selected environment (as describe within the README.md), 2 main improvements were made to the original DQN algorithm. 

## Improvements:
1) Double-Q-Learning  
The popular Q-learning algorithm is known to overestimate action values under certain conditions. The idea of [Double-Q-Learning](https://arxiv.org/abs/1509.06461) has been proven very effective to solve this problem and increase the learning speed.
The main idea is that when updating the local neural net parameters, the action value within the TD target is calculated by selecting the best action with the local net, but estimated using a different (often called "target") net.
 The following lines of code within agent.py implement this idea:  
 ```
next_best_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
Q_targets_next = self.qnetwork_target(next_states).gather(1, next_best_actions)
```


2) Dualling network  
The idea behind this technique is to slightly change the neural architecture, so that the output is a combination of the state value and the state-dependent action value (advantage value). The following figure illustrates the idea.  
The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm.  
You can find the implementation within model.py

## Specifications: 

-epsilon 
The behaviour policy during training was e-greedy with e slowly decayed at each time step (eps_decay=0.995) down to 0.01.
-soft_update
-neural net
-parameters








