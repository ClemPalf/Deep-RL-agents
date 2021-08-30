# Precis
This repository is composed of 3 different agents.
In order to choose which one you want to see perform, make the appropriate changes within *main.py*:  

```
from agent1 import Agent1
agent = Agent(state_size=8, action_size=4, seed=0)
...
agent.qnetwork_local.load_state_dict(torch.load('weights_agent1.pth'))
```

## Agent 1: DDQN
The popular Q-learning algorithm is known to overestimate action values under certain conditions. The idea of [Double-Q-Learning](https://arxiv.org/abs/1509.06461) has been proven very effective to solve this problem and increase the learning speed. 



| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |
