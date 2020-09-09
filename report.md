# Double-jointed arm control using a Proximal Policy Optimization (PPO) Agent

## Introduction
The work is about training a double-jointed arm using deep reinforcement learning to move to target locations. For this project, you will work with the Unity ML-Agents Reacher environment. In this environment, a double-jointed arm can move to target locations. In this work, we will use a version of this environment that contains 20 identical agents, each with its own copy of the environment. This helps in distributing the task of gathering experience and thus speeds up the learning process.

![reacher](https://github.com/MathiasThor/deepReach/blob/master/data/deepReach.png)

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible. The goal is to reach an average score of +30 over 100 consecutive episodes overall 20 agents.
Learning algorithm
The learning algorithm is split into to main functions: the `rollout_manger()` and the `PPO_algorithm()`. 

## Rollout manager
As the name indicates, `rollout_manger()` is responsible for collecting rollouts or trajectories that can be used for training the PPO agent. Specifically, it collects and stores: actions, log probability of the actions, values, rewards, episode done status, and states. It also calculates the future discounted returns as well as estimating the advantages. Finally, it can get data from the trajectories for the PPO agent to use for learning.

## PPO agent
The `PPO_algorithm()` implements the Proximal Policy Optimization algorithm and is in charge of optimizing the agents' selection of actions in order to achieve the highest score. First, it instantiates the actor and critic networks, which convert the 33 input states into a predicted best action and estimated value, respectively. Both networks have the **33 states as input and two fully connected layers with 32 neurons in each. Both networks also use the ReLu activation function between all layers.**

**The actor-network has four outputs**, representing the mean of the predicted distribution of the best actions for the current input current. The standard deviation can be calculated from these means, and the best action can then be sampled from the resulting normal distribution.

**The critic-network has 1 output** which represents the estimated value of the input state. The estimated value is subsequentially converted into the advantage estimate as follows (remember this function is placed in the `rollout_manger()` class):
```python
advantages = torch.tensor(np.zeros((num_agents, 1)), device=DEVICE, dtype=torch.float32)
for i in reversed(range(HORIZON)):
  td = self.rewards[i] + (GAE_DISCOUNT * self.episode_not_dones[i] * self.values[i + 1]) - self.values[i]
  advantages = advantages * GAE_LAMBDA * GAE_DISCOUNT * self.episode_not_dones[i] + td
  self.advantages[i] = advantages.detach() 
```
The advantage estimate basically tells us "How much better was the action that I took based on the expectation of what would normally happen in the state that I was in".

Finally, the policy loss (for the actor) and value loss (for the critic) is calculated as follows:
```python
  # Policy Loss
  ratio = (log_prob_action - sampled_log_probs_old).exp() 
  obj = ratio * sampled_advantages
  obj_clipped = ratio.clamp(1.0 - PPO_CLIP_RANGE, 1.0 + PPO_CLIP_RANGE) * sampled_advantages
  policy_loss = -torch.min(obj, obj_clipped).mean() 

  # Value Loss
  value_loss = 0.5 * (sampled_returns - value).pow(2).mean()
```
Note that the policy loss is clipped according to the PPO algorithm. This is done to ensure that the updated policy does not move too far away from the current policy. When moving too far away, the advantage estimate becomes inaccurate and may destroy the policy. The ratio is used for reweighing the advantages which enable us to use sampled tractories generated under a previous policy.
     
After computing the policy and value loss the weights if the actor and critic networks can be optimized as follows:
```python
  # Optimize network weights
  loss = policy_loss + value_loss
  optimizer.zero_grad()
  loss.backward()
  nn.utils.clip_grad_norm_(self.network.parameters(), 0.75) 
  optimizer.step()
```

## Training, plotting, and testing
For training, the agent with the PPO algorithm `train_agent()` is used. The training function will continue to iterate through a training loop until either a maximum number of episodes is taken or a specified mean score over 100 episodes is achieved. Afterward, the actor and critic network's weights are saved, and the scores for each episode are written to a CSV file. This CSV file can later be used for plotting the scores for the entire or multiple training sessions (as we show later in this report). Finally, the learned network weights can be loaded, and the agent can be tested to verify the results visually. 

## Hyperparameters
The hyperparameters are set as follows:

| Hyperparametser | Value | Description |
|--|--|--|
| HORIZON | 275 | PPO gathers trajectories as far out as the horizon limits, then performs a stochastic gradient descent (SGD) update |
| DISCOUNT_FACTOR | 0.99 | Discount factor used for calculating future returns |
| GAE_DISCOUNT | 0.99 | GAE Discount factor performs a bias-variance trade-off of the trajectories and can be viewed as a form of reward shaping |
| GAE_LAMBDA | 0.95 | GAE Lambda perform a bias-variance trade-off of the trajectories and can be viewed as a form of reward shaping  |
| EPOCH_RANGE | 12 | Number of updates per optimize step |
| MINIBATCH_SIZE | 64 | Minibatch size used for the optimize step |
| PPO_CLIP_RANGE | 0.05, 0.1, 0.15 | PPO uses a surrogate loss function to keep the step from the old policy to the new policy within a safe range. This hyperparameter sets the clipping of this surrogate loss function |
| LEARNING_RATE | 0.0003 | The learning rate used by the optimizer |
| | | |

Due to this project's scope, only the `PPO_CLIP_RANGE` hyperparameter will be tested for different settings, while the remaining hyperparameters will remain fixed. The results of varying `PPO_CLIP_RANGE` can be seen in the following section.

## Experiments and results
The following plot shows the results of using three different `PPO_CLIP_RANGE` values (0.05, 0.10, and 0.15). For each `PPO_CLIP_RANGE` value, the learning algorithm was executed for 300 episodes and repeated five times. This makes it possible to calculate and plot the mean return and standard error.

![plot](https://github.com/MathiasThor/deepReach/blob/master/data/score_episode.png)
*Plot of the mean score for all 20 robots per episode when using three different PPO clip range parameters*

As can be seen, a PPO clipping range of 0.05 performs the best, with the highest converged score and the lowest standard error. Using this clipping range, it is possible to reliably achieve an average score of at least 30 (up to more than 38) for more than 100 episodes in approx. 200 episodes. A PPO clipping range of 0.10 met the score requirement three out of five times, and a range of 0.15 was never able to. Thus, it is beneficial for the task to keep the policy in the proximity of the old policy. In the `/saved_weights` directory a weight set using `PPO_CLIP_RANGE` = 0.05 is placed. This can be loaded and tested with the "Load and test trained agent" cell.

## Future work
In the future, it would be beneficial to do more parameter tuning to get a more robust performance (i.e., less variation in the results) and faster convergence. It would also be interesting to look at different network architectures for the actor and critic networks to see how this affects the learning performance.

