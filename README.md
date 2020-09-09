# Deep Reach

### Introduction

In this work, we train a double-jointed arm using deep reinforcement learning (PPO) to move to target locations. The double-jointed arm is implemented in the Unity ML-Agents Reacher environment. In this work, we will use a version of the environment that contains 20 identical agents, each with its own copy of the environment. This helps in distributing the task of gathering experience and thus speeds up the learning process.

![reacher](https://github.com/MathiasThor/deepReach/blob/master/data/deepReach.gif)

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible. The goal is to reach an average score of +30 over 100 consecutive episodes overall 20 agents.

The observation space consists of 33 variables corresponding to the arm's position, rotation, velocity, and angular velocities. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To solve the environment, the agents must get an average score of +30 (over 100 consecutive episodes, and overall 20 agents). Specifically, after each episode, we add up each agent's rewards (without discounting) to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores. This produces an average score for each episode (where the average is overall 20 agents). The environment is considered solved when the average (over 100 episodes) of those average scores is +30.

### Getting started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - **Twenty (20) Agents**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

2. Place the file in the unity_simulation/ folder, and unzip (or decompress) the file.
3. Follow the dependencies guide at the end of this readme (if you haven't already).
4. Navigate to the project root folder: ```bash source activate drlnd && jupyter notebook ```
5. Specify the patch to the environment in the first cell of `deepReach.ipynb`.
6. Run all the cells to and with *train PPO agent* in `deepReach.ipynb` to start learning.
7. Run the *Plot score* to plot the average score against episodes from the learning session.
8. Run the last cell (*load and test trained agent*) to show the learned agent.


### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/MathiasThor/BananaCollector.git
cd BananaCollector/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment using the drop-down `Kernel` menu. 

<img src="https://sdk.bitmoji.com/render/panel/8fed4863-328c-4c87-a873-ad4c52275b63-fdafebb8-6b14-4983-9cbb-a539f77ab069-v1.png?transparent=1&palette=1" width="250" height="250">


