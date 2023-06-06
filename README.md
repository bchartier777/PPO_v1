# PPO_v1
Implementation of a simple PPO algorithm

## Overview
This is an unofficial implementation of the PPO RL algorithm.  The primary purpose of this implementation is to evaluate:
 - Multiple Actor and Critic networks
 - Two return / reward aggregation methods
 - A limited number of continuous-time Gym environments
 - Average episodic return for a limited number of seeds and hyperparameters

This has been tested for performance (average episodic returns) on  'LunarLanderContinuous-v2' 
and for execution but not performance on 'BipedalWalker-v3', 'Pendulum-v0' and 'MountainCarContinuous-v0'.

## Python version and Conda environment
This has been tested with Python 3.8.0 on Win 10.  Use of a virtual environment is recommended.
Following is a Conda implementation:

```
conda create --name ppov4_env python==3.8 pip
conda activate
pip install -r requirements_v1.txt
```

Note that this set of packages differs from the requirements.txt included in the reference repo listed in the References section below.

## Usage
All execution parameters are implemented in parse_args in utils.py.  Following is a sample train
execution:

```
python main.py --train_test train --act_file "" --crit_file "" --max_tstep 5000 --tstep_batch 2048 --tstep_ep 200 --gamma 0.99 --env_name LunarLanderContinuous-v2 --lr 3e-4 --clip 0.2 --img_rend --img_rend_freq 10 --checkpoint 10 1>ExecOut\stdOutPPO_Lunarv2.txt  2>ExecOut\stdErrPPO_Lunarv2.txt
```

Defining a file in act_file or crit_file will result in the module starting with the weights in 
the respective model file.

Following is a sample test / eval execution:

```
python main.py --train_test test --act_file "models\\agent_LunarLander_actor.pth" --crit_file "" --max_tstep 5000 --tstep_batch 2048 --tstep_ep 200 --gamma 0.99 --env_name LunarLanderContinuous-v2 --lr 3e-4 --clip 0.2 --img_rend --img_rend_freq 10 --checkpoint 10 1>ExecOut\stdOutPPO_Lunarv2.txt  2>ExecOut\stdErrPPO_Lunarv2.txt
```

A trained Actor network is required for test mode.

A few other train execution examples are in Exec1.bat.

# References
The primary reference repo was 'ericyangyu / PPO-for-Beginners', [here](https://github.com/ericyangyu/PPO-for-Beginners).  It is well written and has excellent documentation.  
It is from a [Medium article](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8).

[Original PPO paper](https://arxiv.org/abs/1707.06347), Schulman, J., et al.  "Proximal Policy Optimization Algorithms".  2017. arXiv:1707.06347

