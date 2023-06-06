import gym
import sys
import torch
import Box2D
import argparse

from PPO import PPO
from network import Actor, Critic
from test_model import test_model
from utils import parse_args

# Tested on: 'BipedalWalker-v3', 'LunarLanderContinuous-v2', 'Pendulum-v0', 'MountainCarContinuous-v0'
def train(args, act_file, crit_file):
	print(f"Training environment", args.env_name)
	env = gym.make(args.env_name)
	agent = PPO(args, actor_class=Actor, critic_class=Critic, env=env)

	if act_file != '' and crit_file != '':
		agent.actor.load_state_dict(torch.load(act_file))
		agent.critic.load_state_dict(torch.load(crit_file))
	else:
		print(f"Full retrain.")

	agent.train_agent(args)

def test(args, act_file):
	if act_file == '':
		print(f"Error: Actor model required for Test mode.")
		sys.exit(0)

	env1 = gym.make(args.env_name)
	d_space = env1.observation_space.shape[0]
	d_act = env1.action_space.shape[0]
	model = Actor(d_space, d_act, args.d_hidden)
	model.load_state_dict(torch.load(act_file))
	test_model(args, policy=model, env=env1)

def main(args):
	if args.train_test == 'train':
		train(args, act_file=args.act_file, crit_file=args.crit_file)
	else:
		test(args, act_file=args.act_file)

if __name__ == '__main__':
	args = parse_args() # Parse arguments from command line
	main(args)
