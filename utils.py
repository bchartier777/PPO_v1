import argparse
import numpy as np
import torch

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--train_test', dest='train_test', type=str, default='train')
	parser.add_argument('--act_file', dest='act_file', type=str, default='')
	parser.add_argument('--crit_file', dest='crit_file', type=str, default='')
	parser.add_argument('--d_hidden', dest='d_hidden', type=int, default='128')
	parser.add_argument('--max_tstep', dest='max_tstep', type=int, default='1000000')
	parser.add_argument('--tstep_batch', dest='tstep_batch', type=int, default='2048')
	parser.add_argument('--env_name', dest='env_name', type=str, default='LunarLanderContinuous-v2')
	parser.add_argument('--tstep_ep', dest='tstep_ep', type=int, default='200')
	parser.add_argument('--gamma', dest='gamma', type=float, default='0.99')
	parser.add_argument('--epoch', dest='epoch', type=int, default='10')
	parser.add_argument('--lr', dest='lr', type=float, default='3e-4')
	parser.add_argument('--clip', dest='clip', type=float, default='0.2')
	parser.add_argument('--img_rend', dest='img_rend', action='store_true', default=True)
	parser.add_argument('--img_rend_freq', dest='img_rend_freq', type=int, default='10')
	parser.add_argument('--checkpoint', dest='checkpoint', type=int, default='10')
	parser.add_argument('--seed', dest='seed', type=int, default='10')

	args = parser.parse_args()

	return args

def save_models (args, actor, critic, count):
	if args.env_name == 'BipedalWalker-v3':
		mod_name = 'Bipedal'
	elif args.env_name == 'LunarLanderContinuous-v2':
		mod_name = 'LunarLander'
	elif args.env_name == 'Pendulum-v0':
		mod_name = 'Pendulum'
	elif args.env_name == 'MountainCarContinuous-v0':
		mod_name = 'MountainCar'
	act_fname = "./models/agent_{}_actor.pth".format(mod_name)
	crit_fname = "./models/agent_{}_critic.pth".format(mod_name)
	if count % args.checkpoint == 0:
		torch.save(actor.state_dict(), act_fname)
		torch.save(critic.state_dict(), crit_fname)

def output_progress(event_log):
	t_step = event_log['t_step']
	tstep_batch = event_log['tstep_batch']
	avg_ep_len = np.mean(event_log['ep_len'])
	avg_rew_per_ep = np.mean([np.sum(rew_per_ep) for rew_per_ep in event_log['reward']])
	avg_act_loss = np.mean([losses.float().mean() for losses in event_log['act_loss']])
	avg_crit_loss = np.mean([losses.float().mean() for losses in event_log['crit_loss']])

	print(f"Batch timestep: {tstep_batch}")
	print(f"Avg episode len: {avg_ep_len}")
	print(f"Avg episode return: {avg_rew_per_ep}")
	print(f"Actor loss: {avg_act_loss}")
	print(f"Crit loss: {avg_crit_loss}")
	print(f"Timestep: {t_step}")
	print(f"\r\r")

	event_log['ep_len'], event_log['reward'], event_log['act_losses'] = [], [], []
