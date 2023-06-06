import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from utils import save_models, output_progress

class PPO:
	"""
		The PPO agent class
	"""
	def __init__(self, args, actor_class, critic_class, env):
		self.env = env
		self.d_space = env.observation_space.shape[0]
		self.d_act = env.action_space.shape[0]
		
		# Instantiate actor and critic networks, optimizer
		self.actor = actor_class(self.d_space, self.d_act, args.d_hidden)
		self.critic = critic_class(self.d_space, 1, args.d_hidden)
		self.opt_act = Adam(self.actor.parameters(), lr=args.lr)
		self.opt_crit = Adam(self.critic.parameters(), lr=args.lr)

	def train_agent(self, args):
		torch.manual_seed(args.seed)
		print(f"Using seed {args.seed}")
		v = torch.full(size=(self.d_act,), fill_value=0.5)

		cov = torch.diag(v) # Covariance matrix used to generate actions
		event_log = {
			't_step': 0,   # timesteps
			'tstep_batch': 0,   # batch iterations
			'ep_len': 0,    # length of episodes
			'reward': [],       # returns per episode
			'act_loss': [],     # actor network loss
			'crit_loss': [],     # critic network loss
		}

		print(f"Train model for {args.tstep_ep} timesteps")
		t_step = 0 # Total timesteps 
		tstep_batch = 0 # Batch iterations 
		# Iterate over user-defined max timesteps
		while t_step < args.max_tstep:
			# Generate a batch and all states, actions and rewards
			state, act_bat, log_probs_b, rew_bat, ep_len = self.update_batch(args, event_log, cov)
			t_step += ep_len
			tstep_batch += 1
			event_log['t_step'] = t_step
			event_log['tstep_batch'] = tstep_batch
			value, _ = self.evaluate(state, act_bat, cov)

			# Calculate and normalize the advantages - normalization is not in the original alg
			adv = rew_bat - value.detach()
			adv = (adv - adv.mean()) / (adv.std() + 1e-10)

			# Train the networks for user-defined epochs
			for _ in range(args.epoch):
				value, curr_log_probs = self.evaluate(state, act_bat, cov)

				# Calculate the pi theta ratio and surrogate losses
				ratio = torch.exp(curr_log_probs - log_probs_b)
				surr1 = ratio * adv
				surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv

				# Update actor network
				loss_act = (-torch.min(surr1, surr2)).mean()
				self.opt_act.zero_grad()
				loss_act.backward(retain_graph=True)
				self.opt_act.step()

				# Update critic network
				loss_crit = nn.MSELoss()(value, rew_bat)
				self.opt_crit.zero_grad()
				loss_crit.backward()
				self.opt_crit.step()

				event_log['act_loss'].append(loss_act.detach())
				event_log['crit_loss'].append(loss_crit.detach())

			output_progress(event_log)
			# from utils import save_models
			save_models (args, self.actor, self.critic, tstep_batch)

	def update_batch(self, args, event_log, cov):
		# Update the next batch of states and actions
		state, act_bat, log_probs_b, reward, rew_bat, rew_per_ep = [], [], [], [], [], []
		ep_len = 0

		i = 0
		# Iterate over user-defined timesteps / batch
		while i < args.tstep_batch:
			rew_per_ep = []
			next_state = self.env.reset()
			done = False

			# Iterate over max timestep per episode
			for j in range(args.tstep_ep):
				if args.img_rend and (event_log['tstep_batch'] % args.img_rend_freq == 0) and ep_len == 0:
					self.env.render()
				i += 1
				state.append(next_state) # original location of this append
				action, log_prob = self.get_action(next_state, cov)
				next_state, rew, done, _ = self.env.step(action)
				# state.append(obs) # This is for eval of calc_Gt_v1()
				rew_per_ep.append(rew)
				# reward.append(rew) # This is for eval of calc_Gt_v1()
				act_bat.append(action)
				log_probs_b.append(log_prob)
				if done:
					break
			ep_len += j
			reward.append(rew_per_ep)

		# Convert to tensor for actor/critic training
		state, act_bat, log_probs_b = torch.tensor(state, dtype=torch.float), \
		torch.tensor(act_bat, dtype=torch.float), torch.tensor(log_probs_b, dtype=torch.float)

		rew_bat = self.calc_Gt_v2(args, reward)
		event_log['reward'] = reward
		event_log['ep_len'] = ep_len

		return state, act_bat, log_probs_b, rew_bat, ep_len

    # More common, non-nested, calculation of return - code changes required to enable
	def calc_Gt_v1(self, args, reward):
		Gt = []
		R = 0
		for r in reward[::-1]:
			R = r + args.gamma * R
			Gt.insert(0, R)
		Gt = torch.tensor(Gt, dtype=torch.float)

    # Nested calculation of return
	def calc_Gt_v2(self, args, reward):
		Gt = []
		for i in reversed(reward):
			dis_rew = 0
			for rew in reversed(i):
				dis_rew = rew + dis_rew * args.gamma
				Gt.insert(0, dis_rew)
		Gt = torch.tensor(Gt, dtype=torch.float)
		return Gt

	def get_action(self, obs, cov):
		mu = self.actor(obs)
		prob = MultivariateNormal(mu, cov)
		action = prob.sample()
		log_prob = prob.log_prob(action)
		return action.detach().numpy(), log_prob.detach()

	def evaluate(self, state, act_bat, cov):
		value = self.critic(state).squeeze()
		mu = self.actor(state)
		prob = MultivariateNormal(mu, cov)
		log_probs = prob.log_prob(act_bat)
		return value, log_probs

