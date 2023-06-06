import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
	def __init__(self, d_l1, d_l_final, d_hidden):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(d_l1, d_hidden)
		self.l2 = nn.Linear(d_hidden, d_hidden)
		self.l3 = nn.Linear(d_hidden, d_hidden)
		self.l4 = nn.Linear(d_hidden, d_l_final)

	def forward(self, x):
		if isinstance(x, np.ndarray):
			x = torch.tensor(x, dtype=torch.float)

		act1 = F.relu(self.l1(x))
		act2 = F.relu(self.l2(act1))
		act3 = F.relu(self.l3(act2))
		prob = self.l4(act3)

		return prob
	
class Critic(nn.Module):
	def __init__(self, d_l1, d_l_final, d_hidden):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(d_l1, d_hidden)
		self.l2 = nn.Linear(d_hidden, d_hidden)
		self.l3 = nn.Linear(d_hidden, d_hidden)
		self.l4 = nn.Linear(d_hidden, d_l_final)

	def forward(self, x):
		if isinstance(x, np.ndarray):
			x = torch.tensor(x, dtype=torch.float)

		act1 = F.relu(self.l1(x))
		act2 = F.relu(self.l2(act1))
		act3 = F.relu(self.l3(act2))
		val = self.l4(act3)

		return val