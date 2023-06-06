
# Output the length, return and episode number
def output_return(len, ret, num):
		print(f"Episode number: {num}")
		print(f"Episodic length: {len}")
		print(f"Episodic return: {ret}\n")

# Execute one full episode, calculate return and len of the episode
def proc_episode(args, policy, env):
	done = False
	next_state = env.reset()
	len, ret = 0, 0,  # Epis. len and return

	while not done:
		len += 1
		if args.img_rend:
			env.render()

		# Generate and execute next action
		action = policy(next_state).detach().numpy()
		next_state, rew, done, _ = env.step(action)

		# Aggregate reward
		ret += rew
	return len, ret

# Execute user-defined number of episodes using a trained model
def test_model(args, policy, env):
	i, len, ret = 0, 0, 0

	for i in range(args.img_rend_freq): # Using img_rend_freq as stopping criteria 
		len, ret = proc_episode(args, policy, env)
		output_return(len=len, ret=ret, num=i)
