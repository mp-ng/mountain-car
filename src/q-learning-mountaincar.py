import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95 # importance of future rewards over current rewards
EPISODES = 25000

SHOW_EVERY = 5000
STATS_EVERY = 100

DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5 # how frequent we want to explore / take random steps
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) # (20, 20, 3)

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(int))


for episode in range(EPISODES + 1):

	episode_reward = 0

	if episode % SHOW_EVERY == 0:
		print(episode)
		render = True
	else:
		render = False

	discrete_state = get_discrete_state(env.reset())
	done = False

	while not done:

		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)

		new_state, reward, done, _ = env.step(action)
		episode_reward += reward
		new_discrete_state = get_discrete_state(new_state)

		if render:
			env.render()

		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action, )] = new_q

		elif new_state[0] >= env.goal_position:
			# print(f"We made it on episode {episode}")
			q_table[discrete_state + (action, )] = 0

		discrete_state = new_discrete_state

	if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
		epsilon -= epsilon_decay_value

	ep_rewards.append(episode_reward)

	if episode % STATS_EVERY == 0:
		recent_eps = ep_rewards[-STATS_EVERY:]
		average_reward = sum(recent_eps) / len(recent_eps)
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(recent_eps))
		aggr_ep_rewards['max'].append(max(recent_eps))

		print(f"Episode: {episode} avg: {average_reward} min: {min(recent_eps)} max: {max(recent_eps)}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend()
plt.grid(True)
plt.show()
