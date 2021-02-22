from DQNAgent import DQNAgent
from Network import Network
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



BS_PARAMS = [{'capacity_bandwidth': 500000000, 'coverage': 500,
                'ratios': {'emBB': 0.5, 'mMTC': 0.4, 'URLLC': 0.1},
                'x': 500, 'y': 500}
                ]

SLICE_PARAMS = {'emBB': {
                'delay_tolerance': 10,
                'qos_class': 5,
                'bandwidth_guaranteed': 0,
                'bandwidth_max': 100000,
                'client_weight': 0.45,
                'threshold': 0,
                'usage_pattern': {'distribution': 'randint', 'params': (4000000, 800000000)}
                },
                'mMTC': {
                'delay_tolerance': 10,
                'qos_class': 2,
                'bandwidth_guaranteed': 100000,
                'bandwidth_max': 10000,
                'client_weight': 0.3,
                'threshold': 0,
                'usage_pattern': {'distribution': 'randint', 'params': (800000, 8000000)}
                },
                'URLLC': {
                'delay_tolerance': 10,
                'qos_class': 1,
                'bandwidth_guaranteed': 500000,
                'bandwidth_max': 15000,
                'client_weight': 0.25,
                'threshold': 0,
                'usage_pattern': {'distribution': 'randint', 'params': (800, 8000000)}
                }}

CLIENT_PARAMS = {'location':{'x': {'distribution': 'randint', 'params': (0, 1000)}, 'y': {'distribution': 'randint', 'params': (0, 1000)}}
                , 'usage_frequency': {'distribution': 'randint', 'params': (0, 100000), 'divide_scale': 1000000}}
NUM_CLIENTS = 1000
EPISODES = 1000

def log_all_info(file_name: str, metrics: dict):
    f = open(file_name, "a")
    msg = metrics
    f.write(msg)
    f.close()

def plot_func(episode_info: list):
    emBB = []
    mMTC = []
    URLLC = []
    for batch_info in episode_info:
        emBB.append(batch_info['emBB'][0])
        mMTC.append(batch_info['mMTC'][0])
        URLLC.append(batch_info['URLLC'][0])

    fig = plt.figure(2, figsize=(10, 10))
    ax_1 = plt.subplot(311)
    ax_2 = plt.subplot(312)
    ax_3 = plt.subplot(313)

    ax_1.title.set_text('Connected Clients in  eMBB')
    ax_2.title.set_text('Connected Clients in  mMTC')
    ax_3.title.set_text('Connected Clients in  URLLC')

    ax_1.plot(emBB)
    ax_2.plot(mMTC)
    ax_3.plot(URLLC)
    return fig

def rolling_mean(rewards_per_episode: list, N=13):
    rewards_per_episode = np.array(rewards_per_episode)
    y = np.convolve(rewards_per_episode, np.ones(N)/N, 'same')
    return y

def generate_metrics(rewards_per_episode: list, time_taken_per_episode: list):
    metrics = dict()
    mean_reward = np.mean(np.array(rewards_per_episode))
    reward_variance = np.var(np.array(rewards_per_episode))
    mean_time = np.mean(np.array(time_taken_per_episode))
    time_variance = np.var(np.array(time_taken_per_episode))
    metrics['MEAN_REWARD'] = mean_reward
    metrics['REWARD_VARIANCE'] = reward_variance
    metrics['MEAN_TIME'] = mean_time
    metrics['TIME_VAR'] = time_variance
    return metrics

if __name__ == "__main__":
    env = Network(bs_params=BS_PARAMS, slice_params=SLICE_PARAMS, client_params=CLIENT_PARAMS)
    rewards_per_episode = []
    time_taken_per_episode = []
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 700
    episode_info = []
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        reward_per_episode = 0
        for time in range(500):
            # env.render()
            action = agent.act(state)

            next_state, action, reward, done, info = env.step(action)

            reward = reward
            try:
                next_state = np.reshape(next_state, [1, state_size])
            except ValueError:
                continue
            agent.memorize(state, action, reward, next_state, done)
            reward_per_episode += reward
            state = next_state
            if done:
                print("Done: True\n")
                rewards_per_episode.append(reward_per_episode)      ## Average reward for one episode
                time_taken_per_episode.append(time)
                episode_info.append(info)
                print("episode: {}/{}, score: {}, e: {:.4}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)


    metrics = generate_metrics(rewards_per_episode, time_taken_per_episode)
    print(metrics)
    # log_all_info('Info.txt', metrics)
    fig = plt.figure(1, figsize=(10, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)


    ax1.title.set_text('Reward/Episode')
    ax2.title.set_text('Time Taken/Episode')


    ax1.plot(rewards_per_episode)
    ax1.plot(rolling_mean(rewards_per_episode))
    ax2.plot(time_taken_per_episode)


    fig2 = plot_func(episode_info)

    plt.show()


    # print("Rewards: ", rewards_per_episode)
    # print("Time taken:", time_taken_per_episode)




    

        
               
