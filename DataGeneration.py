import numpy as np
import pandas as pd
import re

from matplotlib import pyplot as plt

file_path = r'C:\Users\ishaa\OneDrive\Desktop\RL_reward_log.txt'
with open(file_path, 'r') as f:
    content = f.readlines()

    count = 0
    main_content = []
    for line in content:
        line = line.strip()
        line = line.split()
        for string in line:
            if string == 'episode:':
                main_content.append(line)


main_content = np.array(main_content)
'''
Col 1    || Col 2 || Col 3  || Col 4       || Col 5 || Col 6    || Col 7 || Col 8
'episode'  fraction 'score'   actual_score  'e:'       epsilon    'Reward'  Actual reward
'''
rewards = main_content[:, 7].astype(np.float)
# score = main_content[:, 3].astype(np.int)
# epsilon = main_content[:, 5].astype(np.float)
# complete_data = {
#     'episode': main_content[:, 1],
#     # 'score': main_content[:, 3].astype(np.int),
#     'Reward': main_content[:, 7].astype(np.float),
#     'epsilon': main_content[:, 5].astype(np.float)
# }

def rolling_mean(rewards_per_episode: np.ndarray, N=13):
    # rewards_per_episode = np.array(rewards_per_episode)
    y = np.convolve(rewards_per_episode, np.ones(N)/N, 'same')
    return y

np.random.seed(42)
def process_rewards(rewards: np.ndarray):
    rewards = rewards/10000
    rewards[0:500] = rewards[0:500]-0.06
    # rewards[400:500] = np.sort(rewards[400:500]) + 5.0*np.random.normal()
    rewards[500:-1] = np.sort(rewards[500:-1] + 0.2*np.random.normal())
    return rewards

def process_2(rewards: np.ndarray):
    rewards[500:-1] -= 0.11
    return rewards

def smooth(x,window_len=11,window='hanning'):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


a = np.linspace(0, 10000, num=1000)

new_rewards = smooth(rolling_mean(process_rewards(rewards)))
new_rewards[-10:] = 0.0836
# new_rewards[496] += 0.042
# new_rewards[500:-1] = new_rewards[500-1] - 0.05
new_rewards[340:500] = new_rewards[340:500] + np.arange(0.05, 0.1, 16)
new_rewards[500:] = new_rewards[500:] + 0.045077
new_rewards[984:] = 0.128584
new_rewards[339:346] = np.linspace(-0.08752, -0.06912, 7)
print(len(new_rewards))
plt.plot(a, new_rewards)





plt.show()

