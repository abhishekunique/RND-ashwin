print("STARTED")
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import gzip
import tensorflow as tf
import glob
import math
import skimage
from matplotlib.patches import Rectangle
import json

tf.enable_eager_execution()

from softlearning.replay_pools.utils import get_replay_pool_from_variant

def get_grid_vals(env, n_samples):
    n_samples = 50
    obs_space = env.observation_space['state_observation']
    xs = np.linspace(obs_space.low[0], obs_space.high[0], n_samples)
    ys = np.linspace(obs_space.low[1], obs_space.high[1], n_samples)
    xys = np.meshgrid(xs, ys)
    return np.array(xys).transpose(1, 2, 0).reshape((n_samples * n_samples, 2)), xys

def get_replay_pool(checkpoint, checkpoint_dir):
    from softlearning.replay_pools.utils import get_replay_pool_from_variant

    variant = checkpoint['variant']
    train_env = checkpoint['training_environment']
    replay_pool = get_replay_pool_from_variant(variant, train_env)

    replay_pool_path = os.path.join(checkpoint_dir, 'replay_pool.pkl')
    replay_pool.load_experience(replay_pool_path)
    return replay_pool

def plot_trajectories(checkpoint, checkpoint_dir, num_trajectories=10):
    replay_pool = get_replay_pool(checkpoint, checkpoint_dir)
    trajectories = replay_pool.last_n_batch(100 * num_trajectories)['observations']['state_observation'] \
                    .reshape(num_trajectories, 100, -1)
    for i in range(num_trajectories):
        plt.plot(trajectories[i,:,0], trajectories[i,:,1], color='w', linewidth=1)



common_dir = '/home/abhigupta/ray_results'
universe = 'gym'
domain = 'Point2D'
task = 'Maze-v0'

checkpoint_to_analyze = 50

base_path = os.path.join(common_dir, universe, domain, task)
exps = sorted(list(glob.iglob(os.path.join(base_path, '*'))))
for i, exp in enumerate(exps):
    print(f'{i} \t {exp.replace(base_path, "")}')

exp_choice = int(input('\n Which experiment do you want to analyze? (ENTER A NUMBER) \t'))

exp_path = exps[exp_choice]
print('\n')
seeds = sorted(list(glob.iglob(os.path.join(exp_path, '*'))))
seeds = [seed for seed in seeds if os.path.isdir(seed)]
checkpoints = []
for i, seed in enumerate(seeds):
    #     print(f'{i} \t {seed.replace(exp_path, "")}')
    checkpoint_dir = os.path.join(seed, f'checkpoint_{checkpoint_to_analyze}')
    try:
        with open(os.path.join(seed, f'{checkpoint_dir}/checkpoint.pkl'), 'rb') as f:
            checkpoint = pickle.load(f)
            checkpoints.append([checkpoint, checkpoint_dir])
    except Exception as e:
        print("ERROR")
        print(e)

print(f"Loaded {len(checkpoints)} seeds")

checkpoint_num = 50

checkpoints = []

for i, seed in enumerate(seeds):
    checkpoint_dir = os.path.join(seed, f'checkpoint_{checkpoint_num}')
    try:
        with open(os.path.join(seed, f'{checkpoint_dir}/checkpoint.pkl'), 'rb') as f:
            checkpoint = pickle.load(f)
            checkpoints.append((checkpoint, checkpoint_dir))
    except:
        pass
print(f"Loaded {len(checkpoints)} seeds")

n_plots = len(checkpoints)
n_columns = int(np.sqrt(n_plots) + 1)
n_rows = np.ceil(n_plots / n_columns)
plt.figure(figsize=(5 * n_columns, 5 * n_rows))
print(n_rows, n_columns)

import IPython
IPython.embed()
