import numpy as np
import os
import pickle

goal_directory = os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '..', '..')) + '/goal_pools/'

# expects a list of paths for each of the goals
GOAL_POOL_PATHS_PER_ENV = {
    # Needs to be 180, 0, since the first goal is 180
    'TurnMultiGoalResetFree-v0': (f'fixed_screw_multigoal_{goal}/' for goal in [180, 0]),
}

def get_example_pools_from_variant(variant):
    task = variant['task']

    goal_example_pools_train, goal_example_pools_validation = [], []
    n_goal_examples = variant['data_params']['n_goal_examples']

    if task in GOAL_POOL_PATHS_PER_ENV:
        file_paths = [goal_directory + path + 'positives.pkl' for path in GOAL_POOL_PATHS_PER_ENV[task]]
        for file_path in file_paths:
            with open(file_path, 'rb') as file:
                goal_examples = pickle.load(file)
                goal_example_pools_train.append({
                    key: goal_examples[key][:n_goal_examples]
                    for key in goal_examples.keys()
                })
                goal_example_pools_validation.append({
                    key: goal_examples[key][n_goal_examples:]
                    for key in goal_examples.keys()
                })
    else:
        raise NotImplementedError

    return goal_example_pools_train, goal_example_pools_validation
