import numpy as np
import os
import pickle

goal_directory = os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '..', '..')) + '/goal_pools/'

# expects a list of paths for each of the goals
# TODO: Split up by vision/state experiments
GOAL_POOL_PATHS_PER_ENV_PER_NUM_GOALS = {
    # Needs to be 180, 0, since the first goal is 180
    'TurnMultiGoalResetFree-v0': {
        '2': (f'fixed_screw_2_goals_{goal}/' for goal in [180, 0]), 
    },
    'TurnFreeValve3MultiGoalResetFree-v0': {
        '2': (f'free_screw_2_goals_less_tiny_box_{goal}/' for goal in (180, 0)),
        # '2': (f'free_screw_2_goals_regular_box_state_{goal}/' for goal in (180, 0)),
        '4': (f'free_screw_4_goals_regular_box_state_{goal}/' for goal in (0, 90, 180, -90)),
    },
    # 'TurnFreeValve3MultiGoalResetFree-v0': {
    #     '2': (f'free_screw_2_goals_less_tiny_box_state_{goal}/' for goal in (180, 0)),
    # },
    'TurnFreeValve3MultiGoal-v0': {
        # '2': (f'free_screw_2_goals_tiny_box_{goal}/' for goal in (180, 0)),
        '2': (f'free_screw_2_goals_regular_box_{goal}/' for goal in (180, 0)),
    },

    # 'TurnMultiGoalResetFree-v0': (f'fixed_screw_multigoal_{goal}/' for goal in [180, 0]),
    # 'TurnMultiGoalResetFree-v0': (f'fixed_screw_multigoal_{goal}/' for goal in [120, 240, 0]),
    # 'TurnMultiGoalResetFree-v0': (f'fixed_screw_5_goals_{goal}/' for goal in [72, 144, 216, 288, 0]),
    # 'TurnMultiGoalResetFree-v0': (f'fixed_screw_4_goals_{goal}/' for goal in [0, 90, 180, 270]),

}

def get_example_pools_from_variant(variant):
    task = variant['environment_params']['training']['task']
    num_goals = variant['num_goals']

    goal_example_pools_train, goal_example_pools_validation = [], []
    n_goal_examples = variant['data_params']['n_goal_examples']
    if task in GOAL_POOL_PATHS_PER_ENV_PER_NUM_GOALS:
        directories = GOAL_POOL_PATHS_PER_ENV_PER_NUM_GOALS[task][f'{num_goals}']
        file_paths = [
            os.path.join(goal_directory, path, 'positives.pkl')
            for path in directories
        ]
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                goal_examples = pickle.load(f)
                total_samples = len(goal_examples[next(iter(goal_examples))])

                # Shuffle the goal images before assigning training/validation
                shuffle = np.random.permutation(total_samples)
                training_indices = shuffle[:n_goal_examples]
                validation_indices = shuffle[n_goal_examples:]

                goal_example_pools_train.append({
                    key: goal_examples[key][training_indices]
                    for key in goal_examples.keys()
                })
                goal_example_pools_validation.append({
                    key: goal_examples[key][validation_indices]
                    for key in goal_examples.keys()
                })
    else:
        raise NotImplementedError

    return goal_example_pools_train, goal_example_pools_validation
