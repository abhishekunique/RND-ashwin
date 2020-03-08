import numpy as np
import os
import pickle

from softlearning.misc.utils import PROJECT_PATH
goal_directory = os.path.join(PROJECT_PATH, 'goal_pools')

# expects a list of paths for each of the goals
# TODO: Split up by vision/state experiments
GOAL_POOL_PATHS_PER_ENV_PER_NUM_GOALS = {
    'TurnFreeValve3ResetFree-v0': {
        '2': (
            f'free_screw_32x32/goal_{i}_{goal}'
            for i, goal in enumerate([-90, 90])
        ),
    },
    'TurnResetFree-v0': {
        '2': (
            f'fixed_screw/goal_{i}_{goal}'
            for i, goal in enumerate([-90, 90])
        ),
    },
    'SlideBeadsResetFree-v0': {
        '2': (
            f'4_beads/4_beads_{goal}'
            for i, goal in enumerate([475, 0])
        ),
    },

    'TurnMultiGoalResetFree-v0': {
        '2': (
            f'fixed_screw_2_goals/goal_{i}_{goal}'
            for i, goal in enumerate([-90, 90])
        ),
        '4': (),
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
    'TurnFreeValve3ResetFreeSwapGoal-v0': {
        # '2': (f'free_screw_2_goals_bowl_{goal}/' for goal in (90, -90)),
        # '2': (
        #     f'free_screw_2_goals/goal_{i}_{goal}'
        #     for i, goal in enumerate([-90, 90])
        # ),
        '2': (
            f'free_screw_2_goals_visible_claw/goal_{i}_{goal}'
            for i, goal in enumerate([-90, 90])
        ),
    },
    'TurnFreeValve3Hardware-v0': {
        # '2': (
        #     f'free_screw_2_goals/goal_{i}_{goal}'
        #     for i, goal in enumerate([-90, 90])
        # ),
        '2': (
            'free_screw_lighting_fix/'
            # 'free_screw_goal_images_black_box/'
            # 'free_screw_goal_images_black_box_more_friction/'
            for i, goal in enumerate([90, 90])
        ),
    },

    'LiftDDResetFree-v0': {
        '1': ('dodecahedron_lifting_bowl_arena/', ),
    },
    # 'SlideBeadsResetFree-v0': {
    #     # '2': ('2_beads_{goal}/' for goal in (np.array([0, 0]), np.array([-0.0875, 0.0875]))),
    #     '2': (f'4_beads_{goal}/' for goal in (
    #         np.array([0, 0, 0, 0]),
    #         np.array([-0.0475, -0.0475, 0.0475, 0.0475]))
    #     ),
    # },
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
