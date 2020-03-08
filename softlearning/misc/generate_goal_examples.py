import numpy as np

from softlearning.environments.utils import get_environment_from_params
import pickle
import os

from softlearning.misc.utils import PROJECT_PATH
goal_directory = os.path.join(PROJECT_PATH, 'goal_classifier')
print(goal_directory)

PICK_TASKS = [
    'StateSawyerPickAndPlaceEnv-v0',
    'Image48SawyerPickAndPlaceEnv-v0',
    'StateSawyerPickAndPlace3DEnv-v0',
    'Image48SawyerPickAndPlace3DEnv-v0',
]

DOOR_TASKS = [
    'StateSawyerDoorPullHookEnv-v0',
    'Image48SawyerDoorPullHookEnv-v0',
]

PUSH_TASKS = [
    'StateSawyerPushSidewaysEnv-v0',
    'Image48SawyerPushSidewaysEnv-v0',
    'StateSawyerPushForwardEnv-v0',
    'Image48SawyerPushForwardEnv-v0',
]

GOAL_IMAGE_PATH_PER_ENVIRONMENT = {
    'TurnFreeValve3ResetFree-v0': 'free_screw_180/',
    # 'TurnFreeValve3Fixed-v0': 'free_screw_180_regular_box_48/',
    'TurnFreeValve3Fixed-v0': 'free_screw_180/',
    'TurnFixed-v0': 'fixed_screw_180_no_normalization/',
    'TurnResetFree-v0': '/home/abhigupta/Libraries/vice/goal_classifier/fixed_screw_180/', #'fixed_screw_-75',
    'TurnFreeValve3Hardware-v0': 'fixed_screw_180', # Dummy goal images
    'TurnMultiGoalResetFree-v0': 'fixed_screw_2_goals_mixed_pool_goal_index/',
    'LiftDDResetFree-v0': 'dodecahedron_lifting_flat_bowl_arena_red',
    'SlideBeadsResetFree-v0': '4_beads_475',
}


def get_goal_example_from_variant(variant):
    train_env_params = variant['environment_params']['training']

    env = get_environment_from_params(train_env_params)
    total_goal_examples = variant['data_params']['n_goal_examples'] \
        + variant['data_params']['n_goal_examples_validation_max']

    task = train_env_params['task']

    if task in DOOR_TASKS:
        goal_examples = generate_door_goal_examples(total_goal_examples, env)
    elif task in PUSH_TASKS:
        goal_examples = generate_push_goal_examples(total_goal_examples, env)
    elif task in PICK_TASKS:
        goal_examples = generate_pick_goal_examples(total_goal_examples, env, variant['task'])
    elif task in GOAL_IMAGE_PATH_PER_ENVIRONMENT.keys():
        env_path = os.path.join(goal_directory, GOAL_IMAGE_PATH_PER_ENVIRONMENT[task])
        path = os.path.join(env_path, 'positives.pkl')
        with open(path, 'rb') as file:
            goal_examples = pickle.load(file)
    else:
        raise NotImplementedError

    n_goal_examples = variant['data_params']['n_goal_examples']
    total_samples = len(goal_examples[next(iter(goal_examples))])

    # Shuffle the goal images before assigning training/validation
    shuffle = np.random.permutation(total_samples)
    positive_indices = shuffle[:n_goal_examples]
    negative_indices = shuffle[n_goal_examples:]

    goal_examples_train = {
        key: goal_examples[key][positive_indices]
        for key in goal_examples.keys()
    }
    goal_examples_validation = {
        key: goal_examples[key][negative_indices]
        for key in goal_examples.keys()
    }

    return goal_examples_train, goal_examples_validation


def generate_pick_goal_examples(total_goal_examples, env, task_name):
    max_attempt = 50
    top_level_attempts = 10*total_goal_examples
    attempts = 0
    n = 0

    goal_examples = []
    gain = 5.0
    for _ in range(top_level_attempts):
        env.reset()

        for i in range(100):

            if '3D' in task_name:
                obj_xy = env.unwrapped.get_obj_pos()[:2]
                hand_xy = env.unwrapped.get_endeff_pos()[:2]
                goal_xy = env.unwrapped.fixed_goal[3:5]

                hand_obj_distance = np.linalg.norm(obj_xy - 0.02 - hand_xy)
                goal_obj_distance = np.linalg.norm(obj_xy - goal_xy)

                if i < 25:
                    if hand_obj_distance > 0.015:
                        action_xy = gain*(obj_xy - hand_xy)
                    else:
                        action_xy = [0., 0.]
                    action = np.asarray([action_xy[0], action_xy[1], 0., -1])
                elif i < 35:
                    action = np.asarray([0., 0, -1, -1.])
                elif i < 45:
                    action = np.asarray([0., 0, -1,  1.])
                elif i < 60:
                    action = np.asarray([0., 0, +1,  1.])
                elif i < 100:
                    if goal_obj_distance > 0.015:
                        action_xy = gain*(goal_xy - obj_xy)
                    else:
                        action_xy = [0., 0.]
                    action = np.asarray([action_xy[0], action_xy[1], 0., 1.])

            else:

                obj_y = env.unwrapped.get_obj_pos()[1] - 0.02
                hand_y = env.unwrapped.get_endeff_pos()[1]
                goal_y = env.unwrapped.fixed_goal[4]

                if i < 25:
                    if obj_y < (hand_y - 0.01):
                        action = np.asarray([-1., 0., -1.])
                    elif obj_y > (hand_y + 0.01):
                        action = np.asarray([1., 0., -1.])
                    else:
                        action = np.asarray([0., 0., -1.])
                elif i < 40:
                    action = np.asarray([0., -1.0, -1.])
                elif i < 60:
                    action = np.asarray([0., -1.0, 1.0])
                elif i < 80:
                    action = np.asarray([0., 1., 1.])
                elif i < 100:
                    if goal_y < (hand_y - 0.01):
                        action = np.asarray([-1., 0., 1.])
                    elif goal_y > (hand_y + 0.01):
                        action = np.asarray([1., 0., 1.])
                    else:
                        action = np.asarray([0., 0., 1.])

            ob, r, d, info = env.step(action)

        if info['obj_success']:
            goal_examples.append(ob)

        if len(goal_examples) >= total_goal_examples:
            break

    assert len(goal_examples) == total_goal_examples, (
        f'Could not generate enough goal examples: {len(goal_examples)}')
    goal_examples = {
        key: np.concatenate([
            goal_example[key][None] for goal_example in goal_examples
        ], axis=0)
        for key in goal_examples[0].keys()
    }

    return goal_examples


def generate_push_goal_examples(total_goal_examples, env):
    max_attempt = 5*total_goal_examples
    attempts = 0
    n = 0
    goal_examples = []

    while n < total_goal_examples and attempts < max_attempt:

        attempts += 1
        env.reset()
        goal_vec = {
            'state_desired_goal': env.unwrapped.fixed_goal
        }

        goal_vec['state_desired_goal'][:2] += np.random.uniform(low=-0.01, high=0.01, size=(2,))
        goal_vec['state_desired_goal'][-2:] += np.random.uniform(low=-0.01, high=0.01, size=(2,))

        env.unwrapped.set_to_goal(goal_vec)

        endeff_pos = env.unwrapped.get_endeff_pos()
        puck_pos = env.unwrapped.get_puck_pos()

        endeff_distance = np.linalg.norm(endeff_pos - goal_vec['state_desired_goal'][:3])
        puck_distance = np.linalg.norm(puck_pos[:2] - goal_vec['state_desired_goal'][3:5])
        puck_endeff_distance = np.linalg.norm(puck_pos[:2] - endeff_pos[:2])

        endeff_threshold = 0.05
        puck_threshold = env.unwrapped.indicator_threshold
        puck_radius = env.unwrapped.puck_radius

        if (endeff_distance < endeff_threshold
            and puck_distance < puck_threshold
            and puck_endeff_distance > puck_radius):
            ob, rew, done, info = env.step(np.asarray([0., 0.]))
            goal_examples.append(ob)
            n += 1

    assert len(goal_examples) == total_goal_examples, 'Could not generate enough goal examples'
    goal_examples = {
        key: np.concatenate([
            goal_example[key][None] for goal_example in goal_examples
        ], axis=0)
        for key in goal_examples[0].keys()
    }

    return goal_examples


def generate_door_goal_examples(total_goal_examples, env):

    max_attempt = 10 * total_goal_examples
    attempts = 0
    n = 0
    goal_examples = []

    while n < total_goal_examples and attempts < max_attempt:

        attempts += 1
        env.reset()
        env.unwrapped._set_door_pos(0 + np.random.uniform(low=0., high=0.1))
        goal_vec = {
            'state_desired_goal': env.unwrapped.fixed_goal
        }

        for j in range(100):

            door_angle = env.unwrapped.get_door_angle()
            if j < 25:
                act = [0.05, 1, -0.5]
            elif j < 100 and door_angle < 0.8:
                act = [0.0, -0.4, 0.0]
            else:
                act = [0., 0., 0.]

            act += np.random.uniform(low=-0.01, high=0.01, size=3)
            ob, rew, done, info = env.step(np.asarray(act))

        # goal_vec['state_desired_goal'][:3] += np.random.uniform(low=-0.01, high=0.01, size=(3,))
        # goal_vec['state_desired_goal'][3] += np.random.uniform(low=-0.01, high=0.01)

        # env.unwrapped.set_to_goal_pos(goal_vec['state_desired_goal'][:3])
        # env.unwrapped.set_to_goal_angle(goal_vec['state_desired_goal'][3])

        pos = env.unwrapped.get_endeff_pos()
        angle = env.unwrapped.get_door_angle()
        endeff_distance = np.linalg.norm(pos - goal_vec['state_desired_goal'][:3])
        angle_distance = np.abs(angle - goal_vec['state_desired_goal'][3])
        #state = np.concatenate([pos, angle])
        angle_threshold = env.unwrapped.indicator_threshold[0]
        endeff_threshold = env.unwrapped.indicator_threshold[1]

        # if endeff_distance < endeff_threshold and angle_distance < angle_threshold:
        if info['angle_success']:
            ob, rew, done, info = env.step(np.asarray([0., 0., 0.]))
            goal_examples.append(ob)
            n += 1

    assert len(goal_examples) == total_goal_examples, 'Could not generate enough goal examples'
    goal_examples = {
        key: np.concatenate([
            goal_example[key][None] for goal_example in goal_examples
        ], axis=0)
        for key in goal_examples[0].keys()
    }

    return goal_examples
