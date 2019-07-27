import argparse
import numpy as np
import dsuite
import gym
from dsuite.dclaw.turn import DClawTurnImage, DClawTurnFixed
from softlearning.environments.adapters.gym_adapter import GymAdapter
import os
import imageio
import pickle

cur_dir = os.path.dirname(os.path.realpath(__file__))
directory = cur_dir + "/fixed_screw_2_goals_mixed_pool_goal_index"

def main():
    # goals = np.arange(0, 360, 360 / 4)
    # goals = [120, 240, 0]
    mixed_goal_pool = True
    one_hot_goal_index = False
    images = True
    goals = [180, 0]
    num_goals = len(goals)

    image_shape = (32, 32, 3)
    NUM_TOTAL_EXAMPLES, ROLLOUT_LENGTH, STEPS_PER_SAMPLE = 200, 25, 5
    observations = []

    for goal_index, goal in enumerate(goals):
        if not mixed_goal_pool:
            observations = []  # reset the observations

        num_positives = 0
        goal_angle = np.pi / 180. * goal # convert to radians

        env_kwargs = {
            'camera_settings': {
                'azimuth': 0.,
                'distance': 0.32,
                'elevation': -45.18,
                'lookat': np.array([0.00047, -0.0005, 0.060])
            },
            'goals': (goal_angle,),
            'goal_collection': True,
            'init_object_pos_range': (goal_angle - 0.05, goal_angle + 0.05),
            'target_pos_range': (goal_angle, goal_angle),
            'pixel_wrapper_kwargs': {
                'pixels_only': False,
                'normalize': False,
                'render_kwargs': {
                    'width': image_shape[0],
                    'height': image_shape[1],
                    'camera_id': -1
                },
            },
            'swap_goals_upon_completion': True,
            'observation_keys': ('pixels', 'claw_qpos', 'last_action', 'goal_index'), # save goal index to mask in the classifier
        }
        env = GymAdapter(
            domain='DClaw',
            task='TurnMultiGoal-v0',
            **env_kwargs
        )

        if mixed_goal_pool:
            path = directory
        else:
            path = directory + str(goal)
        if not os.path.exists(path):
            os.makedirs(path)

        # reset the environment
        while num_positives <= NUM_TOTAL_EXAMPLES:
            observation = env.reset()
            print("Resetting environment...")
            t = 0
            while t < ROLLOUT_LENGTH:
                action = env.action_space.sample()
                for _ in range(STEPS_PER_SAMPLE):
                    observation, _, _, _ = env.step(action)

                #env.render()  # render on display
                obs_dict = env.get_obs_dict()
                # print("OBS DICT:", obs_dict)

                # For fixed screw
                object_target_angle_dist = obs_dict['object_to_target_angle_dist']

                ANGLE_THRESHOLD = 0.15
                if object_target_angle_dist < ANGLE_THRESHOLD:
                    # Add observation if meets criteria
                    if one_hot_goal_index: 
                        one_hot = np.zeros(num_goals).astype(np.float32)
                        one_hot[goal_index] = 1.
                        observation['goal_index'] = one_hot
                    else:
                        observation['goal_index'] = np.array([goal_index])
                    observations.append(observation)
                    print(observation)
                    if images:
                        img_obs = observation['pixels']
                        imageio.imwrite(path + f'/img_{goal}_{num_positives}.jpg', img_obs)
                    num_positives += 1
                t += 1

        goal_examples = {
            key: np.concatenate([
                obs[key][None] for obs in observations
            ], axis=0)
            for key in observations[0].keys()
        }

        with open(path + '/positives.pkl', 'wb') as file:
            pickle.dump(goal_examples, file)


if __name__ == "__main__":
    main()
