import argparse
import numpy as np
import dsuite
import gym
from softlearning.environments.adapters.gym_adapter import GymAdapter
import os
import pickle
import skimage

cur_dir = os.path.dirname(os.path.realpath(__file__))
# directory = cur_dir + "/free_screw_180_regular_box_32"
directory = cur_dir + "/free_screw_180"

if not os.path.exists(directory):
    os.makedirs(directory)

def main():
    num_positives = 0
    NUM_TOTAL_EXAMPLES, ROLLOUT_LENGTH, STEPS_PER_SAMPLE = 250, 25, 4
    goal_angle = np.pi
    observations = []
    images = True
    image_shape = (32, 32, 3)

    env_kwargs = {
        'pixel_wrapper_kwargs': {
            'pixels_only': False,
            'normalize': False,
            'render_kwargs': {
                'width': image_shape[0],
                'height': image_shape[1],
                'camera_id': -1,
            },
        },
        # 'camera_settings': {
        #     'distance': 0.5,
        #     'elevation': -60
        # },
        'camera_settings': {
            'azimuth': 180,
            'distance': 0.35,
            'elevation': -55,
            'lookat': np.array([0, 0, 0.03]),
        },
        'init_qpos_range': (
            (0, 0, 0, 0, 0, goal_angle - 0.05),
            (0, 0, 0, 0, 0, goal_angle + 0.05)
        ),
        'target_qpos_range': (
            (0, 0, 0, 0, 0, goal_angle),
            (0, 0, 0, 0, 0, goal_angle)
        ),
        'observation_keys': (
            'pixels',
            'claw_qpos',
            'last_action',
            'object_xy_position',
            'object_z_orientation_cos',
            'object_z_orientation_sin'
        ),
    }
    env = GymAdapter(
        domain='DClaw',
        task='TurnFreeValve3Fixed-v0',
        **env_kwargs
    )

    ANGLE_THRESHOLD, POSITION_THRESHOLD = 0.15, 0.035
    goal_criteria = lambda angle_dist, pos_dist: angle_dist < ANGLE_THRESHOLD \
        and pos_dist < POSITION_THRESHOLD

    # reset the environment
    while num_positives <= NUM_TOTAL_EXAMPLES:
        observation = env.reset()
        print("Resetting environment...")
        t = 0
        while t < ROLLOUT_LENGTH:
            action = env.action_space.sample()
            for _ in range(STEPS_PER_SAMPLE):
                observation, _, _, _ = env.step(action)
            # env.render()  # render on display
            obs_dict = env.get_obs_dict()

            circle_dist = obs_dict['object_to_target_circle_distance']
            pos_dist = obs_dict['object_to_target_position_distance']
            print(f"Circle dist: {circle_dist}, Position dist: {pos_dist}")

            if goal_criteria(circle_dist, pos_dist):
                # Add observation if meets criteria
                observations.append(observation)
                print(observation)
                if images:
                    img_obs = observation['pixels']
                    # img_0, img_1 = np.split(
                    #     img_obs,
                    #     indices_or_sections=2,
                    #     axis=2
                    # )
                    # concat_obs = np.concatenate([img_0, img_1], axis=1)
                    skimage.io.imsave(directory + f'/img_{num_positives}.png', img_obs)
                num_positives += 1
            t += 1

    goal_examples = {
        key: np.concatenate([
            obs[key][None] for obs in observations
        ], axis=0)
        for key in observations[0].keys()
    }

    with open(directory + '/positives.pkl', 'wb') as file:
        pickle.dump(goal_examples, file)


if __name__ == "__main__":
    main()
