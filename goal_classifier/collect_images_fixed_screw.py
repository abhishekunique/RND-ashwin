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
directory = cur_dir + "/dsuite_fixed_screw_180"
if not os.path.exists(directory):
    os.makedirs(directory)

# TODO: Make this a generic script, taking in an env and a condition for being a goal image.

def main():
    num_positives = 0
    NUM_TOTAL_EXAMPLES, ROLLOUT_LENGTH, STEPS_PER_SAMPLE = 100, 25, 5
    goal_angle = np.pi
    observations = []
    images = False 
    image_shape = (32, 32, 3)

    """
    env = DClawTurnFixed(
        init_angle_range=(goal_angle, goal_angle),
        target_angle_range=(goal_angle, goal_angle),
        observation_keys=('image',),
        image_shape=(32, 32, 3),
        camera_settings={
            'azimuth': 90.18582,
            'distance': 0.32,
            'elevation': -32.42,
            'lookat': np.array([-0.00157929, 0.00336185, 0.10151641])
        }
    ) 
    """

    env_kwargs = {
        'camera_settings': {
            'azimuth': 90.18582,
            'distance': 0.32,
            'elevation': -32.42,
            'lookat': np.array([-0.00157929, 0.00336185, 0.10151641])
        },
        'init_angle_range': (goal_angle - 0.05, goal_angle + 0.05),
        'target_angle_range': (goal_angle, goal_angle),
        'pixel_wrapper_kwargs': {
            'pixels_only': True,
            'render_kwargs': {
                'width': 32,
                'height': 32,
                'camera_id': -1
            }
        }
    }
    env = GymAdapter(
        'DClaw',
        'TurnFixed-v0',
        **env_kwargs
    )

    # reset the environment
    while num_positives <= NUM_TOTAL_EXAMPLES:
        observation = env.reset()
        print("Resetting environment...")
        t = 0
        while t < ROLLOUT_LENGTH:
            action = env.action_space.sample()
            for _ in range(STEPS_PER_SAMPLE):
                observation, _, _, _ = env.step(action)

            print("OBSERVATION:", observation)
            # env.render()  # render on display
            obs_dict = env.get_obs_dict()
            print("OBS DICT:", obs_dict)

            # For fixed screw
            object_target_angle_dist = obs_dict['object_to_target_angle_dist'] 

            ANGLE_THRESHOLD = 0.15
            if object_target_angle_dist < ANGLE_THRESHOLD:
                # Add observation if meets criteria
                observations.append(observation)
                if images:
                    img_obs = observation['pixels']
                    image = img_obs[:np.prod(image_shape)].reshape(image_shape)
                    print_image = (image + 1.) * 255. / 2.
                    imageio.imwrite(directory + '/img%i.jpg' % num_positives, print_image)
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
