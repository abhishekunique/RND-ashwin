import argparse
import numpy as np
import dsuite
import gym
from dsuite.dclaw.turn import DClawTurnImage, DClawTurnFixed
from softlearning.environments.adapters.gym_adapter import GymAdapter
import os
import imageio
import pickle
import gzip
from softlearning.models.state_estimation import normalize

cur_dir = os.path.dirname(os.path.realpath(__file__))
directory = cur_dir + "/free_screw_state_estimator_data_invisible_claw_11"
# directory = cur_dir + "/antialiasing"

if not os.path.exists(directory):
    os.makedirs(directory)

def main():
    NUM_TOTAL_EXAMPLES, ROLLOUT_LENGTH, STEPS_PER_SAMPLE = 50000, 50, 4
    SAVE_FREQ = 50000
    pixels = []
    states = []
    images = False
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
        'camera_settings': {
            'azimuth': 180,
            'distance': 0.35,
            'elevation': -55,
            'lookat': (0, 0, 0.03),
        },
        'init_angle_range': (-np.pi, np.pi),
        'init_x_pos_range': (-0.075, 0.075),
        'init_y_pos_range': (-0.075, 0.075),
        'target_angle_range': (np.pi, np.pi),
        'observation_keys': (
            'pixels',
            'claw_qpos',
            'last_action'),
    }
    env = GymAdapter(
        domain='DClaw',
        # task='TurnFreeValve3ResetFreeSwapGoal-v0',
        task='TurnFreeValve3Fixed-v0',
        **env_kwargs
    )

    num_positives = 0

    # reset the environment
    while num_positives <= NUM_TOTAL_EXAMPLES:
        observation = env.reset()
        # print("Resetting environment...")
        t = 0
        while t < ROLLOUT_LENGTH:
            action = env.action_space.sample()
            for _ in range(STEPS_PER_SAMPLE):
                observation, _, _, _ = env.step(action)

            # env.render()  # render on display
            obs_dict = env.get_obs_dict()

            pixels_obs = observation['pixels']
            pixels.append(pixels_obs)
            xy = normalize(obs_dict['object_xy_position'], -0.1, 0.1, -1, 1)
            cos, sin = obs_dict['object_orientation_cos'][2], obs_dict['object_orientation_sin'][2]
            angle = np.arctan2(sin, cos)
            # print("OBJECT XY: ", xy)
            # print("ANGLE: ", angle * 180 / np.pi)
            ground_truth_state = np.concatenate([
                xy,
                cos[None],
                sin[None]
            ])
            states.append(ground_truth_state)
            if images:
                imageio.imwrite(directory + '/img%i.png' % num_positives, pixels_obs)
            num_positives += 1
            if num_positives % 1000 == 0:
                print("\n---", num_positives, "---")
            t += 1

            if num_positives % 1000 == 0:
                print('DUMPING DATA... total # examples:', num_positives)
                pixels_concat = np.stack(pixels, axis=0)
                states_concat = np.stack(states, axis=0)
                dump_dict = {
                    'pixels': pixels_concat,
                    'states': states_concat
                }
                with gzip.open(directory + '/data.pkl', 'wb') as f:
                    pickle.dump(dump_dict, f)

if __name__ == "__main__":
    main()
