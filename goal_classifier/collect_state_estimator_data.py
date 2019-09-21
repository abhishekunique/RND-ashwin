import argparse
import numpy as np
import dsuite
import gym
from dsuite.dclaw.turn import DClawTurnImage, DClawTurnFixed
from softlearning.environments.adapters.gym_adapter import GymAdapter
import os
import skimage
import pickle
import gzip
from softlearning.models.state_estimation import normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-trajectories',
                        type=int,
                        default=500,
                        help='Number of trajectories to collect')
    parser.add_argument('--save-path-name',
                        type=str,
                        default='screw_data',
                        help='Save directory name')
    parser.add_argument('--rollout-length',
                        type=int,
                        default=100,
                        help='Number of timesteps per rollout')
    parser.add_argument('--dump-frequency',
                        type=int,
                        default=50,
                        help='Number of trajectories per dump')
    parser.add_argument('--image-shape',
                        type=lambda x: eval(x),
                        default=(32, 32, 3),
                        help='(width, height, channels) to save for pixels')
    parser.add_argument('--save-images',
                        type=lambda x: eval(x),
                        default=False,
                        help='Whether or not to save images while collecting')

    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(cur_dir, args.save_path_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    NUM_TOTAL_TRAJECTORIES = args.num_trajectories
    ROLLOUT_LENGTH = args.rollout_length
    DUMP_FREQUENCY = args.dump_frequency

    image_shape = args.image_shape
    save_images = args.save_images

    trajectories = []
    trajectories_since_last_dump = []

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
            'distance': 0.4,
            'elevation': -65,
            'lookat': (1e-3, 0, -1e-2),
        },
        'init_qpos_range': (
            (-0.075, -0.075, 0, 0, 0, -np.pi),
            (0.075, 0.075, 0, 0, 0, np.pi),
        ),
        'observation_keys': (
            'pixels',
            'claw_qpos',
            # 'last_action'
        ),
    }
    env = GymAdapter(
        domain='DClaw',
        task='TurnFreeValve3Fixed-v0',
        **env_kwargs
    )

    # reset the environment
    for n_trajectory in range(NUM_TOTAL_TRAJECTORIES):
        # All the observations keys will be added below
        trajectory = {
            'actions': [],
            'states': [],
        }
        env.reset()

        t = 0
        while t < ROLLOUT_LENGTH:
            # 1. Collect and perform actions (sampled uniformly)
            action = env.action_space.sample()

            # env.render()  # render on display
            obs_dict = env.get_obs_dict()

            # 2. Collect observations (including claw position and pixels)
            observation, _, _, _ = env.step(action)
            # Add to all the observation keys
            for k, v in observation.items():
                if k not in trajectory:
                    trajectory[k] = []
                trajectory[k].append(v)
            trajectory['actions'].append(action)

            # 3. Calculate the ground truth state
            xy = normalize(obs_dict['object_xy_position'], -0.1, 0.1, -1, 1)
            cos, sin = (
                obs_dict['object_orientation_cos'][2],
                obs_dict['object_orientation_sin'][2]
            )
            ground_truth_state = np.concatenate([
                xy,
                cos[None],
                sin[None]
            ])
            trajectory['states'].append(ground_truth_state)

            # Save an image if the flag is True
            if save_images:
                skimage.io.imsave(
                    os.path.join(directory, f'img{t}.png'),
                    observation['pixels']
                )
            t += 1

        # Concat everything nicely
        for k, v in trajectory.items():
            trajectory[k] = np.stack(v)

        trajectories.append(trajectory)
        trajectories_since_last_dump.append(trajectory)

        if n_trajectory % 20 == 0:
            print(f"\n{n_trajectory} trajectories collected...")

        if n_trajectory > 0 and n_trajectory % DUMP_FREQUENCY == 0:
            print('DUMPING DATA... total # trajectories:', n_trajectory)
            with gzip.open(os.path.join(directory, f'data_{n_trajectory}.pkl'), 'wb') as f:
                pickle.dump(trajectories_since_last_dump, f)
            trajectories_since_last_dump = []

    import ipdb; ipdb.set_trace()
    # Save everything

    # TODO: Fix to make one dictionary of np arrays instead of one array of many dicts
    with gzip.open(os.path.join(directory, f'data.pkl'), 'wb') as f:
        pickle.dump(trajectories, f)


if __name__ == "__main__":
    main()