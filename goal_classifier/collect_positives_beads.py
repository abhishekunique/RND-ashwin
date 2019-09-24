import numpy as np
import dsuite
import gym
from softlearning.environments.adapters.gym_adapter import GymAdapter
import os
import imageio
import pickle

cur_dir = os.path.dirname(os.path.realpath(__file__))
# directory = cur_dir + "/2_beads_"
directory = cur_dir + "/4_beads_"


def main():
    # goals = [np.array((0, 0)), np.array((-0.0875, 0.0875))]
    goals = [
        np.array([-0.0475, -0.0475, 0.0475, 0.0475]),
        np.array([0, 0, 0, 0])
    ]
    for goal_index, goal in enumerate(goals):
        num_positives = 0
        NUM_TOTAL_EXAMPLES, ROLLOUT_LENGTH, STEPS_PER_SAMPLE = 200, 25, 4
        POSITION_THRESHOLD = 0.015
        observations = []
        images = True
        image_shape = (32, 32, 3)

        goal_keys = (
            'pixels',
            # 'object_position',
            # 'object_orientation_sin',
            # 'object_orientation_cos',
            # 'goal_index',
        )

        env_kwargs = {
            'pixel_wrapper_kwargs': {
                'pixels_only': False,
                'normalize': False,
                'render_kwargs': {
                    'width': image_shape[0],
                    'height': image_shape[1],
                    'camera_id': -1
                },
            },
            # 'camera_settings': {
            #     'azimuth': 23.234042553191497,
            #     'distance': 0.2403358053524018,
            #     'elevation': -29.68085106382978,
            #     'lookat': (-0.00390331,  0.01236683,  0.01093447),
            # },
            'camera_settings': {
                'azimuth': 90,
                'lookat': (0,  0.04581637, -0.01614516),
                'elevation': -45,
                'distance': 0.37,
            },
            'target_qpos_range': [goal],
            'observation_keys': goal_keys,
            'init_qpos_range': (goal - 0.01, goal + 0.01),
            # 'num_objects': 2,
            'num_objects': 4,
        }
        env = GymAdapter(
            domain='DClaw',
            task='SlideBeadsFixed-v0',
            **env_kwargs
        )

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

                # env.render()  # render on display
                obs_dict = env.get_obs_dict()
                # print("OBS DICT:", obs_dict)

                # print(obs_dict['object_to_target_circle_distance'], obs_dict['object_to_target_position_distance'])
                if np.max(obs_dict['objects_to_targets_distances']) < POSITION_THRESHOLD:
                # if obs_dict['object_to_target_circle_distance'] < ANGLE_THRESHOLD and obs_dict['object_to_target_position_distance'] < POSITION_THRESHOLD:
                    # Add observation if meets criteria
                    observation['goal_index'] = np.array([goal_index]).astype(np.float32)
                    observations.append(observation)
                    print(observation)
                    num_positives += 1

                    if images:
                        img_obs = observation['pixels']
                        imageio.imwrite(path + '/img%i.png' % num_positives, img_obs)

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
