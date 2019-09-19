import numpy as np
import dsuite
import gym
from softlearning.environments.adapters.gym_adapter import GymAdapter
import os
import imageio
import pickle

cur_dir = os.path.dirname(os.path.realpath(__file__))
directory = cur_dir + "/free_screw_2_goals_bowl_"

def main():
    pos_goals = [(0.01, 0.01), (-0.01, -0.01)]
    angle_goals = [180, 0]
    for goal_index, (angle_goal, pos_goal) in enumerate(zip(angle_goals, pos_goals)):
        num_positives = 0
        NUM_TOTAL_EXAMPLES, ROLLOUT_LENGTH, STEPS_PER_SAMPLE = 200, 25, 4
        ANGLE_THRESHOLD, POSITION_THRESHOLD = 0.15, 0.035
        goal_radians = np.pi / 180. * angle_goal  # convert to radians
        observations = []
        images = True
        image_shape = (32, 32, 3)

        x, y = pos_goal
        env_kwargs = {
            'camera_settings': {
                'azimuth': 0,
                'distance': 0.32,
                'elevation': -45,
                'lookat': (0, 0, 0.03)
            },
            'pixel_wrapper_kwargs': {
                'pixels_only': False,
                'normalize': False,
                'render_kwargs': {
                    'width': image_shape[0],
                    'height': image_shape[1],
                    'camera_id': -1,
                }
            },

            # 'camera_settings': {
            #     'azimuth': 45.,
            #     'distance': 0.32,
            #     'elevation': -55.88,
            #     'lookat': np.array([0.00097442, 0.00063182, 0.03435371])
            # },
            # 'camera_settings': {
            #     'azimuth': 30.,
            #     'distance': 0.35,
            #     'elevation': -38.18,
            #     'lookat': np.array([0.00047, -0.0005, 0.054])
            # },
            'goals': ((x, y, 0, 0, 0, goal_radians),),
            'goal_collection': True,
            'init_angle_range': (goal_radians - 0.05, goal_radians + 0.05),
            'target_angle_range': (goal_radians, goal_radians),
            'observation_keys': ('pixels', 'claw_qpos', 'last_action', 'goal_index'),
            'goal_completion_orientation_threshold': ANGLE_THRESHOLD,
            'goal_completion_position_threshold': POSITION_THRESHOLD,
        }

        env = GymAdapter(
            domain='DClaw',
            task='TurnFreeValve3MultiGoal-v0',
            **env_kwargs
        )

        path = directory + str(angle_goal)
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
                # obs_dict = env.get_obs_dict()
                # print("OBS DICT:", obs_dict)

                if env.get_goal_completion():
                    # Add observation if meets criteria
                    # some hacky shit, find a better way to do this
                    observation['goal_index'] = np.array([goal_index])
                    observations.append(observation)
                    print(observation)
                    if images:
                        img_obs = observation['pixels']
                        imageio.imwrite(path + '/img%i.jpg' % num_positives, img_obs)
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
