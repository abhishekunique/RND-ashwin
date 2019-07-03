import argparse
import numpy as np
import dsuite
import gym
# from dsuite.dclaw.turn import DClawImageTurnFixed
from dsuite.dclaw.turn_free_object import DClawTurnFreeValve3Image
import os
import imageio
import pickle
cur_dir = os.path.dirname(os.path.realpath(__file__))
directory = cur_dir + "/dsuite_free_screw_90"
if not os.path.exists(directory):
    os.makedirs(directory)

# TODO: Make this a generic script, taking in an env and a condition for being a goal image.
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--environment-id", type=str, default="SawyerLift")
#     parser.add_argument("--num-episodes", type=int, default=1)
#     parser.add_argument("--episode-length", type=int, default=1000)
#     parser.add_argument("--render-mode", type=str, default="video")
#     args = parser.parse_args()
#     return args

def main():
    # args = parse_args()

    num_positives = 0
    NUM_TOTAL_EXAMPLES, ROLLOUT_LENGTH, STEPS_PER_SAMPLE = 200, 25, 5
    goal_angle = np.pi
    observations = []
    images = False 
    image_shape = (32, 32, 3)

    """
    env = DClawImageTurnFixed(
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
    env = DClawTurnFreeValve3Image(
        init_angle_range=(goal_angle - 0.025, goal_angle + 0.025),
        target_angle_range=(goal_angle, goal_angle),
        # init_x_pos_range=(-0.025, 0.025),
        # init_y_pos_range=(-0.025, 0.025),
        observation_keys=('image',),
        image_shape=(32, 32, 3),
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
            # env.render()  # render on display
            print(observation, observation.shape)
            obs_dict = env.get_obs_dict()
            
            # For fixed screw
            # object_target_dist = obs_dict['object_to_target_angle_dist']

            # For free screw
            # only the z rotation matters
            object_target_angle_dist = obs_dict["object_to_target_circle_distance"][2] 
            # only take x, y l2-norm
            object_target_position_dist = np.linalg.norm(
                    obs_dict["object_to_target_relative_position"][:2])
            
            print("ANGLE DIST TO TARGET", object_target_angle_dist)
            print("POSITION DISTANCE TO TARGET", object_target_position_dist)

            ANGLE_THRESHOLD, POSITION_THRESHOLD = 0.15, 0.04

            if object_target_angle_dist < ANGLE_THRESHOLD \
                and object_target_position_dist < POSITION_THRESHOLD:
                observations.append(observation)
                if images:
                    image = observation[:np.prod(image_shape)].reshape(image_shape)
                    print_image = (image + 1.) * 255. / 2.
                    imageio.imwrite(directory + '/img%i.jpg' % num_positives, print_image)
                num_positives += 1
            if num_positives % 5 == 0:
                with open(directory + '/positives.pkl', 'wb') as file:
                    pickle.dump(np.array(observations), file)
            t += 1

if __name__ == "__main__":
    main()
