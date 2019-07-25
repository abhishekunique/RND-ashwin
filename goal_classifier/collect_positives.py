import numpy as np
import imageio
import os
import pickle
import sys
from sac_envs.envs.dclaw.dclaw3_screw_v2 import DClaw3ImageScrewV2, DClaw3ScrewV2
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
cur_dir = os.path.dirname(os.path.realpath(__file__))

images = True
if images:
    directory = cur_dir + '/goal_0_obs'
else:
    directory = cur_dir + '/goal_180_state'
if not os.path.exists(directory):
    os.makedirs(directory)

goal_pos = 0
image_shape = (32,32,3)
if images:
    env = DClaw3ImageScrewV2(is_hardware=False,
                            image_shape=image_shape,
                            object_initial_position_range=(goal_pos, goal_pos),
                            target_initial_position_range=(goal_pos, goal_pos))
else:
    env = DClaw3ScrewV2(is_hardware=False,
                            object_initial_position_range=(goal_pos, goal_pos),
                            target_initial_position_range=(goal_pos, goal_pos))
observations = []
num_positives = 0
while num_positives <= 1000:
    t = 0
    env.reset_model()
    print('Resetting')
    while t < 25:
        action = np.random.uniform(env.action_space.low, env.action_space.high, size=(9,))
        for _ in range(5):
            env.step(action)
        (time, hand_position, hand_velocity, lid_position,
         lid_velocity) = env.robot.get_obs(env, mimic=False)
        print(abs(lid_position - goal_pos)[0])
        obs = env._get_obs()
        if abs(lid_position - goal_pos)[0] < 0.1:
            observations.append(obs)
            if images:
                image = obs[:np.prod(image_shape)].reshape(image_shape)
                print_image = (image + 1)*255/2
                imageio.imwrite(directory + '/img%i.jpg' %num_positives, print_image)
            num_positives += 1
        if num_positives % 5 == 0:
            with open(directory + '/positives.pkl', 'wb') as file:
                pickle.dump(np.array(observations), file)
        t += 1
