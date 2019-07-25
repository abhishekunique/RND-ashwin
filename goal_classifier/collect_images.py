import numpy as np
import imageio
import os
import pickle
import sys
from sac_envs.envs.dclaw.dclaw3_screw_v2 import ImageDClaw3ScrewV2, DClaw3ScrewV2
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
cur_dir = os.path.dirname(os.path.realpath(__file__))

# import argparse
# parser = argparse.ArgumentParser("simple_example")
# parser.add_argument('--record_goal', action='store_true', default=False)
# args = parser.parse_args()
# record_goal_images = args.record_goal

# goal_pos = np.pi
image_shape = (32,32,3)
env = ImageDClaw3ScrewV2(is_hardware=False,
                        image_shape=image_shape,
                        object_initial_position_range=(np.pi, np.pi))
# goal_images, neg_goal_images = [], []
# i, j = 0, 0
# while i < 1000 and j < 3000:
#     if not record_goal_images:
#         # start_pos = (np.random.beta(a=2, b=2) * 2 - 1)* np.pi
#         start_pos = np.random.uniform(-1, 1) * np.pi
#         env.object_initial_position_range = (start_pos, start_pos)
#     env.reset_model()
#     t = 0
#     while t < 30:
#         action = np.random.uniform(env.action_space.low, env.action_space.high, size=(9,))
#         for _ in range(5):
#             env.step(action)
#         # print(action)
#         (time, hand_position, hand_velocity, lid_position,
#          lid_velocity) = env.robot.get_obs(env, mimic=False)
#         print(lid_position)
#         obs = env._get_obs()
#         image = obs[:32*32*3].reshape([32,32,3])
#         image = (image + 1)*255/2
#         if np.abs(lid_position - goal_pos) < 0.15 and i < 1000 and record_goal_images:
#             goal_images.append([image, 1, lid_position[0]])
#             imageio.imwrite(cur_dir + '/goal_images_180/img%i.jpg' %i, image)
#             i += 1
#         elif j < 3000 and not record_goal_images:
#             neg_goal_images.append([image, 0, lid_position[0]])
#             imageio.imwrite(cur_dir + '/goal_neg_images_180_unif/img%i.jpg' %j, image)
#             j += 1
#         t += 1
# if record_goal_images:
#     pickle.dump(np.array(goal_images), open(cur_dir + '/goal_images_180/goal_images_180.pkl', 'wb'))
# else:
#     pickle.dump(np.array(neg_goal_images), open(cur_dir + '/goal_neg_images_180_unif/goal_neg_images_180_unif.pkl', 'wb'))

images = []
i = 0
while i <= 10000:
    start_pos = np.random.uniform(0, 2) * np.pi
    env.object_initial_position_range = (start_pos, start_pos)
    env.reset_model()
    i += 1
    while i % 25:
        action = np.random.uniform(env.action_space.low, env.action_space.high, size=(9,))
        for _ in range(5):
            env.step(action)
        # print(action)
        (time, hand_position, hand_velocity, lid_position,
         lid_velocity) = env.robot.get_obs(env, mimic=False)
        print(lid_position)
        obs = env._get_obs()
        image = obs[:np.prod(image_shape)].reshape(image_shape)
        images.append([image, lid_position[0]])
        print_image = (image + 1)*255/2
        imageio.imwrite(cur_dir + '/screw_imgs_2/img%i.jpg' %i, print_image)
        i += 1
    if i % 1000 == 0:
        pickle.dump(np.array(images), open(cur_dir + '/screw_imgs_2/screw_imgs.pkl', 'wb'))
