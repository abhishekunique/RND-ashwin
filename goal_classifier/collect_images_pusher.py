import numpy as np
import imageio
import os
import pickle
import sys
# from sac_envs.envs.dclaw.dclaw3_screw_v2 import DClaw3ImageScrewV2, DClaw3ScrewV2
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from softlearning.environments.gym.mujoco.image_pusher_2d import ImagePusher2dEnv
cur_dir = os.path.dirname(os.path.realpath(__file__))

from sac_envs.utils.quatmath import euler2quat

directory = cur_dir + '/pusher'

if not os.path.exists(directory):
    os.makedirs(directory)

image_shape = (32,32,3)
env = ImagePusher2dEnv(goal=(0, -1.35), image_shape=image_shape)

observations = []
num_positives = 0
TOTAL_EXAMPLES = 500
# What percent of the total examples are the "subgoals"
SUBGOAL_EXAMPLE_RATIO = 0.5

while num_positives <= TOTAL_EXAMPLES:
    t = 0
    env.reset_model()
    print('Resetting')
    env.step(np.zeros(3,))
    prox_idx = env.sim.model.body_name2id("proximal_1")
    dist_1_idx = env.sim.model.body_name2id("distal_1")
    dist_2_idx = env.sim.model.body_name2id("distal_2")
    arm_idx = env.sim.model.body_name2id("distal_4")

    """
    collecting_subgoals = num_positives < SUBGOAL_EXAMPLE_RATIO * TOTAL_EXAMPLES
    if not collecting_subgoals:
        env.reset_model(puck_x_range=(-1.38, -1.32), puck_y_range=(-0.15, 0.15))
    #    print(dir(env.model))
    #    env.model.body_pos[arm_idx] = np.array([0, 0, 0])
        env.model.body_quat[prox_idx] = euler2quat(np.array((0, 0, -1.57)))
        env.model.body_quat[dist_1_idx] = euler2quat(np.array((0, 0, 0)))
        env.model.body_quat[dist_2_idx] = euler2quat(np.array((0, 0, 0)))
    else:
        env.model.body_quat[prox_idx] = euler2quat(np.array((0, 0, 0.45)))
        env.model.body_quat[dist_1_idx] = euler2quat(np.array((0, 0, -0.785)))
        env.model.body_quat[dist_2_idx] = euler2quat(np.array((0, 0, -1.57)))
    """

    while t < 20:
        action = np.random.uniform(env.action_space.low, env.action_space.high, size=(3,))
        for _ in range(6):
            env.step(action)

        env.render()
        obs = env._get_obs()
        super_obs = super(ImagePusher2dEnv, env)._get_obs()

        goal = env._goal

        # print(obs)
        # print(super_obs)
        arm_pos = super_obs[-6:-3][:2] # 3rd dimension is fixed
        puck_pos = super_obs[-3:][:2]
        # print("ARM_POS: ", arm_pos)
        # print("PUCK_POS: ", puck_pos)

        goal_puck_distance = np.linalg.norm(goal - puck_pos)
        arm_puck_distance = np.linalg.norm(arm_pos - puck_pos)

        # print("GOAL-PUCK DIST: ", goal_puck_distance)
        # print("ARM-PUCK DIST: ", arm_puck_distance)

        # if (collecting_subgoals and arm_puck_distance < 0.2 and goal_puck_distance < 0.8) \
        #     or (not collecting_subgoals and goal_puck_distance < 0.1 and arm_puck_distance < 0.3):
        #     observations.append(obs)
        #     image = obs[:np.prod(image_shape)].reshape(image_shape)
        #     print_image = (image + 1)*255/2
        #     imageio.imwrite(directory + '/img%i.jpg' %num_positives, print_image)
        #     num_positives += 1
        # if num_positives % 5 == 0:
        #     with open(directory + '/positives.pkl', 'wb') as file:
        #         pickle.dump(np.array(observations), file)
        t += 1
