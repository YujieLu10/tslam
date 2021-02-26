from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------

register(
    id='mjrl_point_mass-v0',
    entry_point='mjrl.envs:PointMassEnv',
    max_episode_steps=25,
)

register(
    id='mjrl_swimmer-v0',
    entry_point='mjrl.envs:SwimmerEnv',
    max_episode_steps=500,
)

register(
    id='mjrl_reacher_7dof-v0',
    entry_point='mjrl.envs:Reacher7DOFEnv',
    max_episode_steps=50,
)

register(
    id='mjrl_peg_insertion-v0',
    entry_point='mjrl.envs:PegEnv',
    max_episode_steps=50,
)

register(
    id='adroit_2-v4',
    entry_point='mj_envs.hand_manipulation_suite:AdroitEnv2V4',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.adroit_v4_2 import AdroitEnv2V4

register(
    id='adroit_2-v5',
    entry_point='mj_envs.hand_manipulation_suite:AdroitEnv2V5',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.adroit_v5_2 import AdroitEnv2V5

# register(
#     id='adroit-v1',
#     entry_point='mj_envs.hand_manipulation_suite:AdroitEnvV0',
#     max_episode_steps=200,
# )
# from mj_envs.hand_manipulation_suite.adroit_v1 import AdroitEnvV0

# register(
#     id='adroit-v2',
#     entry_point='mj_envs.hand_manipulation_suite:AdroitEnvV2',
#     max_episode_steps=200,
# )
# from mj_envs.hand_manipulation_suite.adroit_v2 import AdroitEnvV2

# register(
#     id='adroit-v3',
#     entry_point='mj_envs.hand_manipulation_suite:AdroitEnvV3',
#     max_episode_steps=200,
# )
# from mj_envs.hand_manipulation_suite.adroit_v3 import AdroitEnvV3

# register(
#     id='adroit-v4',
#     entry_point='mj_envs.hand_manipulation_suite:AdroitEnvV4',
#     max_episode_steps=200,
# )
# from mj_envs.hand_manipulation_suite.adroit_v4 import AdroitEnvV4

# register(
#     id='adroit_3-v4',
#     entry_point='mj_envs.hand_manipulation_suite:AdroitEnv3V4',
#     max_episode_steps=200,
# )
# from mj_envs.hand_manipulation_suite.adroit_v4_3 import AdroitEnv3V4

# register(
#     id='adroit_4-v4',
#     entry_point='mj_envs.hand_manipulation_suite:AdroitEnv4V4',
#     max_episode_steps=200,
# )
# from mj_envs.hand_manipulation_suite.adroit_v4_4 import AdroitEnv4V4
# from mjrl.mjrl.envs.mujoco_env import MujocoEnv
# # ^^^^^ so that user gets the correct error
# # message if mujoco is not installed correctly
# from mjrl.mjrl.envs.point_mass import PointMassEnv
# from mjrl.mjrl.envs.swimmer import SwimmerEnv
# from mjrl.mjrl.envs.reacher_sawyer import Reacher7DOFEnv
# from mjrl.mjrl.envs.peg_insertion_sawyer import PegEnv

from mjrl.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from mjrl.envs.point_mass import PointMassEnv
from mjrl.envs.swimmer import SwimmerEnv
from mjrl.envs.reacher_sawyer import Reacher7DOFEnv
from mjrl.envs.peg_insertion_sawyer import PegEnv
