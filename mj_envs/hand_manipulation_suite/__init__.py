from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv

# Swing the door open
register(
    id='door-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
)
from mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0
from mj_envs.hand_manipulation_suite.adroit_v4_2 import AdroitEnv2V4
from mj_envs.hand_manipulation_suite.adroit_v5_2 import AdroitEnv2V5

# Adroit
# register(
#     id='adroit-v1',
#     entry_point='mj_envs.hand_manipulation_suite:AdroitEnvV0',
#     max_episode_steps=500,
# )
# from mj_envs.hand_manipulation_suite.adroit_v1 import AdroitEnvV0
# from mj_envs.hand_manipulation_suite.adroit_v2 import AdroitEnvV2
# from mj_envs.hand_manipulation_suite.adroit_v3 import AdroitEnvV3
# from mj_envs.hand_manipulation_suite.adroit_v4 import AdroitEnvV4
# from mj_envs.hand_manipulation_suite.adroit_v4_3 import AdroitEnv3V4
# from mj_envs.hand_manipulation_suite.adroit_v4_4 import AdroitEnv4V4

# register(
#     id='adroit-v0',
#     entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
#     max_episode_steps=150,
# )
# from mj_envs.hand_manipulation_suite.adroit_v0 import RelocateEnvV0

# Relcoate an object to the target
# register(
#     id='relocate-v0',
#     entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
#     max_episode_steps=200,
# )
# from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0
