from gym.envs.registration import register

# Swing the door open
register(
    id='door-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=100,
)
from mj_envs.envs.hand_manipulation_suite.door_v0 import DoorEnvV0

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
from mj_envs.envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=50,
)
from mj_envs.envs.hand_manipulation_suite.pen_v0 import PenEnvV0

# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0


register(
    id='adroit-v4',
    entry_point='mj_envs.envs.hand_manipulation_suite:AdroitEnvV4',
    max_episode_steps=int(1e3),
)
from mj_envs.envs.hand_manipulation_suite.adroit_v4 import AdroitEnvV4

register(
    id='adroit-v0',
    entry_point='mj_envs.envs.hand_manipulation_suite:AdroitEnvV0',
    max_episode_steps=int(1e3),
)
from mj_envs.envs.hand_manipulation_suite.adroit_v0 import AdroitEnvV0

register(
    id='adroit-v1',
    entry_point='mj_envs.envs.hand_manipulation_suite:AdroitEnvV1',
    max_episode_steps=int(1e3),
)
from mj_envs.envs.hand_manipulation_suite.adroit_v1 import AdroitEnvV1

register(
    id='adroit-v2',
    entry_point='mj_envs.envs.hand_manipulation_suite:AdroitEnvV2',
    max_episode_steps=int(1e3),
)
from mj_envs.envs.hand_manipulation_suite.adroit_v2 import AdroitEnvV2