from gym.envs.registration import register

# Swing the door open
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

register(
    id='adroit-v3',
    entry_point='mj_envs.envs.hand_manipulation_suite:AdroitEnvV3',
    max_episode_steps=int(1e3),
)
from mj_envs.envs.hand_manipulation_suite.adroit_v3 import AdroitEnvV3

register(
    id='adroit-v4',
    entry_point='mj_envs.envs.hand_manipulation_suite:AdroitEnvV4',
    max_episode_steps=int(1e3),
)
from mj_envs.envs.hand_manipulation_suite.adroit_v4 import AdroitEnvV4
