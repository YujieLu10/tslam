# Tactile Slam
`tslam` uses environments/tasks in [mj_envs](https://github.com/vikashplus/mj_envs.git) and RL algorithms in [mjrl](https://github.com/vikashplus/mjrl.git).

## Quick Start
1. Repo preparation
```
$ git clone --recursive https://github.com/YujieLu10/tslam.git
$ cd tslam  
$ git submodule update --remote
$ pip install -e .
```

2. Env
mjrl env registration example(mjrl/mjrl/envs/__init__.py):
```
register(
    id='adroit_2-v4',
    entry_point='mj_envs.hand_manipulation_suite:AdroitEnv2V4',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.adroit_v4_2 import AdroitEnv2V4
```
In adroit_v4_2, we use `DAPG_constrainedhand.xml` as mujocoenv.

3. Policy Visualization
policy visualization example:
```
$ cd mjrl 
$ python3 ../mj_envs/utils/visualize_env.py --env_name adroit-v2 --episodes ${1} --policy ../mjrl/job_dir/adroit_2-v4_nn_ppo_exp5/iterations/best_policy.pickle
```

4. Policy Training
Args: env_name exp_version ppo_clip_coef_value
policy training example:
```
$ cd mjrl
$ python3 ../mjrl/examples/linear_nn_comparison_adroitvx_expx.py adroit_2-v4 exp5 0.2
```

5. Visualizeion
`visualization tools` are in mjrl/vis_tool
`point cloud` and `mesh` visualization results are in mjrl/visualization_result