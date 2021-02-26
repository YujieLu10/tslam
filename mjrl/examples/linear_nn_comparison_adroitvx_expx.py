from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.ppo_clip import PPO
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer
import sys
SEED = 500

env_name = sys.argv[1]
exp_num = sys.argv[2]
coef = float(sys.argv[3])
print(env_name)
print(exp_num)
# NN policy
# ==================================
e = GymEnv(env_name)
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED, min_log_std=3, init_log_std=10)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=5, learn_rate=1e-3)
# agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)
agent = PPO(e, policy, baseline, clip_coef=coef, normalized_step_size=0.1, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='{}_nn_ppo_{}'.format(env_name, exp_num),
            agent=agent,
            seed=SEED,
            niter=100,
            gamma=0.995,  
            gae_lambda=0.97,
            num_cpu=8,
            sample_mode='trajectories',
            num_traj=150,
            save_freq=5,
            evaluation_rollouts=5)
print("time taken for NN policy training = %f" % (timer.time()-ts))


# Linear policy
# ==================================
# e = GymEnv(env_name)
# policy = LinearPolicy(e.spec, seed=SEED)
# baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=5, learn_rate=1e-3)
# # agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)
# agent = PPO(e, policy, baseline, clip_coef=coef, normalized_step_size=0.1, seed=SEED, save_logs=True)

# ts = timer.time()
# train_agent(job_name='{}_linear_ppo_{}'.format(env_name, exp_num),
#             agent=agent,
#             seed=SEED,
#             niter=100,
#             gamma=0.995,  
#             gae_lambda=0.97,
#             num_cpu=16,
#             sample_mode='trajectories',
#             num_traj=600,
#             save_freq=20,
#             evaluation_rollouts=20)
# print("time taken for linear policy training = %f" % (timer.time()-ts))
