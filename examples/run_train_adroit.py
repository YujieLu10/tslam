from exptools.launching.affinity import affinity_from_code
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context
import exptools.logging.logger as logger

import os, sys

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.ppo_clip import PPO
from mjrl.utils.train_agent import train_agent


def main(affinity_code, log_dir, run_ID, **kwargs):
    affinity = affinity_from_code(affinity_code)

    args = load_variant(log_dir)
    args["policy_kwargs"]["seed"] = args["seed"]
    args["algo_kwargs"]["seed"] = args["seed"]
    args["train_agent_kwargs"]["seed"] = args["seed"]

    name = "train_adroit_ppo"
    # This helps you know what GPU is recommand to you for this experiment
    # gpu_idx = affinity["cuda_idx"]
    
    with logger_context(log_dir, run_ID, name, args):
        run_experiment(os.path.join(log_dir, "run_{}".format(run_ID)), args)

def run_experiment(log_dir, args):
    env = GymEnv(args["env_name"], args["env_kwargs"])
    if "hidden_sizes" in args["policy_kwargs"]: args["policy_kwargs"]["hidden_sizes"] = tuple(args["policy_kwargs"]["hidden_sizes"])
    policy = MLP(env_spec= env.spec, **args["policy_kwargs"])
    if "hidden_sizes" in args["baseline_kwargs"]: args["baseline_kwargs"]["hidden_sizes"] = tuple(args["baseline_kwargs"]["hidden_sizes"])
    baseline = MLPBaseline(env_spec= env.spec, **args["baseline_kwargs"])
    agent = PPO(env, policy, baseline, **args["algo_kwargs"])

    train_agent(
        job_name= log_dir, # using this interface to guide the algorithm log files into our designated log_dir
        agent= agent,
        **args["train_agent_kwargs"],
    )
    
if __name__ == "__main__":
    main(*sys.argv[1:])