from exptools.launching.affinity import affinity_from_code
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context
import exptools.logging as logger

import os, sys

from mjrl.utils.gym_env import GymEnv
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.ppo_clip import PPO
from mjrl.utils.train_agent import train_agent
from mjrl.utils.train_generic_agent import train_generic_agent
from mjrl.utils.train_variant_agent import train_variant_agent

def main(affinity_code, log_dir, run_ID, **kwargs):
    # affinity = affinity_from_code(affinity_code)

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
    # use 3d fixed voxel grid
    if "3d" in args["env_kwargs"]["forearm_orientation_name"]:
        from mjrl.policies.gaussian_mlp_3d import MLP
    else:
        from mjrl.policies.gaussian_mlp import MLP
    policy = MLP(env_spec= env.spec, **args["policy_kwargs"])
    if "hidden_sizes" in args["baseline_kwargs"]: args["baseline_kwargs"]["hidden_sizes"] = tuple(args["baseline_kwargs"]["hidden_sizes"])
    baseline = MLPBaseline(env_spec= env.spec, **args["baseline_kwargs"])
    agent = PPO(env, policy, baseline, **args["algo_kwargs"])

    if args["env_kwargs"]["generic"]:
        if args["env_name"] == "adroit_v4": #variant
            train_variant_agent(
                job_name= log_dir, # using this interface to guide the algorithm log files into our designated log_dir
                agent= agent,
                env_kwargs= args["env_kwargs"],
                **args["train_agent_kwargs"],
            )
        else:
            train_generic_agent(
                job_name= log_dir, # using this interface to guide the algorithm log files into our designated log_dir
                agent= agent,
                env_kwargs= args["env_kwargs"],
                **args["train_agent_kwargs"],
            )
    else:
        train_agent(
            job_name= log_dir, # using this interface to guide the algorithm log files into our designated log_dir
            agent= agent,
            env_kwargs= args["env_kwargs"],
            **args["train_agent_kwargs"],
        )
    
if __name__ == "__main__":
    main(*sys.argv[1:])