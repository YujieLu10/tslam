from exptools.launching.affinity import affinity_from_code
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context
import exptools.logging.logger as logger
import sys
import os
from PIL import Image

from mjrl.policies.gaussian_mlp import MLP
from mjrl.utils import gym_env

def main(affinity_code, log_dir, run_ID, **kwargs):
    affinity = affinity_from_code(affinity_code)

    args = load_variant(log_dir)

    name = "sample_pointclouds"
    # This helps you know what GPU is recommand to you for this experiment
    # gpu_idx = affinity["cuda_idx"]
    
    with logger_context(log_dir, run_ID, name, args):
        run_experiment(os.path.join(log_dir, "run_{}".format(run_ID)), args)

def run_experiment(log_dir, args):
    env = gym_env.GymEnv(
                args["env_name"],
                env_kwargs= args["env_kwargs"],
    )
    env.set_seed(args["seed"])
    obs = env.reset()

    frame = env.env.env.sim.render(width=640, height=480,
                        mode='offscreen', camera_name=None, device_id=0)
    frame = frame[::-1, :, :]
    im = Image.fromarray(frame)
    im.save(os.path.join(log_dir, "render.png"))

    if args["sample_method"] == "policy":
        policy = MLP(env.spec, hidden_sizes=(32,32), seed=args["seed"], init_log_std=1.0)

    for _ in range(args["total_timesteps"]):
        if args["sample_method"] == "policy":
            obs, rew, done, _ = env.step(policy.get_action(obs)[0])
        elif args["sample_method"] == "action":
            obs, rew, done, _ = env.step(env.action_space.sample())
        
        
    
if __name__ == "__main__":
    main(*sys.argv[1:])
