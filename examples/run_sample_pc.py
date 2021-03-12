from exptools.launching.affinity import affinity_from_code
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context
import exptools.logging.logger as logger
import sys
import os
import numpy as np
import torch
from colorsys import hsv_to_rgb

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

    for cam_name in ['fixed', 'vil_camera', 'view_1', 'view_2', 'view_4']:
        frame = env.env.env.sim.render(width=640, height=480,
                            mode='offscreen', camera_name=cam_name, device_id=0)
        frame = frame[::-1, :, :]
        logger.tb_image("rendered {}".format(cam_name), np.transpose(frame, (2,0,1)))

    if args["sample_method"] == "policy":
        args["policy_kwargs"]["hidden_sizes"] = tuple(args["policy_kwargs"]["hidden_sizes"])
        policy = MLP(env.spec, **args["policy_kwargs"])

    gif_frames = list()
    for i in range(args["total_timesteps"]):
        if args["sample_method"] == "policy":
            obs, rew, done, info = env.step(policy.get_action(obs)[0])
        elif args["sample_method"] == "action":
            obs, rew, done, info = env.step(env.action_space.sample())

        logger.record_tabular("step", i, itr= i)
        logger.record_tabular("total_reward", rew, itr= i)
        logger.record_tabular("n_points", len(info["pointcloud"]), itr= i)
        for k, v in info.items():
            if "_p" in k or "_r" in k:
                logger.record_tabular(k, v, itr= i)
        if (i+1) % int(4e3) == 0:
            pc = np.array(info["pointcloud"]) # (N, 3)
            # log pointcloud to tensorboard
            z_max, z_min = np.max(pc[:, 2]), np.min(pc[:, 2])
            colors = np.zeros_like(pc) # (N, 3)
            for pc_idx in range(pc.shape[0]):
                h = pc[pc_idx, 2]
                colors[pc_idx] = hsv_to_rgb(h, 100.0, 100.0)
            logger.log("pc.shape {}".format(pc.shape))
            logger.log("colors.shape {}".format(colors.shape))
            logger._tb_writer.add_mesh("pointcloud",
                vertices= torch.from_numpy(np.expand_dims(pc, axis= 0)),
                colors= torch.from_numpy(np.expand_dims(colors, axis= 0)),
                global_step= i,
            )
            
        if (i+1) <= 100:
            frame = env.env.env.sim.render(width=640, height=480,
                                mode='offscreen', camera_name="view_2", device_id=0)
            frame = np.transpose(frame[::-1, :, :], (2,0,1))
            gif_frames.append(frame)
        if (i+1) == 100:
            logger.record_gif("rendered", gif_frames, itr= i)

        logger.dump_tabular()
    
if __name__ == "__main__":
    main(*sys.argv[1:])
