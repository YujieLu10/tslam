from exptools.launching.affinity import affinity_from_code
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context
# import exptools.logging.logger as logger
from exptools.logging import logger
import sys
import os
import numpy as np
import torch
from colorsys import hsv_to_rgb
import pickle

import matplotlib.pyplot as plt
from mjrl.policies.gaussian_mlp import MLP
from mjrl.utils import gym_env

def main(affinity_code, log_dir, run_ID, **kwargs):
    affinity = affinity_from_code(affinity_code)

    args = load_variant(log_dir)

    name = "agent" #"sample_pointclouds"
    # This helps you know what GPU is recommand to you for this experiment
    # gpu_idx = affinity["cuda_idx"]
    
    with logger_context(log_dir, run_ID, name, args):
        run_experiment(os.path.join(log_dir, "run_{}".format(run_ID)), args)

def get_box_from_voxel_array(voxel_array):
    idx_list = [i for i,v in enumerate(voxel_array) if v > 0]
    return idx_list

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
        logger.log_image("rendered {}".format(cam_name), np.transpose(frame, (2,0,1)))

    if args["sample_method"] == "policy":
        args["policy_kwargs"]["hidden_sizes"] = tuple(args["policy_kwargs"]["hidden_sizes"])
        policy = MLP(env.spec, **args["policy_kwargs"])

    if args["sample_method"] == "agent" or args["sample_method"] == "explore":
        policy = pickle.load(open(os.path.join("/home/jianrenw/prox/tslam/data/local/best_policy", str(args["env_kwargs"]["obj_name"]), "policy_18.pickle"), 'rb'))
        # policy = pickle.load(open(os.path.join("/home/jianrenw/ziwenz/tslam/data/local/train_adroit/20210314", "obj" + str(args["env_kwargs"]["obj_bid_idx"]), "run_0/iterations", "best_policy.pickle"), 'rb'))

    gif_frames = list()
    pc_frames = list()
    for i in range(args["total_timesteps"]):
        if args["sample_method"] == "explore":
            obs, rew, done, info = env.step(policy.get_action(obs)[0])
        elif args["sample_method"] == "action":
            obs, rew, done, info = env.step(env.action_space.sample())
        elif args["sample_method"] == "agent":
            obs, rew, done, info = env.step(policy.get_action(obs)[1]['evaluation'])
        elif args["sample_method"] == "policy":
            obs, rew, done, info = env.step(policy.get_action(obs)[0])

        logger.log_scalar("step", i, i)
        logger.log_scalar("total_reward", rew, i)
        logger.log_scalar("n_points", len(info["pointcloud"]), i)
        for k, v in info.items():
            if "_p" in k or "_r" in k or "occupancy" in k:
                logger.log_scalar(k, v, i)
        if (i+1) % int(1e4) == 0:
            pc = np.array(info["pointcloud"]) # (N, 3)
            # log pointcloud to tensorboard
            z_max, z_min = np.max(pc[:, 2]), np.min(pc[:, 2])
            colors = np.zeros_like(pc) # (N, 3)
            for pc_idx in range(pc.shape[0]):
                h = pc[pc_idx, 2]
                colors[pc_idx] = hsv_to_rgb(h/10, 100.0, 100.0)
            logger.log_text("pc.shape {}".format(pc.shape))
            logger.log_text("colors.shape {}".format(colors.shape))
            logger.tb_writer.add_mesh("pointcloud",
                vertices= torch.from_numpy(np.expand_dims(pc, axis= 0)),
                colors= torch.from_numpy(np.expand_dims(colors, axis= 0)),
                global_step= i,
            )
            # np.savez_compressed(os.path.join(log_dir, str(i+1)+"pointcloud.npz"), pcd=pc)
        if (i+1) % int(50) == 0:
            if not os.path.isdir(os.path.join(log_dir, "2dnewpointcloud")):
                os.mkdir(os.path.join(log_dir, "2dnewpointcloud"))
            # str(args["env_kwargs"]["forearm_orientation"])
            pc_frame = np.array(info["pointcloud"])
            np.savez_compressed(os.path.join(log_dir, "2dnewpointcloud", "obj" + str(args["env_kwargs"]["obj_name"]) + "_orien_" + "_step_" + str(i)) + ".npz",pcd=pc_frame)
            # pc_frames.append(pc_frame)
            # ax = plt.axes(projection='3d')
            # ax.scatter(pc_frame[:, 0], pc_frame[:, 1], pc_frame[:, 2], c=pc_frame[:, 2], cmap='viridis', linewidth=0.5)
            # visualize 2d
            ax = plt.axes()
            ax.scatter(pc_frame[:, 0], pc_frame[:, 1], cmap='viridis', linewidth=0.5)
            plt.savefig("{}.png".format(os.path.join(log_dir, "2dnewpointcloud", "obj" + str(args["env_kwargs"]["obj_name"]) + "_step_" + str(i))))
            plt.close()

        if (i+1) <= 300:
            frame = env.env.env.sim.render(width=640, height=480,
                                mode='offscreen', camera_name="view_1", device_id=0)
            frame = np.transpose(frame[::-1, :, :], (2,0,1))
            gif_frames.append(frame)
        if (i+1) == 300:
            logger.log_gif("rendered", gif_frames, i)

        logger.dump_tabular()
    
if __name__ == "__main__":
    main(*sys.argv[1:])
