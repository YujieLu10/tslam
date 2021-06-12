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
from tool.visualization import save_voxel_visualization
import matplotlib.pyplot as plt
from mjrl.policies.gaussian_mlp import MLP
from mjrl.utils import gym_env
import shutil

save_agent_eval_root = "/home/jianrenw/prox/tslam/data/result/agent_eval"

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
    name_map = ['duck', 'watch', 'doorknob', 'headphones', 'bowl', 'cubesmall', 'spheremedium', 'train', 'piggybank', 'cubemedium', 'cubelarge', 'elephant', 'flute', 'wristwatch', 'pyramidmedium', 'gamecontroller', 'toothbrush', 'pyramidsmall', 'body', 'cylinderlarge', 'cylindermedium', 'cylindersmall', 'fryingpan', 'stanfordbunny', 'scissors', 'pyramidlarge', 'stapler', 'flashlight', 'mug', 'hand', 'stamp', 'rubberduck', 'binoculars', 'apple', 'mouse', 'eyeglasses', 'airplane', 'coffeemug', 'cup', 'toothpaste', 'torusmedium', 'cubemiddle', 'phone', 'torussmall', 'spheresmall', 'knife', 'banana', 'teapot', 'hammer', 'alarmclock', 'waterbottle', 'camera', 'table', 'wineglass', 'lightbulb', 'spherelarge', 'toruslarge', 'glass', 'heart', 'donut']

    obs = env.reset(name_map.index(args["env_kwargs"]["obj_name"]))


    conf_eval_dir = "geneTrue_rotTrue_{}/normal_cf{}knn{}voxel{}".format(args["env_kwargs"]["forearm_orientation_name"], int(args["env_kwargs"]["chamfer_r_factor"]), int(args["env_kwargs"]["knn_r_factor"]), int(args["env_kwargs"]["new_voxel_r_factor"])) if "chamfer_r_factor" in args["env_kwargs"] else "geneTrue_rotTrue_{}/normal_npoint{}_ntouch{}_random{}".format(args["env_kwargs"]["forearm_orientation_name"], int(args["env_kwargs"]["npoint_r_factor"]), int(args["env_kwargs"]["ntouch_r_factor"]), int(args["env_kwargs"]["random_r_factor"]))

    save_agent_eval_dir = os.path.join(save_agent_eval_root, str(args["env_kwargs"]["obj_name"]), conf_eval_dir)
    if os.path.isdir(save_agent_eval_dir) == True:
        shutil.rmtree(save_agent_eval_dir)
        print(">>> clear file {}!".format(save_agent_eval_dir))
    if not os.path.isdir(save_agent_eval_dir): os.makedirs(save_agent_eval_dir)

    for cam_name in ['fixed', 'vil_camera', 'view_1', 'view_2', 'view_4']:
        frame = env.env.env.sim.render(width=640, height=480,
                            mode='offscreen', camera_name=cam_name, device_id=0)
        frame = frame[::-1, :, :]
        logger.log_image("rendered {}".format(cam_name), np.transpose(frame, (2,0,1)))

    if args["sample_method"] == "policy":
        args["policy_kwargs"]["hidden_sizes"] = tuple(args["policy_kwargs"]["hidden_sizes"])
        policy = MLP(env.spec, **args["policy_kwargs"])

    if args["sample_method"] == "agent" or args["sample_method"] == "explore":
        policy = pickle.load(open(os.path.join("/home/jianrenw/prox/tslam/data/result/best_policy", "lightbulb", conf_eval_dir.replace("10k", "500").replace("fixup", "fixdown"), "bpFalse_brTrue_best_policy.pickle"), 'rb'))
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
        
        if (i+1) % int(100) == 0:
            pc_frame = np.array(info["pointcloud"])
            # ====== voxel visualization
            vis_data_tuple = [args["env_kwargs"]["obj_name"], args["env_kwargs"]["obj_orientation"], args["env_kwargs"]["obj_relative_position"], args["env_kwargs"]["obj_scale"], pc_frame, i]
            overlap = save_voxel_visualization(vis_data_tuple, save_agent_eval_dir)
            # ====== 3d pointcloud
            np.savez_compressed(os.path.join(save_agent_eval_dir, "iternum_" + str(i)) + "_pointcloud_overlap-{}.npz".format(overlap),pcd=pc_frame)
            # ====== 2d visualization
            ax = plt.axes()
            ax.scatter(pc_frame[:, 0], pc_frame[:, 1], cmap='viridis', linewidth=0.5)
            plt.savefig(os.path.join(save_agent_eval_dir, "iternum_" + str(i)) + "_2dpointcloud_overlap-{}.png".format(overlap))
            plt.close()

        # record gif
        if i < 500:
            frame = env.env.env.sim.render(width=640, height=480,
                                mode='offscreen', camera_name="view_1", device_id=0)
            frame = np.transpose(frame[::-1, :, :], (2,0,1))
            gif_frames.append(frame)
        if (i+1) == 500:
            logger.log_gif("rendered", gif_frames, i)

        logger.dump_tabular()
    
if __name__ == "__main__":
    main(*sys.argv[1:])
