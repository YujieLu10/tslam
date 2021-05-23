import logging
logging.disable(logging.CRITICAL)
import math
from tabulate import tabulate
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
import numpy as np
import torch
import pickle
import imageio
import time as timer
import os
import copy
import matplotlib.pyplot as plt

try:
    import exptools
    from colorsys import hsv_to_rgb
    import pyvista as pv
except ImportError:
    exptools = None

def _load_latest_policy_and_logs(agent, *, policy_dir, logs_dir):
    """Loads the latest policy.
    Returns the next step number to begin with.
    """
    assert os.path.isdir(policy_dir), str(policy_dir)
    assert os.path.isdir(logs_dir), str(logs_dir)

    log_csv_path = os.path.join(logs_dir, 'log.csv')
    if not os.path.exists(log_csv_path):
        return 0   # fresh start

    print("Reading: {}".format(log_csv_path))
    agent.logger.read_log(log_csv_path)
    last_step = agent.logger.max_len - 1
    if last_step <= 0:
        return 0   # fresh start


    # find latest policy/baseline
    i = last_step
    while i >= 0:
        policy_path = os.path.join(policy_dir, 'policy_{}.pickle'.format(i))
        baseline_path = os.path.join(policy_dir, 'baseline_{}.pickle'.format(i))

        if not os.path.isfile(policy_path):
            i = i -1
            continue
        else:
            print("Loaded last saved iteration: {}".format(i))

        with open(policy_path, 'rb') as fp:
            agent.policy = pickle.load(fp)
        with open(baseline_path, 'rb') as fp:
            agent.baseline = pickle.load(fp)

        # additional
        # global_status_path = os.path.join(policy_dir, 'global_status.pickle')
        # with open(global_status_path, 'rb') as fp:
        #     agent.load_global_status( pickle.load(fp) )

        agent.logger.shrink_to(i + 1)
        assert agent.logger.max_len == i + 1
        return agent.logger.max_len

    # cannot find any saved policy
    raise RuntimeError("Log file exists, but cannot find any saved policy.")

def save_voxel_visualization(obj_name, reset_mode_conf, reward_conf, obj_orientation, obj_relative_position, obj_scale, pc_frame, iternum, is_best_policy):
    uniform_gt_data = np.load("/home/jianrenw/prox/tslam/assets/uniform_gt/uniform_{}_o3d.npz".format(obj_name))['pcd']
    data_scale = uniform_gt_data * obj_scale
    data_rotate = data_scale.copy()
    x = data_rotate[:, 0].copy()
    y = data_rotate[:, 1].copy()
    z = data_rotate[:, 2].copy()
    x_theta = obj_orientation[0]
    data_rotate[:, 0] = x
    data_rotate[:, 1] = y*math.cos(x_theta) - z*math.sin(x_theta)
    data_rotate[:, 2] = y*math.sin(x_theta) + z*math.cos(x_theta)
    data_trans = data_rotate.copy()
    data_trans[:, 0] += obj_relative_position[0]
    data_trans[:, 1] += obj_relative_position[1]
    data_trans[:, 2] += obj_relative_position[2]

    uniform_gt_data = data_trans.copy()
    data = pc_frame
    resolution = 0.01
    sep_x = math.ceil(0.25 / resolution)
    sep_y = math.ceil(0.225 / resolution)
    sep_z = math.ceil(0.1 / resolution)
    x, y, z = np.indices((sep_x, sep_y, sep_z))

    cube1 = (x<0) & (y <1) & (z<1)
    gtcube = (x<0) & (y <1) & (z<1)
    voxels = cube1
    gt_voxels = gtcube
    # draw cuboids in the top left and bottom right corners, and a link between them
    map_list = []
    for idx,val in enumerate(data):
        idx_x = math.floor((val[0] + 0.125) / resolution)
        idx_y = math.floor((val[1] + 0.25) / resolution)
        idx_z = math.floor((val[2] - 0.16) / resolution)
        if idx_z > 6:
            continue
        name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
        if name not in map_list:
            map_list.append(name)
        cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
        # combine the objects into a single boolean array
        voxels += cube

    # draw gt
    gt_map_list = []
    for idx,val in enumerate(uniform_gt_data):
        idx_x = math.floor((val[0] + 0.125) / resolution)
        idx_y = math.floor((val[1] + 0.25) / resolution)
        idx_z = math.floor((val[2] - 0.16) / resolution)
        if idx_z > 6:
            continue
        name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
        if name not in gt_map_list:
            gt_map_list.append(name)
        cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
        # combine the objects into a single boolean array
        gt_voxels += cube
    # gt_obj4:668
    occupancy = len(map_list) / len(gt_map_list)
    # print(len(map_list) / sep_x / sep_y / sep_z )

    obj_name = "obj{}".format(obj_name)
    # set the colors of each object
    vis_voxel = gt_voxels | voxels
    colors = np.empty(vis_voxel.shape, dtype=object)
    colors[gt_voxels] = 'white'
    colors[voxels] = 'cyan'
    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_zlim(1,20)
    ax.voxels(vis_voxel, facecolors=colors, edgecolor='g', alpha=.4, linewidth=.05)
    # plt.savefig('uniform_gtbox_{}.png'.format(step))
    is_best_reconstruct = False
    files = os.listdir('/home/jianrenw/prox/tslam/data/result/best_eval/{}/{}/{}/'.format(obj_name, reset_mode_conf, reward_conf))
    for file in files:
        if "overlap" in file:
            file_str = str(file)
            previous_occup = file_str[(file_str.index("-")+1):file_str.index(".png")]
            if occupancy > previous_occup:
                is_best_reconstruct = True
        else:
            is_best_reconstruct = True

    if is_best_policy or is_best_reconstruct:
        plt.savefig('/home/jianrenw/prox/tslam/data/result/best_eval/{}/{}/{}/bp{}_br{}_overlap-{}.png'.format(obj_name, reset_mode_conf, reward_conf, is_best_policy, is_best_reconstruct, occupancy))    
    plt.savefig('voxel/iter-{}-{}-overlap-{}.png'.format(iternum, obj_name, occupancy))
    plt.close()

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_zlim(1,20)
    ax.voxels(gt_voxels, facecolors=colors, edgecolor='g', alpha=.4, linewidth=.05)
    if is_best_policy or is_best_reconstruct:
        plt.savefig('/home/jianrenw/prox/tslam/data/result/best_eval/{}/{}/{}/gt_.png'.format(obj_name))    
    plt.savefig('voxel/iter-{}-{}-gt.png'.format(iternum, obj_name))
    plt.close()

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_zlim(1,20)
    ax.voxels(voxels, facecolors=colors, edgecolor='g', alpha=.4, linewidth=.05)
    if is_best_policy or is_best_reconstruct:
        plt.savefig('/home/jianrenw/prox/tslam/data/result/best_eval/{}/{}/{}/bp{}_br{}_exp.png'.format(obj_name, reset_mode_conf, reward_conf, is_best_policy, is_best_reconstruct))    
    plt.savefig('voxel/iter-{}-{}-exp.png'.format(iternum, obj_name))
    plt.close()
    return is_best_reconstruct

def train_agent(job_name, agent,
                seed = 0,
                niter = 101,
                gamma = 0.995,
                gae_lambda = None,
                num_cpu = 16,
                sample_mode = 'trajectories',
                horizon= int(150), 
                num_traj = 50,
                num_samples = 50000, # has precedence, used with sample_mode = 'samples'
                save_freq = 10,
                evaluation_rollouts = None,
                plot_keys = ['stoc_pol_mean'],
                env_kwargs= dict(),
                visualize_kwargs= dict(),
                sample_paths_kwargs= dict(),
                ):
    print("num_cpu{}".format(num_cpu))
    np.random.seed(seed)
    if os.path.isdir(job_name) == False:
        os.mkdir(job_name)
    previous_dir = os.getcwd()
    obj_name = env_kwargs["obj_name"]
    reset_mode_conf = env_kwargs["reset_mode"]
    reward_conf = "cf{}knn{}voxel{}".format(env_kwargs["chamfer_r_factor"], env_kwargs["knn_r_factor"], env_kwargs["new_voxel_r_factor"])
    os.chdir(job_name) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('2dpointcloud') == False: os.mkdir('2dpointcloud')
    if os.path.isdir('pointcloudnpz') == False: os.mkdir('pointcloudnpz')
    if os.path.isdir('/home/jianrenw/prox/tslam/data/result/best_policy/{}/{}/{}'.format(obj_name, reset_mode_conf, reward_conf)) == False: os.makedirs('/home/jianrenw/prox/tslam/data/result/best_policy/{}/{}/{}'.format(obj_name, reset_mode_conf, reward_conf))
    if os.path.isdir('/home/jianrenw/prox/tslam/data/result/best_eval/{}/{}/{}'.format(obj_name, reset_mode_conf, reward_conf)) == False: os.makedirs('/home/jianrenw/prox/tslam/data/result/best_eval/{}/{}/{}'.format(obj_name, reset_mode_conf, reward_conf))
    if os.path.isdir('voxel') == False: os.mkdir('voxel')
    if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')
    best_policy = copy.deepcopy(agent.policy)
    best_perf = -1e8
    train_curve = best_perf*np.ones(niter)
    mean_pol_perf = 0.0
    e = GymEnv(agent.env.env_id, env_kwargs)

    # Load from any existing checkpoint, policy, statistics, etc.
    # Why no checkpointing.. :(
    i_start = _load_latest_policy_and_logs(agent,
                                           policy_dir='iterations',
                                           logs_dir='logs')
    if i_start:
        print("Resuming from an existing job folder ...")

    for i in range(i_start, niter):
        print("......................................................................................")
        print("ITERATION : %i " % i)
        is_best_policy = False
        if train_curve[i-1] > best_perf:
            if exptools: exptools.logging.logger.log_text("update best_policy")
            best_policy = copy.deepcopy(agent.policy)
            best_perf = train_curve[i-1]
            is_best_policy = True

        N = num_traj if sample_mode == 'trajectories' else num_samples
        stats = agent.train_step(
            N=N, 
            sample_mode=sample_mode,
            horizon= horizon, 
            gamma=gamma,
            gae_lambda=gae_lambda,
            num_cpu=num_cpu,
            env_kwargs= env_kwargs,
            sample_paths_kwargs= sample_paths_kwargs,
        )
        train_curve[i] = stats[0]

        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths(
                num_traj=evaluation_rollouts,
                env=e.env_id,
                policy=agent.policy,
                eval_mode=True,
                base_seed=seed,
                num_cpu=num_cpu,
                env_kwargs= env_kwargs,
                **sample_paths_kwargs)
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)
                if exptools: exptools.logging.logger.log_scalar('eval_score', mean_pol_perf, i)
            if exptools:
                env_infos = [path["env_infos"] for path in eval_paths] # a list of dict
                rewards = dict()
                total_points = list()
                if env_infos:
                    # get decomposed reward statistics
                    keys = [k for k in env_infos[0].keys() if "_p" in k[-2:] or "_r" in k[-2:] or "occupancy" in k]
                    for k in keys:
                        rewards[k] = list()
                        for env_info in env_infos:
                            rewards[k].append(env_info[k])
                    for env_info in env_infos:
                        total_points.append(len(env_info["pointcloud"]))
                for k, v in rewards.items():
                    exptools.logging.logger.log_scalar_batch(k, v, i)
                exptools.logging.logger.log_scalar_batch("total_num_points", total_points, i)
            print(">>> finish evaluation rollouts")

        if (i % save_freq == 0 and i > 0):
            if agent.save_logs:
                agent.logger.save_log('logs/')
                make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
            obj_orientation = env_kwargs["obj_orientation"]
            obj_relative_position = env_kwargs["obj_relative_position"]
            obj_scale = env_kwargs["obj_scale"]  
            policy_file = 'policy_%i.pickle' % i
            baseline_file = 'baseline_%i.pickle' % i
            # pickle.dump(agent.policy, open('iterations/' + policy_file, 'wb'))
            # pickle.dump(agent.baseline, open('iterations/' + baseline_file, 'wb'))
            # pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))

            # pickle.dump(agent.global_status, open('iterations/global_status.pickle', 'wb'))

            # save videos and pointcloud and reconstruted mesh          
            if exptools:
                video, env_infos = e.visualize_policy_offscreen(
                    policy= agent.policy,
                    **visualize_kwargs,
                ) # (T, C, H, W)
                video_explore, env_infos_explore = e.visualize_policy_explore(
                    policy= agent.policy,
                    **visualize_kwargs,
                ) # (T, C, H, W)
                pc_frame = np.array(env_infos[-1]["pointcloud"] if len(env_infos[-1]["pointcloud"]) > 0 else np.empty((0, 3)))

                # 3d voxel visualization
                is_best_reconstruct = save_voxel_visualization(obj_name, reset_mode_conf, reward_conf, obj_orientation, obj_relative_position, obj_scale, pc_frame, i, is_best_policy)
                if is_best_policy or is_best_reconstruct:
                    pickle.dump(best_policy, open('/home/jianrenw/prox/tslam/data/result/best_policy/{}/{}/{}/bp{}_br{}_best_policy.pickle'.format(obj_name, reset_mode_conf, reward_conf, is_best_policy, is_best_reconstruct), 'wb'))
                if is_best_policy or is_best_reconstruct:
                    np.savez_compressed("pointcloudnpz/alpha_pointcloud_"+str(i)+".npz",pcd=pc_frame)
                    np.savez_compressed("/home/jianrenw/prox/tslam/data/result/best_eval/{}/{}/{}/bp{}_br{}_alpha_pointcloud.npz".format(obj_name, reset_mode_conf, reward_conf, is_best_policy, is_best_reconstruct), pcd=pc_frame)
                else:
                    np.savez_compressed("pointcloudnpz/pointcloud_"+str(i)+".npz",pcd=pc_frame)
                
                # pc_frames.append(pc_frame)
                ax = plt.axes()
                ax.scatter(pc_frame[:, 0], pc_frame[:, 1], cmap='viridis', linewidth=0.5)
                if is_best_policy or is_best_reconstruct:
                    plt.savefig("2dpointcloud/alpha_{}.png".format('2dpointcloud' + str(i)))
                    plt.savefig("/home/jianrenw/prox/tslam/data/result/best_eval/{}/{}/{}/bp{}_br{}_alpha_2dpointcloud.png".format(obj_name, reset_mode_conf, reward_conf, is_best_policy, is_best_reconstruct))
                else:
                    plt.savefig("2dpointcloud/{}.png".format('2dpointcloud' + str(i)))
                plt.close()
                # =======================================================
                exptools.logging.logger.record_image("rendered", video[-1], i)
                exptools.logging.logger.record_gif("rendered", video, i)
                exptools.logging.logger.record_image("rendered_explore", video_explore[-1], i)
                exptools.logging.logger.record_gif("rendered_explore", video_explore, i)

        # print results to console
        if i == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        result_file = open('results.txt', 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (i, train_curve[i], mean_pol_perf, best_perf))
        result_file.close()
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       agent.logger.get_current_log().items()))
            print(tabulate(print_data))
        if exptools:
            exptools.logging.logger.log_scalar("Iter", i, i)
            exptools.logging.logger.log_scalar("SamplingPol", train_curve[i], i)
            exptools.logging.logger.log_scalar("EvaluationPol", mean_pol_perf, i)
            exptools.logging.logger.log_scalar("BestSampled", best_perf, i)
            exptools.logging.logger.dump_data()

    # final save
    pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
    if agent.save_logs:
        agent.logger.save_log('logs/')
        make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
    os.chdir(previous_dir)
