import logging
logging.disable(logging.CRITICAL)

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
    os.chdir(job_name) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
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

        if train_curve[i-1] > best_perf:
            if exptools: exptools.logging.logger.log_text("update best_polic")
            best_policy = copy.deepcopy(agent.policy)
            best_perf = train_curve[i-1]

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
                    keys = [k for k in env_infos[0].keys() if "_p" in k[-2:] or "_r" in k[-2:]]
                    for k in keys:
                        rewards[k] = list()
                        for env_info in env_infos:
                            rewards[k].append(env_info[k])
                    for env_info in env_infos:
                        total_points.append(len(env_info["pointcloud"][-1]))
                for k, v in rewards.items():
                    exptools.logging.logger.log_scalar_batch(k, v, i)
                exptools.logging.logger.log_scalar_batch("total_num_points", total_points, i)

        if i % save_freq == 0 and i > 0:
            if agent.save_logs:
                agent.logger.save_log('logs/')
                make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
            policy_file = 'policy_%i.pickle' % i
            baseline_file = 'baseline_%i.pickle' % i
            pickle.dump(agent.policy, open('iterations/' + policy_file, 'wb'))
            pickle.dump(agent.baseline, open('iterations/' + baseline_file, 'wb'))
            pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
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
                np.savez_compressed("pointcloud_"+str(i)+".npz",pcd=pc_frame)
                # pc_frames.append(pc_frame)
                ax = plt.axes()
                # ax.scatter(pc_frame[:, 0], pc_frame[:, 1], pc_frame[:, 2], c=pc_frame[:, 2], cmap='viridis', linewidth=0.5)
                ax.scatter(pc_frame[:, 0], pc_frame[:, 1], cmap='viridis', linewidth=0.5)
                plt.savefig("{}.png".format('2dpointcloud' + str(i)))
                plt.close()
                exptools.logging.logger.record_image("rendered", video[-1], i)
                exptools.logging.logger.record_gif("rendered", video, i)
                exptools.logging.logger.record_image("rendered_explore", video_explore[-1], i)
                exptools.logging.logger.record_gif("rendered_explore", video_explore, i)
                pc = env_infos[-1]["pointcloud"] if len(env_infos[-1]["pointcloud"]) > 0 else np.empty((0, 3)) # (N, 3)
                colors = np.zeros_like(pc)
                for pc_idx in range(pc.shape[0]):
                    h = pc[pc_idx, 2]
                    colors[pc_idx] = hsv_to_rgb(h, 100.0, 100.0)
                exptools.logging.logger.tb_writer.add_mesh("pointcloud",
                    vertices= torch.from_numpy(np.expand_dims(pc, axis= 0)),
                    # colors= torch.from_numpy(np.expand_dims(colors, axis= 0)),
                    global_step= i,
                )
                mesh = pv.PolyData(pc).delaunay_3d(alpha= env_kwargs["mesh_reconstruct_alpha"]).extract_geometry()
                exptools.logging.logger.tb_writer.add_mesh("reconstruction",
                    vertices= torch.from_numpy(np.expand_dims(mesh.points, 0)),
                    faces= torch.from_numpy(np.expand_dims(mesh.faces.reshape(-1, 4)[:, 1:], 0)),
                    global_step= i,
                )
                

        # print results to console
        if i == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        # print("[ %s ] %4i %5.2f %5.2f %5.2f " % (timer.asctime(timer.localtime(timer.time())),
        #                                          i, train_curve[i], mean_pol_perf, best_perf))
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
