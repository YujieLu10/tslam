from exptools.launching.variant import VariantLevel, make_variants, update_config
import numpy as np

default_config = dict(
    env_name = "adroit-v0",
    env_kwargs = dict(
        obj_bid_idx= 2,
        obj_orientation= [0, 0, 0], # object orientation
        obj_relative_position= [0, 0.5, 0.07], # object position related to hand (z-value will be flipped when arm faced down)
        goal_threshold= int(8e3), # how many points touched to achieve the goal
        new_point_threshold= 0.001, # minimum distance new point to all previous points
        forearm_orientation= "up", # ("up", "down")
        # scale all reward/penalty to the scale of 1.0
        chamfer_r_factor= 1,
        mesh_p_factor= 1,
        mesh_reconstruct_alpha= 0.01,
        palm_r_factor= 1,
        untouch_p_factor= 1,
        newpoints_r_factor= 1,
        knn_r_factor= 1,
        chamfer_use_gt=False,
    ),
    policy_name = "MLP",
    policy_kwargs = dict(
        hidden_sizes= (64,64),
        min_log_std= -3,
        init_log_std= None,
        m_f= 1e4,
        n_f= 1e-6,
        in_ss= True,
        out_ss= True,
        # seed= seed,
    ),
    baseline_kwargs = dict(
        inp_dim=None,
        inp='obs',
        learn_rate=1e-3,
        reg_coef=0.0,
        batch_size=64,
        epochs=1,
        use_gpu=False,
        hidden_sizes=(128, 128),
    ),
    algo_name = "PPO",
    algo_kwargs = dict(
        clip_coef = 0.2,
        epochs = 10,
        mb_size = 64,
        learn_rate = 3e-4,
        # seed = seed,
        save_logs = True,
    ),
    train_agent_kwargs = dict(
        # seed = seed,
        niter = 1000,
        gamma = 0.995,
        gae_lambda = 0.97,
        num_cpu = 16,
        sample_mode = 'trajectories',
        num_traj = 150,
        num_samples = 50000, # has precedence, used with sample_mode = 'samples'
        save_freq = 5,
        evaluation_rollouts = 5,
        plot_keys = ['stoc_pol_mean'],
        visualize_kwargs = dict(
            horizon=150,
            num_episodes= 1,
            mode='exploration',
            width= 640, height= 480,
            camera_name= "view_1",
            device_id= 0,
        )
    ),
    seed= 123,
)

def main(args):
    # set up variants
    variant_levels = list()

    values = [
        # [0, "down", [0, 0, 0],  [0, 0.5, 0.05], ],
        # [False, 1, "up", [1.57, 0, 0],  [0, 0.6, 0.05], 0, 1, 1, 1e5, 1, True, False, 0.5],
        # [False, 2, "up", [0, 0, 0],  [0, 0.5, 0.05], 0, 1, 1, 5e4, 1, True, True, 0.5],
        # [False, 3, "up", [0.77, 0.97, 0],  [0, 0.5, 0.04], 0, 1, 1, 5e4, 1, True, True, 0.5],
        # [False, 4, "up", [1.57, 0, 0],  [0, 0.6, 0.04], 0, 1, 1, 1e5, 1, True, False, 0.5],
        [True, 5, "up", [1.57, 0, 0],  [0, 0.6, 0.04], 0, 10, 100, 1e4, 1, False, False, 0.5],
        # [True, 5, "up", [1.57, 0, 0],  [0, 0.6, 0.04], 0, 100, 100, 1e4, 1, False, False, 0.5],
        [False, 6, "up", [1.57, 0, 0],  [0, 0.6, 0.02], 0, 100, 10, 1e4, 1, False, False, 0.5],
        # [False, 6, "up", [1.57, 0, 0],  [0, 0.6, 0.02], 0, 1, 1, 1e4, 1, False, False, 0.5],
        # [7, "down", [0, 0, 0],  [0, 0.5, 0.05], ],
        # [False, 8, "up", [0.77, 0, 0],  [0, 0.55, 0.02], 0, 1, 1, 1e5, 1, True, True, 0.5],
        # [False, 8, "up", [0.77, 0, 0],  [0, 0.55, 0.02], 0, 1, 1, 1e4, 1, False, False, 0.5],
        # [False, 8, "up", [0.77, 0, 0],  [0, 0.55, 0.02], 0, 1, 1, 1e4, 1e-6], # no mesh penalty
        # [False, 8, "up", [0.77, 0, 0],  [0, 0.55, 0.02], 0, 1, 1, 1e5, 1e-6], # no mesh penalty
        # [False, 8, "down", [0.77, 0, 0],  [0, 0.55, 0.02], 1, 0, 1], # no chamfer
        # [False, 8, "down", [0.77, 0, 0],  [0, 0.55, 0.02], 1, 1, 0], # no knn
        # [False, 8, "down", [0.77, 0, 0],  [0, 0.55, 0.02], 1, 10, 1], # big chamfer reward
        # [False, 9, "up", [0.77, 0, 0],  [0, 0.55, 0.015], 0, 10, 100, 1e4, 1, False, False, 0.5],
        # [False, 9, "up", [0.77, 0, 0],  [0, 0.55, 0.015], 0, 100, 100, 1e4, 1, False, False, 0.5],
        # [False, 9, "up", [0.77, 0, 0],  [0, 0.55, 0.015], 0, 200, 100, 1e4, 1, False, False, 0.5],
        # [False, 9, "up", [0.77, 0, 0],  [0, 0.55, 0.015], 0, 200, 200, 1e4, 1, False, False, 0.5],
        # [False, 9, "up", [0.77, 0, 0],  [0, 0.55, 0.015], 0, 1, 1, 1e6, 1e-5, 0.5],
        # [False, 9, "down", [0.77, 0, 0],  [0, 0.55, 0.01], 10, 1, 10], # big mesh penalty and big knn reward
    ]
    dir_names = ["obj{}_envv1_mpf{}_crf{}_krf{}_mf{}nf{}_in{}_out{}_logstd{}".format(v[1],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12]) for v in values]
    keys = [
        ("env_kwargs","chamfer_use_gt"),
        ("env_kwargs", "obj_bid_idx"),
        ("env_kwargs", "forearm_orientation"),
        ("env_kwargs", "obj_orientation"),
        ("env_kwargs", "obj_relative_position"),
        ("env_kwargs", "mesh_p_factor"),
        ("env_kwargs", "chamfer_r_factor"),
        ("env_kwargs", "knn_r_factor"),
        ("policy_kwargs", "m_f"),
        ("policy_kwargs", "n_f"),
        ("policy_kwargs", "in_ss"),
        ("policy_kwargs", "out_ss"),
        ("policy_kwargs", "init_log_std"),
    ] # each entry in the list is the string path to your config
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # get all variants and their own log directory
    variants, log_dirs = make_variants(*variant_levels)
    for i, variant in enumerate(variants):
        variants[i] = update_config(default_config, variant)

    experiment_title = "train_adroit"
    if args.where == "local":
        from exptools.launching.affinity import encode_affinity, quick_affinity_code
        from exptools.launching.exp_launcher import run_experiments
        affinity_code = quick_affinity_code(n_parallel= len(variants))
        run_experiments(
            script= "examples/run_train_adroit_yj.py",
            affinity_code= affinity_code,
            experiment_title= experiment_title + ("--debug" if args.debug else ""),
            runs_per_setting= 1, # how many times to run repeated experiments
            variants= variants,
            log_dirs= log_dirs,
            debug_mode= args.debug, # if greater than 0, the launcher will run one variant in this process)
        )
    elif args.where == "slurm":
        from exptools.launching.slurm import build_slurm_resource
        from exptools.launching.exp_launcher import run_on_slurm
        slurm_resource = build_slurm_resource(
            mem= "16G",
            time= "3-12:00:00",
            n_gpus= 1,
            partition= "short",
            cuda_module= "cuda-10.0",
        )
        run_on_slurm(
            script= "examples/run_train_adroit.py",
            slurm_resource= slurm_resource,
            experiment_title= experiment_title + ("--debug" if args.debug else ""),
            # experiment_title= "temp_test" + ("--debug" if args.debug else ""),
            script_name= experiment_title,
            runs_per_setting= 1,
            variants= variants,
            log_dirs= log_dirs,
            debug_mode= args.debug,
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug', help= 'A common setting of whether to entering debug mode for remote attach',
        type= int, default= 0,
    )
    parser.add_argument(
        '--where', help= 'slurm or local',
        type= str, default= "local",
        choices= ["slurm", "local"],
    )

    args = parser.parse_args()
    # setup for debugging if needed
    if args.debug > 0:
        # configuration for remote attach and debug
        import ptvsd
        import sys
        ip_address = ('0.0.0.0', 6789)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=ip_address, redirect_output= True)
        # Pause the program until a remote debugger is attached
        ptvsd.wait_for_attach()
        print("Process attached, start running into experiment...", flush= True)
        ptvsd.break_into_debugger()

    main(args)