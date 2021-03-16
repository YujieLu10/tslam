from exptools.launching.variant import VariantLevel, make_variants, update_config
import numpy as np

default_config = dict(
    env_name = "adroit-v4",
    env_kwargs = dict(
        obj_bid_idx= 2,
        obj_orientation= [0, 0, 0], # object orientation
        obj_relative_position= [0, 0.5, 0.07], # object position related to hand (z-value will be flipped when arm faced down)
        goal_threshold= int(8e3), # how many points touched to achieve the goal
        new_point_threshold= 0.0001, # minimum distance new point to all previous points
        forearm_orientation= "up", # ("up", "down")
        # scale all reward/penalty to the scale of 1.0
        chamfer_r_factor= 1e3,
        mesh_p_factor= 1,
        mesh_reconstruct_alpha= 0.01,
        palm_r_factor= 1e2,
        untouch_p_factor= 1,
        newpoints_r_factor= 1e2,
    ),
    policy_name = "MLP",
    policy_kwargs = dict(
        hidden_sizes= (64,64),
        min_log_std= -3,
        init_log_std= 0,
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
        niter = 200,
        gamma = 0.995,
        gae_lambda = 0.97,
        num_cpu = 8,
        sample_mode = 'trajectories',
        num_traj = 150,
        num_samples = 50000, # has precedence, used with sample_mode = 'samples'
        save_freq = 10,
        evaluation_rollouts = 5,
        plot_keys = ['stoc_pol_mean'],
        visualize_kwargs = dict(
            horizon=100,
            num_episodes= 1,
            mode='exploration',
            width= 640, height= 480,
            camera_name= "view_1",
            device_id= 0,
        ),
        sample_paths_kwargs = dict(
            horizon=1e6,
            max_process_time=300,
            max_timeouts=4,
            suppress_print=False,
        ),
    ),
    seed= 123,
)

def main(args):
    # set up variants
    variant_levels = list()

    values = [
        # [0, "down", [0, 0, 0],  [0, 0.5, 0.05], ],
        [1, "up", [0.77, 0, 0],  [0, 0.5, 0.07], ], # hard 0
        [2, "down", [0, 0, 0],  [0, 0.5, 0.05], ], # hard 1
        [3, "up", [0.77, 0, 0],  [0, 0.5, 0.05], ], # medium 0
        [4, "down", [0.77, 0, 0],  [0, 0.5, 0.07], ], # medium 1
        # [5, "down", [0.77, 0, 0],  [0, 0.5, 0.07], ],
        # [6, "up", [0.77, 0, 0],  [0, 0.5, 0.07], ],
        # [7, "down", [0, 0, 0],  [0, 0.5, 0.05], ],
        [8, "down", [0.77, 0, 0],  [0, 0.55, 0.045], ], # simple 0
        [9, "down", [0.77, 0, 0],  [0, 0.55, 0.05], ],  # simple 1
    ]
    dir_names = ["obj{}".format(v[0]) for v in values]
    keys = [
        ("env_kwargs", "obj_bid_idx"),
        ("env_kwargs", "forearm_orientation"),
        ("env_kwargs", "obj_orientation"),
        ("env_kwargs", "obj_relative_position"),
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
        affinity_code = encode_affinity(
            n_cpu_core= 12,
            n_gpu= 4,
            contexts_per_gpu= 3,
        )
        run_experiments(
            script= "examples/run_train_adroit.py",
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