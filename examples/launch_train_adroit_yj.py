from exptools.launching.variant import VariantLevel, make_variants, update_config
import numpy as np

default_config = dict(
    env_name = "adroit-v1",
    env_kwargs = dict(
        obj_bid_idx= 2,
        obj_orientation= [0, 0, 0], # object orientation
        obj_relative_position= [0, 0.5, 0.07], # object position related to hand (z-value will be flipped when arm faced down)
        goal_threshold= int(8e3), # how many points touched to achieve the goal
        new_point_threshold= 0.001, # minimum distance new point to all previous points
        forearm_orientation_name= "up", # ("up", "down")
        # scale all reward/penalty to the scale of 1.0
        chamfer_r_factor= 0,
        mesh_p_factor= 0,
        mesh_reconstruct_alpha= 0.01,
        palm_r_factor= 0,
        untouch_p_factor= 0,
        newpoints_r_factor= 0,
        knn_r_factor= 0,
        new_voxel_r_factor= 0,
        use_voxel= False,
        ground_truth_type= "nope",
        forearm_orientation= [0, 0, 0], # forearm orientation
        forearm_relative_position= [0, 0.5, 0.07], # forearm position related to hand (z-value will be flipped when arm faced down)
        reset_mode= "normal",
        knn_k= 1,
        voxel_conf= ['2d', 16, 4, False],
        sensor_obs= False,
    ),
    policy_name = "MLP",
    policy_kwargs = dict(
        hidden_sizes= (64,64),
        min_log_std= -3,
        init_log_std= None,
        m_f= 1,
        n_f= 1,
        reinitialize= True,
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
        learn_rate = 3e-3,
        # seed = seed,
        save_logs = True,
    ),
    train_agent_kwargs = dict(
        # seed = seed,
        niter = 600,
        gamma = 0.995,
        gae_lambda = 0.97,
        num_cpu = 8,
        sample_mode = 'trajectories',
        horizon= 150, 
        num_traj = 150,
        num_samples = 50000, # has precedence, used with sample_mode = 'samples' 50000
        save_freq = 3,
        evaluation_rollouts = 3,
        plot_keys = ['stoc_pol_mean'],
        visualize_kwargs = dict(
            horizon=150,
            num_episodes= 1,
            mode='evaluation',
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
        ["normal"],
        # ["intermediate"],
        # ["random"],
    ]
    dir_names = ["reset{}".format(*tuple(str(vi) for vi in v)) for v in values]
    keys = [
        ("env_kwargs", "reset_mode"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        ["sample"],
        # ["mesh"],
        # ["nope"],
    ]
    dir_names = ["gt{}".format(*tuple(str(vi) for vi in v)) for v in values]
    keys = [
        ("env_kwargs", "ground_truth_type"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # [True, True, 4, "down", [-1.57, 0, 0],  [0, -0.14, 0.22], [-1.57, 0, 3],  [0, -0.7, 0.28]], #3-21
        # [False, False, 4, "down", [-1.57, 0, 0],  [0, -0.14, 0.22], [-1.57, 0, 3],  [0, -0.7, 0.28]],
        [False, False, 4, "up", [-1.57, 0, 0],  [0, -0.14, 0.22], [-1.57, 0, 0],  [0, -0.7, 0.17]], #3-25
        [True, False, 4, "up", [-1.57, 0, 0],  [0, -0.14, 0.22], [-1.57, 0, 0],  [0, -0.7, 0.17]], #3-25
        # [True, False, 4, "up", [-1.57, 0, 0],  [0, -0.14, 0.22], [-1.57, 0, 0],  [0, -0.7, 0.17], 0, 10, 10],
        # [False, True, 4, "down", [-1.57, 0, 0],  [0, -0.14, 0.22], [-1.57, 0, 3],  [0, -0.7, 0.28]],
    ]
    dir_names = ["voxel{}_rw{}_obj{}_orien{}_{}_{}_{}_{}".format(*tuple(str(vi) for vi in v)) for v in values]
    keys = [
        ("env_kwargs", "use_voxel"),
        ("policy_kwargs", "reinitialize"),
        ("env_kwargs", "obj_bid_idx"),
        ("env_kwargs", "forearm_orientation_name"),
        ("env_kwargs", "obj_orientation"),
        ("env_kwargs", "obj_relative_position"),
        ("env_kwargs", "forearm_orientation"),
        ("env_kwargs", "forearm_relative_position"),
    ] # each entry in the list is the string path to your config
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # voxel_conf= ['2d', 16, 4, False]
    values = [
        [0, 0, 1, 0.5, 5, ['3d', 0, 0.005, False], False],
        [0, 0, 1, 0.5, 5, ['3d', 0, 0.01, False], False],
    ]
    dir_names = ["cf{}_knn{}_vr{}_lstd{}_knnk{}_vconf{}_sensor{}".format(*tuple(str(vi) for vi in v)) for v in values]
    keys = [
        ("env_kwargs", "chamfer_r_factor"),
        ("env_kwargs", "knn_r_factor"),
        ("env_kwargs", "new_voxel_r_factor"),
        ("policy_kwargs", "init_log_std"),
        ("env_kwargs", "knn_k"),
        ("env_kwargs", "voxel_conf"),
        ("env_kwargs", "sensor_obs"),
    ]
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
