''' A demo file telling you how to configure variant and launch experiments
    NOTE: By running this demo, it will create a `data` directory from where you run it.
'''
from exptools.launching.variant import VariantLevel, make_variants, update_config
import numpy as np

seed = 123
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
        init_log_std= 0,
        m_f= 1e4,
        n_f= 1e-6,
        reinitialize= True,
        seed= seed,
    ),
    sample_method = "policy", # `action`:env.action_space.sample(), `policy`
    policy_path = "",
    total_timesteps = int(150),
    seed= seed,
)

def main(args):
    experiment_title = "agent" #"sample_pointclouds"

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
        # [True, False, 4, "up", [-1.57, 0, 0],  [0, -0.14, 0.22], [-1.57, 0, 0],  [0, -0.7, 0.17]], #3-25
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
        # [0, 0, 1, 0.5, 5, ['2d', 64, 8, False], False],
        # [0, 0, 1, 0.5, 5, ['2d', 100, 10, False]],
        # [0, 0, 1, 0.5, 5, ['2d', 144, 12, False]],
        [0, 0, 1, 0.5, 5, ['2d', 256, 16, False], False],
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

    values = [
        # ["action"],
        # ["policy"],
        # ["agent"],
        ["explore"],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("sample_method", ), ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # get all variants and their own log directory
    variants, log_dirs = make_variants(*variant_levels)
    for i, variant in enumerate(variants):
        variants[i] = update_config(default_config, variant)

    if args.where == "local":
        from exptools.launching.affinity import encode_affinity, quick_affinity_code
        from exptools.launching.exp_launcher import run_experiments
        affinity_code = encode_affinity(
            n_cpu_core= 8,
            n_gpu= 4,
            contexts_per_gpu= 2,
        )
        run_experiments(
            script= "examples/run_agent_eval.py",
            affinity_code= affinity_code,
            experiment_title= experiment_title,
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
            script= "examples/run_sample_pc.py",
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