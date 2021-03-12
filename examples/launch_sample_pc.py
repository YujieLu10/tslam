''' A demo file telling you how to configure variant and launch experiments
    NOTE: By running this demo, it will create a `data` directory from where you run it.
'''
from exptools.launching.variant import VariantLevel, make_variants, update_config
import numpy as np

seed = 123
default_config = dict(
    env_name = "adroit-v4",
    env_kwargs = dict(
        obj_bid_idx= 2,
        obj_orientation= [0, 0, 0], # object orientation
        obj_relative_position= [0, 0.5, 0.07], # object position related to hand (z-value will be flipped when arm faced down)
        goal_threshold= int(8e3), # how many points touched to achieve the goal
        new_point_threshold= 0.001, # minimum distance new point to all previous points
        forearm_orientation= "up", # ("up", "down")
        chamfer_r_factor= 1,
        mesh_p_factor= 0,
        mesh_reconstruct_alpha= 0.01,
        palm_r_factor= 1,
        untouch_p_factor= 1,
        newpoints_r_factor= 1,
    ),
    policy_name = "MLP",
    policy_kwargs = dict(
        hidden_sizes= (64,64),
        min_log_std= -3,
        init_log_std= 0,
        seed= seed,
    ),
    sample_method = "action", # `action`, `policy`
    total_timesteps = int(5e4),
    seed= seed,
)

def main(args):
    experiment_title = "sample_pointclouds"

    # set up variants
    variant_levels = list()

    # These are the settings which makes most contact points when
    # uniformly sample from action space
    # values = [
    #     [0, "down", [0, 0, 0],  [0, 0.5, 0.05], ],
    #     [1, "up", [0.77, 0, 0],  [0, 0.5, 0.07], ],
    #     [2, "down", [0, 0, 0],  [0, 0.5, 0.05], ],
    #     [3, "up", [0.77, 0, 0],  [0, 0.5, 0.05], ],
    #     [4, "down", [0.77, 0, 0],  [0, 0.5, 0.07], ],
    #     [5, "down", [0.77, 0, 0],  [0, 0.5, 0.07], ],
    #     [6, "up", [0.77, 0, 0],  [0, 0.5, 0.07], ],
    #     [7, "down", [0, 0, 0],  [0, 0.5, 0.05], ],
    #     [8, "down", [0.77, 0, 0],  [0, 0.5, 0.07], ],
    #     [9, "down", [0.77, 0, 0],  [0, 0.5, 0.07], ],
    # ]
    # dir_names = ["obj{}".format(v[0]) for v in values]
    # keys = [
    #     ("env_kwargs", "obj_bid_idx"),
    #     ("env_kwargs", "forearm_orientation"),
    #     ("env_kwargs", "obj_orientation"),
    #     ("env_kwargs", "obj_relative_position"),
    # ] # each entry in the list is the string path to your config
    # variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # [0,], # running 2490
        # [1,],
        # [2,],
        # [3,],
        # [4,],
        # [5,],
        # [6,],
        # [7,],
        # [8,],
        # [9,],
    ]
    dir_names = ["obj{}".format(*v) for v in values]
    keys = [("env_kwargs", "obj_bid_idx")] # each entry in the list is the string path to your config
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        ["action"],
        # ["policy"],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("sample_method", ), ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        ["up"],
        ["down"],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("env_kwargs", "forearm_orientation"), ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        [[0, 0, 0],  [0, 0.5, 0.05],  ],
        [[0.77, 0, 0],  [0, 0.5, 0.05],  ],
        [[0.77, 0, 0],  [0, 0.5, 0.07],  ],
        [[1.57, 0, 0],  [0, 0.5, 0.05],  ],
    ]
    dir_names = ["orient{}dist{}".format(v[0][0], v[1][2]) for v in values]
    keys = [("env_kwargs", "obj_orientation"), ("env_kwargs", "obj_relative_position"),]
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
            script= "examples/run_sample_pc.py",
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