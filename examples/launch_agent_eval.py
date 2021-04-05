''' A demo file telling you how to configure variant and launch experiments
    NOTE: By running this demo, it will create a `data` directory from where you run it.
'''
from exptools.launching.variant import VariantLevel, make_variants, update_config
import numpy as np

seed = 123
default_config = dict(
    env_name = "adroit-v0",
    env_kwargs = dict(
        obj_bid_idx= 2,
        obj_orientation= [0, 0, 0], # object orientation
        obj_relative_position= [0, 0.5, 0.07], # object position related to hand (z-value will be flipped when arm faced down)
        goal_threshold= int(8e3), # how many points touched to achieve the goal
        new_point_threshold= 0.001, # minimum distance new point to all previous points
        forearm_orientation= "up", # ("up", "down")
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
        init_log_std= 0,
        m_f= 1e4,
        n_f= 1e-6,
        in_ss= True,
        out_ss= True,
        seed= seed,
    ),
    sample_method = "policy", # `action`:env.action_space.sample(), `policy`
    policy_path = "",
    total_timesteps = int(1e3),
    seed= seed,
)

def main(args):
    experiment_title = "agent" #"sample_pointclouds"

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
        # [0, "down", [0, 0, 0],  [0, 0.5, 0.05], ],
        # [False, 1, "down", [1.57, 0, 0],  [0, 0.6, 0.05], 10, 1, 1],
        # [True, 1, "down", [1.57, 0, 0],  [0, 0.6, 0.05], 10, 1, 1],
        # [True, 1, "up", [1.57, 0, 0],  [0, 0.6, 0.05], 10, 1, 1],
        # [True, 2, "up", [0, 0, 0],  [0, 0.5, 0.05], 0, 1, 1],
        # [False, 2, "up", [0, 0, 0],  [0, 0.5, 0.05], 0, 1, 1],
        # [False, 3, "down", [0.77, 0.97, 0],  [0, 0.5, 0.04], 10, 1, 1],
        # [True, 3, "up", [0.77, 0.97, 0],  [0, 0.5, 0.04], 10, 1, 1],
        # [False, 4, "down", [1.57, 0, 0],  [0, 0.6, 0.04], 0, 1, 1],
        # [True, 4, "down", [1.57, 0, 0],  [0, 0.6, 0.04], 0, 100, 100],
        [False, 4, "down", [1.57, 0, 0],  [0, 0.6, 0.04], 0, 0, 100],
        # [False, 5, "down", [1.57, 0, 0],  [0, 0.6, 0.04], 10, 1, 1],
        # [True, 5, "up", [1.57, 0, 0],  [0, 0.6, 0.04], 0, 100, 100],
        # [True, 5, "down", [1.57, 0, 0],  [0, 0.6, 0.04], 0, 100, 10],
        # [True, 6, "down", [1.57, 0, 0],  [0, 0.6, 0.02], 0, 1, 1],
        # [False, 6, "up", [1.57, 0, 0],  [0, 0.6, 0.02], 0, 1, 1],
        # [7, "down", [0, 0, 0],  [0, 0.5, 0.05], ],
        # [False, 8, "down", [0.77, 0, 0],  [0, 0.55, 0.02], 0, 1, 1],
        # [8, "up", [0.77, 0, 0],  [0, 0.55, 0.02], 0, 1],
        # [False, 8, "down", [0.77, 0, 0],  [0, 0.55, 0.02], 0, 1, 1], # no mesh penalty
        # [False, 8, "down", [0.77, 0, 0],  [0, 0.55, 0.02], 1, 0, 1], # no chamfer
        # [False, 8, "down", [0.77, 0, 0],  [0, 0.55, 0.02], 1, 1, 0], # no knn
        # [False, 8, "down", [0.77, 0, 0],  [0, 0.55, 0.02], 1, 10, 1], # big chamfer reward
        # [False, 9, "down", [0.77, 0, 0],  [0, 0.55, 0.015], 0, 1, 1],
        # [9, "up", [0.77, 0, 0],  [0, 0.55, 0.015], 0, 1],
        # [False, 9, "down", [0.77, 0, 0],  [0, 0.55, 0.01], 10, 1, 10], # big mesh penalty and big knn reward
    ]
    dir_names = ["obj{}_{}_{}".format(v[0],v[1],v[2]) for v in values]
    keys = [
        ("env_kwargs","chamfer_use_gt"),
        ("env_kwargs", "obj_bid_idx"),
        ("env_kwargs", "forearm_orientation"),
        ("env_kwargs", "obj_orientation"),
        ("env_kwargs", "obj_relative_position"),
        ("env_kwargs", "mesh_p_factor"),
        ("env_kwargs", "chamfer_r_factor"),
        ("env_kwargs", "knn_r_factor"),
    ] # each entry in the list is the string path to your config
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # values = [
    #     # [0,], # running 2490
    #     # [1,],
    #     # [2,],
    #     # [3,],
    #     # [4,],
    #     # [5,],
    #     # [6,],
    #     # [7,],
    #     [8,],
    #     [9,],
    # ]
    # dir_names = ["obj{}".format(*v) for v in values]
    # keys = [("env_kwargs", "obj_bid_idx")] # each entry in the list is the string path to your config
    # variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # ["action"],
        # ["policy"],
        ["agent"],
        # ["explore"],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("sample_method", ), ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # values = [
    #     # ["up"],
    #     ["down"],
    # ]
    # dir_names = ["{}".format(*v) for v in values]
    # keys = [("env_kwargs", "forearm_orientation"), ]
    # variant_levels.append(VariantLevel(keys, values, dir_names))

    # values = [
    #     # [[0, 0, 0],  [0, 0.5, 0.07],  ],
    #     # [[0.77, 0, 0],  [0, 0.55, 0.02],  ],
    #     [[0.77, 0, 0],  [0, 0.55, 0.02],  ],
    #     [[0.77, 0, 0],  [0, 0.55, 0.015],  ],
    #     # [[1.57, 0, 0],  [0, 0.6, 0.04],  ],
    #     # [[1.57, 0, 0],  [0, 0.6, 0.04],  ],
    # ]
    # dir_names = ["orient{}dist{}hdist{}".format(v[0][0], v[1][2], v[1][1]) for v in values]
    # keys = [("env_kwargs", "obj_orientation"), ("env_kwargs", "obj_relative_position"),]
    # variant_levels.append(VariantLevel(keys, values, dir_names))

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