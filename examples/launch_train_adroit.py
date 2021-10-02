from exptools.launching.variant import VariantLevel, make_variants, update_config
import numpy as np
import math

default_config = dict(
    env_name = "adroit-v2", # adroit-v2: our best policy # adroit-v3: variant using knn reward or chamfer reward # adroit-v4: new points reward and only touch reward; adroit-v2 coverage_voxel_r(old new voxel r):coverage curiosity_voxel_r:curiosity # adroit-v5 un-supervised chamfer
    env_kwargs = dict(
        obj_orientation= [0, 0, 0], # object orientation
        obj_relative_position= [0, 0.5, 0.07], # object position related to hand (z-value will be flipped when arm faced down)
        goal_threshold= int(0.3), # how many points touched to achieve the goal 8e3 => occupancy threshold set to 0.3
        new_point_threshold= 0.001, # minimum distance new point to all previous points
        forearm_orientation_name= "up", # ("up", "down")
        # scale all reward/penalty to the scale of 1.0
        # chamfer_r_factor= 0,
        # disagree_r_factor= 0,
        mesh_p_factor= 0,
        mesh_reconstruct_alpha= 0.01,
        palm_r_factor= 0,
        untouch_p_factor= 0,
        newpoints_r_factor= 0,
        # npoint_r_factor= 0,
        # ntouch_r_factor= 0,
        # random_r_factor= 0,
        ground_truth_type= "nope",
        # pretrain_mode= "pretrain",
        knn_r_factor= 0,
        # new_voxel_r_factor= 0, # new => coverage_voxel_r
        coverage_voxel_r_factor= 0, # new and touched objects
        curiosity_voxel_r_factor= 0, # new voxel
        use_voxel= False,
        forearm_orientation= [0, 0, 0], # forearm orientation
        forearm_relative_position= [0, 0.5, 0.07], # forearm position related to hand (z-value will be flipped when arm faced down)
        reset_mode= "normal",
        knn_k= 1,
        voxel_conf= ['2d', 4],
        obj_scale= 0.01,
        obj_name= "airplane",
        generic= False,
        base_rotation= False,
        obs_type= [False, False], # 1: fix voxel + 3dconv; 2: sensor obs
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
        batch_size=16,
        epochs=1,
        use_gpu=True,
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
        niter = 10000,
        gamma = 0.995,
        gae_lambda = 0.97,
        num_cpu = 8,
        sample_mode = 'trajectories',
        horizon= 200, 
        num_traj = 60,
        num_samples = 5000, # has precedence, used with sample_mode = 'samples' 50000
        save_freq = 3,
        evaluation_rollouts = 5,
        plot_keys = ['stoc_pol_mean'],
        visualize_kwargs = dict(
            horizon=200,
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
        ["intermediate"],
        ["random"],
    ]
    reset = int(args.reset)
    values = values[reset:(reset+1)]
    dir_names = ["reset{}".format(*tuple(str(vi) for vi in v)) for v in values]
    keys = [
        ("env_kwargs", "reset_mode"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # set pretrain task
    # values = [
    #     ["pretrain"],
    #     ["randominit"],
    # ]
    # dir_names = ["pretrainmode{}".format(*tuple(str(vi) for vi in v)) for v in values]
    # keys = [
    #     ("env_kwargs", "pretrain_mode"),
    # ]
    # variant_levels.append(VariantLevel(keys, values, dir_names))

    # set ground truth type, unified touching policy, hand base rotation
    values = [
        ["sample", True, True],
        # ["mesh"],
        # ["nope"],
    ]
    dir_names = ["gt{}_gene{}_rot{}".format(*tuple(str(vi) for vi in v)) for v in values]
    keys = [
        ("env_kwargs", "ground_truth_type"),
        ("env_kwargs", "generic"),
        ("env_kwargs", "base_rotation"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))
    # tabletop
    values = [
        [True, False, "glass", "up", [1.57, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.015],
        [True, False, "donut", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.01],
        [True, False, "heart", "up", [-1.57, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.0006],
        [True, False, "airplane", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "alarmclock", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "apple", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "banana", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "binoculars", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "body", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.1],
        [True, False, "bowl", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "camera", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "coffeemug", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "cubelarge", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "cubemedium", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "cubemiddle", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "cubesmall", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "cup", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "cylinderlarge", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "cylindermedium", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "cylindersmall", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "doorknob", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "duck", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "elephant", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "eyeglasses", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "flashlight", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "flute", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "fryingpan", "up", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.8],
        [True, False, "gamecontroller", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "hammer", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "hand", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "headphones", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "knife", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "lightbulb", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "mouse", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "mug", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "phone", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "piggybank", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "pyramidlarge", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "pyramidmedium", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "pyramidsmall", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "rubberduck", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "scissors", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "spherelarge", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "spheremedium", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "spheresmall", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "stamp", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "stanfordbunny", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "stapler", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "table", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.5],
        [True, False, "teapot", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "toothbrush", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "toothpaste", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "toruslarge", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "torusmedium", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "torussmall", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "train", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "watch", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "waterbottle", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "wineglass", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1], #58
        [True, False, "wristwatch", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
    ]
    # generic policy with several hand poses
    idx = int(args.obj)
    if idx < 0:
        values = [
                    # [True, False, "generic", "fixdown3d", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 1], # fix voxel grid with 3dconv
                    # [True, False, "generic", "fixup3d", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
                    # [True, False, "generic", "500fixdown", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.3], 1], # long horizon -7
                    # [True, False, "generic", "500fixup", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.13], 1],
                    [True, False, "generic", "unified3d", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926], [0, -0.56, 0.1], 1], # long horizon -7
                    # [True, False, "generic", "unified", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926], [0, -0.56, 0.3], 1], # long horizon -7
                ]
        # values = values[-idx-1:-idx]
    else:
        values = values[idx*2:min((idx+1)*2, len(values) - 1)]
    dir_names = ["voxel{}_rw{}_obj{}_orien{}".format(*tuple(str(vi) for vi in v[0:4])) for v in values]
    keys = [
        ("env_kwargs", "use_voxel"),
        ("policy_kwargs", "reinitialize"),
        ("env_kwargs", "obj_name"),
        ("env_kwargs", "forearm_orientation_name"),
        ("env_kwargs", "obj_orientation"),
        ("env_kwargs", "obj_relative_position"),
        ("env_kwargs", "forearm_orientation"),
        ("env_kwargs", "forearm_relative_position"),
        ("env_kwargs", "obj_scale"),
    ] # each entry in the list is the string path to your config
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # reward setting and voxel observatoin mode
    values = [
        # [1, 0, 0.5, ['3d', 6], [True, False]], # curiosity
        # [0, 1, 0.5, ['3d', 6], [True, False]], # coverage : old best policy
        [1, 3, 0.5, ['3d', 8], [True, False]], # cur & cove : ours
        # [3, 1, 0.5, ['3d', 32], [True, False]], # cur & cove : ours
        # [1, 1, 0.5, ['3d', 6], [True, False]], # cur & cove : ours
        # [3, 1, 0.5, ['3d', 8], [True, False]], # cur & cove : ours
        # [1, 0.5, ['3d', 6], [True, False]], # disagreement variant
        # [0, 0, 1, 0.5, 5, ['3d', 6], [True, False]], # best policy | random
        # [0, 1, 0, 0.5, 5, ['3d', 6], [True, False]], # knn variant | ntouch
        # [1, 0, 0, 0.5, 5, ['3d', 6], [True, False]], # chamfer variant | npoint
        # [0, 0, 1, 0.5, 5, ['3d', 8], [True, False]],
        # [0, 0, 1, 0.5, 5, ['3d', 0.02], [True, False]],
    ]
    # dir_names = ["cf{}_knn{}_vr{}_lstd{}_knnk{}_vconf{}_obst{}".format(*tuple(str(vi) for vi in v)) for v in values]
    # dir_names = ["npoint{}_ntouch{}_random{}_lstd{}_knnk{}_vconf{}_obst{}".format(*tuple(str(vi) for vi in v)) for v in values]
    dir_names = ["curf{}covf{}_lstd{}_vconf{}_obst{}".format(*tuple(str(vi) for vi in v)) for v in values]
    # dir_names = ["disagreef{}_lstd{}_vconf{}_obst{}".format(*tuple(str(vi) for vi in v)) for v in values]
    keys = [
        ("env_kwargs", "curiosity_voxel_r_factor"),
        ("env_kwargs", "coverage_voxel_r_factor"),
        # ("env_kwargs", "chamfer_r_factor"),
        # ("env_kwargs", "knn_r_factor"),
        # ("env_kwargs", "new_voxel_r_factor"),
        # ("env_kwargs", "npoint_r_factor"),
        # ("env_kwargs", "ntouch_r_factor"),
        # ("env_kwargs", "random_r_factor"),
        # ("env_kwargs", "disagree_r_factor"),
        ("policy_kwargs", "init_log_std"),
        # ("env_kwargs", "knn_k"),
        ("env_kwargs", "voxel_conf"),
        ("env_kwargs", "obs_type"),
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
    parser.add_argument(
        '--obj', help= 'obj',
        type= int, default= -1,
    )
    parser.add_argument(
        '--reset', help= 'reset',
        type= int, default= 0,
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
