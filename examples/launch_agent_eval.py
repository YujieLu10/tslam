''' A demo file telling you how to configure variant and launch experiments
    NOTE: By running this demo, it will create a `data` directory from where you run it.
'''
from exptools.launching.variant import VariantLevel, make_variants, update_config
import numpy as np

seed = 123
default_config = dict(
    env_name = "adroit-v2", # adroit-v2: our best policy # adroit-v3: variant using knn reward or chamfer reward # adroit-v4: new points reward and only touch reward
    env_kwargs = dict(
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
        # npoint_r_factor= 0,
        # ntouch_r_factor= 0,
        # random_r_factor= 0,
        ground_truth_type= "nope",
        knn_r_factor= 0,
        new_voxel_r_factor= 0,
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
    sample_method = "policy", # `action`:env.action_space.sample(), `policy`
    policy_path = "",
    total_timesteps = int(10000),
    seed= seed,
)

def main(args):
    experiment_title = "policy" #"sample_pointclouds"

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

    values = [
        # [True, False, "glass", "up", [1.57, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.015],
        # [True, False, "donut", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.01],
        # [True, False, "heart", "up", [-1.57, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.0006],
        # [True, False, "airplane", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        [True, False, "airplane", "10kfixdown", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 1], # fix voxel grid
        # [True, False, "airplane", "10kfixup", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "alarmclock", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "apple", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "banana", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "binoculars", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "body", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.1],
        # [True, False, "body", "10kfixdown", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 0.1], # fix voxel grid
        # [True, False, "body", "10kfixup", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.1],
        # [True, False, "bowl", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "camera", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "coffeemug", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "cubelarge", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "cubemedium", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "cubemiddle", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "cubesmall", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "cup", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "cup", "10kfixdown", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 1], # fix voxel grid
        # [True, False, "cup", "10kfixup", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "cylinderlarge", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "cylindermedium", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "cylindersmall", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "doorknob", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "duck", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "elephant", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "eyeglasses", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "flashlight", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "flute", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "fryingpan", "up", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.8],
        # [True, False, "fryingpan", "10kfixdown", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 0.8], # fix voxel grid
        # [True, False, "fryingpan", "10kfixup", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 0.8],
        # [True, False, "gamecontroller", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "hammer", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "hand", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "headphones", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "knife", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "lightbulb", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        #[True, False, "lightbulb", "10kfixdown", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 1], # fix voxel grid
        # [True, False, "lightbulb", "10kfixup", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "mouse", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "mug", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "phone", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "piggybank", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "pyramidlarge", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "pyramidmedium", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "pyramidsmall", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "rubberduck", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "scissors", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "spherelarge", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        #[True, False, "spherelarge", "10kfixdown", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 1], # fix voxel grid
        # [True, False, "spherelarge", "10kfixup", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "spheremedium", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "spheresmall", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "stamp", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "stanfordbunny", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "stapler", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "table", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "teapot", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "toothbrush", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "toothpaste", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "toruslarge", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "torusmedium", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "torussmall", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "train", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "watch", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "waterbottle", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
        # [True, False, "wineglass", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1], #58
        # [True, False, "wristwatch", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
    ]
    # generic policy with several hand poses
    # idx = int(args.obj)
    # if idx < 0:
    #     values = [
    #                 # [True, False, "generic", "down", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 1],
    #                 # [True, False, "generic", "up", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
    #                 [True, False, "generic", "fixdown", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 1], # fix voxel grid
    #                 [True, False, "generic", "fixup", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
    #                 # [True, False, "generic", "fixdown3d", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 1], # fix voxel grid with 3dconv
    #                 # [True, False, "generic", "fixup3d", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
    #                 # [True, False, "generic", "500fixdown", [0, 0, 0],  [0, -0.12, 0.23], [-1.57, 0, 3.14151926],  [0, -0.7, 0.27], 1], # long horizon -7
    #                 # [True, False, "generic", "500fixup", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 0, 0],  [0, -0.7, 0.17], 1],
    #             ]
    #     # values = values[-idx-1:-idx]
    # else:
    #     values = values[idx*2:min((idx+1)*2, len(values) - 1)]
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
        [0, 0, 1, 0.5, 5, ['3d', 6], [True, False]], # best policy | random
        # [0, 1, 0, 0.5, 5, ['3d', 6], [True, False]], # knn variant | ntouch
        # [1, 0, 0, 0.5, 5, ['3d', 6], [True, False]], # chamfer variant | npoint
        # [0, 0, 1, 0.5, 5, ['3d', 8], [True, False]],
        # [0, 0, 1, 0.5, 5, ['3d', 0.02], [True, False]],
    ]
    dir_names = ["cf{}_knn{}_vr{}_lstd{}_knnk{}_vconf{}_obst{}".format(*tuple(str(vi) for vi in v)) for v in values]
    # dir_names = ["npoint{}_ntouch{}_random{}_lstd{}_knnk{}_vconf{}_obst{}".format(*tuple(str(vi) for vi in v)) for v in values]
    keys = [
        ("env_kwargs", "chamfer_r_factor"),
        ("env_kwargs", "knn_r_factor"),
        ("env_kwargs", "new_voxel_r_factor"),
        # ("env_kwargs", "npoint_r_factor"),
        # ("env_kwargs", "ntouch_r_factor"),
        # ("env_kwargs", "random_r_factor"),
        ("policy_kwargs", "init_log_std"),
        ("env_kwargs", "knn_k"),
        ("env_kwargs", "voxel_conf"),
        ("env_kwargs", "obs_type"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # ["action"],
        ["policy"],
        # ["agent"],
        # ["explore"],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("sample_method", ), ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # get all variants and their own log directory
    variants, log_dirs = make_variants(*variant_levels)
    for i, variant in enumerate(variants):
        # print(variant)
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
    parser.add_argument(
        '--obj', help= 'obj',
        type= int, default= 0,
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