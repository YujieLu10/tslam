''' A demo file telling you how to configure variant and launch experiments
    NOTE: By running this demo, it will create a `data` directory from where you run it.
'''
from exptools.launching.variant import VariantLevel, make_variants, update_config
import numpy as np

seed = 123
default_config = dict(
    env_name = "adroit-v2", # adroit-v0 heuristic adroit-v2: our best policy + coverage + curiosity # adroit-v3: variant using knn reward or chamfer reward # adroit-v4: new points reward and only touch reward
    env_kwargs = dict(
        obj_orientation= [0, 0, 0], # object orientation
        obj_relative_position= [0, 0.5, 0.07], # object\ position related to hand (z-value will be flipped when arm faced down)
        goal_threshold= int(8e3), # how many points touched to achieve the goal
        new_point_threshold= 0.001, # minimum distance new point to all previous points
        forearm_orientation_name= "up", # ("up", "down")
        # scale all reward/penalty to the scale of 1.0
        chamfer_r_factor= 0,
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
        knn_r_factor= 0,
        new_voxel_r_factor= 0,
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
    sample_method = "agent", # `action`:env.action_space.sample(), `policy`
    policy_path = "",
    total_timesteps = int(800),
    seed= seed,
)

def main(args):
    experiment_title = "agent" #"sample_pointclouds"

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

    # tabletop
    values = [
        ["glass", 0.015, [0, 0, 0], [0, 0, 0.05510244]], # 0.06
        ["donut", 0.01, [0, 0, 0], [0, 0, 0.01466367]],
        ["heart", 0.0006, [0.70738827, -0.70682518, 0], [0, 0, 0.8]],
        ["airplane", 1, [0, 0, 0], [0, 0, 2.58596408e-02]],
        ["alarmclock", 1, [0.70738827, -0.70682518, 0], [0, 0, 2.47049890e-02]],
        ["apple", 1, [0, 0, 0], [0, 0, 0.04999409]],
        ["banana", 1, [0, 0, 0], [0, 0, 0.02365614]],
        ["binoculars", 1, [0.70738827, -0.70682518, 0], [0, 0, 0.07999943]],
        ["body", 0.1, [0, 0, 0], [0, 0, 0.0145278]],
        ["bowl", 1, [0, 0, 0], [0, 0, 0.03995771]],
        ["camera", 1, [0, 0, 0], [0, 0, 0.03483407]],
        ["coffeemug", 1, [0, 0, 0], [0, 0, 0.05387171]],
        ["cubelarge", 1, [0, 0, 0], [0, 0, 0.06039196]],
        ["cubemedium", 1, [0, 0, 0], [0, 0, 0.04103902]],
        ["cubemiddle", 1, [0, 0, 0], [0, 0, 0.04103902]],
        ["cubesmall", 1, [0, 0, 0], [0, 0, 0.02072159]],
        ["cup", 1, [0, 0, 0], [0, 0, 0.05127277]],
        ["cylinderlarge", 1, [0, 0, 0], [0, 0, 0.06135697]],
        ["cylindermedium", 1, [0, 0, 0], [0, 0, 0.04103905]],
        ["cylindersmall", 1, [0, 0, 0], [0, 0, 0.02072279]],
        ["doorknob", 1, [0, 0, 0], [0, 0, 0.0379012]],
        ["duck", 1, [0, 0, 0], [0, 0, 0.04917608]],
        ["elephant", 1, [0, 0, 0], [0, 0, 0.05097572]],
        ["eyeglasses", 1, [0, 0, 0], [0, 0, 0.02300015]],
        ["flashlight", 1, [0, 0, 0], [0, 0, 0.07037258]],
        ["flute", 1, [0, 0, 0], [0, 0, 0.0092959]],
        ["fryingpan", 0.8, [0, 0, 0], [0, 0, 0.01514528]],
        ["gamecontroller", 1, [0, 0, 0], [0, 0, 0.02604568]],
        ["hammer", 1, [0, 0, 0], [0, 0, 0.01267463]],
        ["hand", 1, [0, 0, 0], [0, 0, 0.07001909]],
        ["headphones", 1, [0, 0, 0], [0, 0, 0.02992321]],
        ["knife", 1, [0, 0, 0], [0, 0, 0.00824503]],
        ["lightbulb", 1, [0, 0, 0], [0, 0, 0.03202522]],
        ["mouse", 1, [0, 0, 0], [0, 0, 0.0201307]],
        ["mug", 1, [0, 0, 0], [0, 0, 0.05387171]],
        ["phone", 1, [0, 0, 0], [0, 0, 0.02552063]],
        ["piggybank", 1, [0, 0, 0], [0, 0, 0.06923257]],
        ["pyramidlarge", 1, [0, 0, 0], [0, 0, 0.05123203]],
        ["pyramidmedium", 1, [0, 0, 0], [0, 0, 0.04103812]],
        ["pyramidsmall", 1, [0, 0, 0], [0, 0, 0.02072198]],
        ["rubberduck", 1, [0, 0, 0], [0, 0, 0.04917608]],
        ["scissors", 1, [0, 0, 0], [0, 0, 0.00802606]],
        ["spherelarge", 1, [0, 0, 0], [0, 0, 0.05382598]],
        ["spheremedium", 1, [0, 0, 0], [0, 0, 0.03729011]],
        ["spheresmall", 1, [0, 0, 0], [0, 0, 0.01897534]],
        ["stamp", 1, [0, 0, 0], [0, 0, 0.0379012]],
        ["stanfordbunny", 1, [0, 0, 0], [0, 0, 0.06453102]],
        ["stapler", 1, [0, 0, 0], [0, 0, 0.02116039]],
        ["table", 5, [0, 0, 0], [0, 0, 0.01403165]],
        ["teapot", 1, [0, 0, 0], [0, 0, 0.05761634]],
        ["toothbrush", 1, [0, 0, 0], [0, 0, 0.00701304]],
        ["toothpaste", 1, [0, 1.5, 0], [0, 0, 0.02]],
        ["toruslarge", 1, [0, 0, 0], [0, 0, 0.02080752]],
        ["torusmedium", 1, [0, 0, 0], [0, 0, 0.01394647]],
        ["torussmall", 1, [0, 0, 0], [0, 0, 0.00734874]],
        ["train", 1, [0, 0, 0], [0, 0, 0.04335064]],
        ["watch", 1, [0, 0, 0], [0, 0, 0.0424445]],
        ["waterbottle", 1, [0, 0, 0], [0, 0, 0.08697578]],
        ["wineglass", 1, [0, 0, 0], [0, 0, 0.0424445]],
        ["wristwatch", 1, [0, 0, 0], [0, 0, 0.06880109]],
    ]
    dir_names = ["obj{}".format(*tuple(str(vi) for vi in v[0:1])) for v in values]
    keys = [
        ("env_kwargs", "obj_name"),
        ("env_kwargs", "obj_scale"),
        ("env_kwargs", "obj_orientation"),
        ("env_kwargs", "obj_relative_position"),
    ] # each entry in the list is the string path to your config
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # [True, False, "unified3d1", [-1.57, 0, 3.14151926],  [0, -0.56, 0.1]], # unified3d
        [True, False, "unified3d2", [1.57, 0, 0],  [0, 0.56, 0.1]], # unified3d
        # [True, False, "unified3d3", [-1.57, -1.57, 3.14151926],  [0.55, 0, 0.1]], # unified3d
        # [True, False, "unified3d4", [-1.57, 1.57, 3.14151926],  [-0.55, 0, 0.1]], # unified3d
        # [True, False, "unifiedpose1", [-1.57, 0, 3.14151926],  [0, -0.56, 0.2]], # fix voxel grid downback
        # [True, False, "unifiedpose2", [1.57, 0, 0],  [0, 0.56, 0.3]],
        # [True, False, "unifiedpose3", [-1.57, -1.57, 3.14151926],  [0.55, 0, 0.2]],
        # [True, False, "unifiedpose4", [-1.57, 1.57, 3.14151926],  [-0.55, 0, 0.2]],
        # [True, False, "10kdownfront", [0, 0, 0],  [0, -0.14, 0.23], [1.57, 0, 0],  [0, 0.4, 0.3]],
        # [True, False, "10kdownright", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, -1.57, 3.14151926],  [0.55, -0.15, 0.3]],
        # [True, False, "10kdownleft", [0, 0, 0],  [0, -0.14, 0.23], [-1.57, 1.57, 3.14151926],  [-0.55, -0.15, 0.3]],
    ]
    dir_names = ["voxel{}_rw{}_orien{}".format(*tuple(str(vi) for vi in v[0:3])) for v in values]
    keys = [
        ("env_kwargs", "use_voxel"),
        ("policy_kwargs", "reinitialize"),
        ("env_kwargs", "forearm_orientation_name"),
        ("env_kwargs", "forearm_orientation"),
        ("env_kwargs", "forearm_relative_position"),
    ] # each entry in the list is the string path to your config
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # reward setting and voxel observatoin mode
    values = [
        # [0, 0, 1, 0.5, 5, ['3d', 6], [True, False]], # best policy | random
        # [0, 1, 0, 0.5, 5, ['3d', 6], [True, False]], # knn variant | ntouch
        # [0, 0, 0, 0.5, 5, ['3d', 6], [True, False]], # heuristic
        # [1, 0, 0, 0.5, 5, ['3d', 6], [True, False]], # chamfer variant | npoint
        [1, 3, 0.5, ['3d', 6], [True, False]], # cur & cove : ours
        # [1, 0.5, ['3d', 6], [True, False]], # disagreef
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

    values = [
        # ["action"],
        # ["policy"], # random
        ["agent"],
        # ["heuristic"],
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