import os
import numpy as np
import shutil
import math
import matplotlib.pyplot as plt

policy_type = ["voxel1"] #["knn1", "cf1", "voxel1", "random1", "npoint1", "ntouch1"]
clear_files = False
res = 32
vis_root = "exp" # exp uniform_gt two_pose long_step
eval_dir = "best_eval"
save_root = "../data/result/{}/".format(eval_dir)

for root, dirs, files in os.walk("../data/result/{}/".format(eval_dir)):
    is_hit_type = False
    for ptype in policy_type:
        is_hit_type = True if ptype in root else False
    if "voxel" in root and is_hit_type and not "pose" in root:
        voxel_cls = root[root.index(eval_dir)+(len(eval_dir))+1:root.index('/gene')]
        voxel_config = root[root.index('/gene')+1:].replace('/','_').replace('down', 'pose').replace('up', 'pose')
        save_path = os.path.join(save_root, voxel_cls, voxel_config)
        # print(root)
        if clear_files:
            print(">>> clear files {}".format(save_path))
            if os.path.isdir(save_path) == True: shutil.rmtree(save_path)
        else:
            if os.path.isdir(save_path) == False: os.makedirs(save_path)
            # ======== load max overlap file
            max_overlap = 0
            for file in files:
                if ".npz" in file:
                    file_str = str(file)
                    previous_occup = file_str[(file_str.index("-")+1):file_str.index(".npz")]
                    if float(previous_occup) > max_overlap:
                        max_overlap = float(previous_occup)
                        recon_pcd_file = os.path.join(root, file)
            file_path = recon_pcd_file
            vis_data = np.load(file_path)['pcd']
            save_file_path = os.path.join(save_path, "twopose_alpha_pointcloud.npz")
            save_imgfile_path = os.path.join(save_path, "twopose_alpha_pointcloud.png")
            # print(save_file_path)
            if "down" in root:
                # obj_relative_position = [0, -0.12, 0.23]
                obj_relative_position = [0, -0.2, 0] # move to up pos
            # else:
            #     obj_relative_position = [0, -0.14, 0.23]
                vis_data[:, 0] += obj_relative_position[0]
                vis_data[:, 1] += obj_relative_position[1]
                vis_data[:, 2] += obj_relative_position[2]

            if os.path.exists(save_file_path):
                exist_data = np.load(save_file_path)['pcd']
                concat_data = np.concatenate([vis_data, exist_data])
                np.savez(save_file_path, pcd=concat_data)
                print(">>> exist {} vis {} final{}".format(len(exist_data), len(vis_data), len(concat_data)))
            else:
                print(">>> save {}".format(save_file_path))
                np.savez(save_file_path, pcd=vis_data)
