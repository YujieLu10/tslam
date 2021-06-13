import os
import numpy as np
import shutil
import math
import matplotlib.pyplot as plt
import random

policy_type = ["voxel1"] #["knn1", "cf1", "voxel1", "random1", "npoint1", "ntouch1"]
clear_files = False
test_obj_list = ["airplane", "cup", "spherelarge", "body", "fryingpan", "lightbulb"]
vis_root = "exp" # exp uniform_gt two_pose long_step
eval_dir = "agent_eval"
save_root = "../data/result/{}/".format(eval_dir)

for root, dirs, files in os.walk("../data/result/{}/".format(eval_dir)):
    is_hit_type = False
    for ptype in policy_type:
        is_hit_type = True if ptype in root else False
    is_obj_test = False
    for obj_test in test_obj_list:
        if obj_test in root: is_obj_test = True
    if is_obj_test and is_hit_type and not "pose" in root and not "downleft" in root and not "downright" in root and not "upfront" in root and not "upback" in root:
        voxel_cls = root[root.index(eval_dir)+(len(eval_dir))+1:root.index('/gene')]
        voxel_config = root[root.index('/gene')+1:].replace('/','_').replace("downback", "pose").replace('downfront', 'pose').replace('downleft', 'pose').replace('downright', 'pose').replace('upback','pose').replace('upfront', 'pose').replace('upleft', 'pose').replace('upright', 'pose')
        save_path = os.path.join(save_root, voxel_cls, voxel_config)
        # print(root)
        if clear_files:
            print(">>> clear files {}".format(save_path))
            if os.path.isdir(save_path) == True: shutil.rmtree(save_path)
        else:
            if os.path.isdir(save_path) == False: os.makedirs(save_path)
            if not eval_dir == "agent_eval":
                # ======== load max overlap file
                max_overlap = 0
                min_overlap = 1
                # if "random1" in root: # random no best policy random choose
                #     npz_files = []
                #     for file in files:
                #         if ".npz" in file:
                #             npz_files.append(file)
                #     recon_pcd_file = os.path.join(root, random.choice(npz_files))
                for file in files:
                    if ".npz" in file:
                        file_str = str(file)
                        previous_occup = file_str[(file_str.index("-")+1):file_str.index(".npz")]
                        if not "random1" in root and float(previous_occup) > max_overlap:
                            max_overlap = float(previous_occup)
                            recon_pcd_file = os.path.join(root, file)
                        elif "random1" in root and float(previous_occup) <= min_overlap: # random
                            min_overlap = float(previous_occup)
                            recon_pcd_file = os.path.join(root, file)
                file_path = recon_pcd_file
                vis_data = np.load(file_path)['pcd']
                save_file_path = os.path.join(save_path, "fourpose_alpha_pointcloud.npz")
                # save_imgfile_path = os.path.join(save_path, "twopose_alpha_pointcloud.png")

                # ========= save file
                if os.path.exists(save_file_path):
                    exist_data = np.load(save_file_path)['pcd']
                    concat_data = np.concatenate([vis_data, exist_data])
                    np.savez(save_file_path, pcd=concat_data)
                    print(">>> exist {} vis {} final{}".format(len(exist_data), len(vis_data), len(concat_data)))
                else:
                    print(">>> save {}".format(save_file_path))
                    np.savez(save_file_path, pcd=vis_data)
            else: # save each iter
                for file in files:
                    if ".npz" in file and not "pose_alpha" in file:
                        file_str = str(file)
                        try:
                            iter_num = int(file_str[(file_str.index("iternum_")+8):file_str.index("_pointcloud")])
                        except:
                            print(file_str[(file_str.index("iternum_")+9):file_str.index("_pointcloud")])
                        if (iter_num + 1) % 500 == 0:
                            recon_pcd_file = os.path.join(root, file)
                            file_path = recon_pcd_file
                            vis_data = np.load(file_path)['pcd']
                            save_file_path = os.path.join(save_path, "iternum_{}_fourpose_alpha_pointcloud.npz".format(iter_num))
                            # save_imgfile_path = os.path.join(save_path, "twopose_alpha_pointcloud.png")
                            # ========= save file
                            if os.path.exists(save_file_path):
                                exist_data = np.load(save_file_path)['pcd']
                                concat_data = np.concatenate([vis_data, exist_data])
                                np.savez(save_file_path, pcd=concat_data)
                                # print(">>> exist {} vis {} final{}".format(len(exist_data), len(vis_data), len(concat_data)))
                            else:
                                print(">>> save {}".format(save_file_path))
                                np.savez(save_file_path, pcd=vis_data)
