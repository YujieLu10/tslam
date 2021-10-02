import os
import numpy as np
import math
import voxels
from voxels import VoxelGrid
import shutil
from numpy.lib.npyio import save
import matplotlib.pyplot as plt
import random
import argparse

parser = argparse.ArgumentParser(
    description='Run generation'
)

parser.add_argument('-clear_files' , default=False, type=bool)

args = parser.parse_args()
is_clear = args.clear_files

iternum_list = [199] #[99,199,299,399,499]
# iternum_list = [9, 19, 39, 169, 199, 259, 399, 599, 899, 1049, 1099, 1139, 1499, 1999, 2199] # for trajectory
# pose_num_list = ["eight", "four"]
pose_num_list = ["unified"]
test_obj_list = ["airplane", "cup", "lightbulb"]
policy_type = ["curf1covf3"]
is_save_png = True
vis_root = "exp" # exp uniform_gt exp two_pose long_step
# obj_relative_position_down = [0, -0.14, 0.23]#[0, -0.12, 0.23]
# obj_relative_position_up = [0, -0.14, 0.23]
obj_relative_position_down = [0, 0, 0.03]#[0, -0.12, 0.23]
obj_relative_position_up = [0, 0, 0.03]
eval_dir = "agent_eval_rebbuttal"
res_list = [32]
save_root = "../../prox/tslam/data/result/{}_standard_voxel/{}".format(eval_dir, vis_root)

def get_transform_gt_data(origin_gt_data, obj_name, obj_relative_position):
    obj_to_scale_map = {'duck':1, 'watch':1, 'doorknob':1, 'headphones':1, 'bowl':1, 'cubesmall':1, 'spheremedium':1, 'train':1, 'piggybank':1, 'cubemedium':1, 'cubelarge':1, 'elephant':1, 'flute':1, 'wristwatch':1, 'pyramidmedium':1, 'gamecontroller':1, 'toothbrush':1, 'pyramidsmall':1, 'body':0.1, 'cylinderlarge':1, 'cylindermedium':1, 'cylindersmall':1, 'fryingpan':0.8, 'stanfordbunny':1, 'scissors':1, 'pyramidlarge':1, 'stapler':1, 'flashlight':1, 'mug':1, 'hand':1, 'stamp':1, 'rubberduck':1, 'binoculars':1, 'apple':1, 'mouse':1, 'eyeglasses':1, 'airplane':1, 'coffeemug':1, 'cup':1, 'toothpaste':1, 'torusmedium':1, 'cubemiddle':1, 'phone':1, 'torussmall':1, 'spheresmall':1, 'knife':1, 'banana':1, 'teapot':1, 'hammer':1, 'alarmclock':1, 'waterbottle':1, 'camera':1, 'table':0.05, 'wineglass':1, 'lightbulb':1, 'spherelarge':1, 'toruslarge':1, 'glass':0.015, 'heart':0.0006, 'donut':0.01}
    obj_scale = obj_to_scale_map[obj_name]
    data_scale = origin_gt_data * obj_scale
    obj_orientation = [0, 0, 0]
    # obj_relative_position = [0, 0, 0] # keep center [0,0,0]
    if obj_name == "heart":
        obj_orientation = [-1.57, 0, 0]
    data_rotate = data_scale.copy()
    x = data_rotate[:, 0].copy()
    y = data_rotate[:, 1].copy()
    z = data_rotate[:, 2].copy()
    x_theta = obj_orientation[0]
    data_rotate[:, 0] = x
    data_rotate[:, 1] = y*math.cos(x_theta) - z*math.sin(x_theta)
    data_rotate[:, 2] = y*math.sin(x_theta) + z*math.cos(x_theta)
    data_trans = data_rotate.copy()
    data_trans[:, 0] += obj_relative_position[0]
    data_trans[:, 1] += obj_relative_position[1]
    data_trans[:, 2] += obj_relative_position[2]
    uniform_gt_data = data_trans.copy()
    return uniform_gt_data

for root, dirs, files in os.walk("../../prox/tslam/data/result/{}/".format(eval_dir)):
    is_hit_type = False
    for ptype in policy_type:
        if ptype in root: is_hit_type = True
        is_obj_test = False
        for obj_test in test_obj_list:
            if obj_test in root: is_obj_test = True
        if is_obj_test and is_hit_type and "unifiedpose" in root: # paper first version "10kpose" # combined pose glass_geneTrue_rotTrue_down_normal_cf0knn0voxel1 type
            voxel_cls = root[root.index(eval_dir)+(len(eval_dir))+1:root.index('/gene')]
            voxel_config = root[root.index('/gene')+1:].replace('/','_')
            save_path = os.path.join(save_root, voxel_cls, voxel_config) if vis_root != "uniform_gt" else os.path.join(save_root, voxel_cls)
            if is_clear:
                if os.path.isdir(save_path) == True:
                    shutil.rmtree(save_path)
                    print(">>> clear file {}!".format(save_path))
                continue
            if os.path.isdir(save_path) == False: os.makedirs(save_path)
            log_file = open(os.path.join(save_path, "log.txt"), "a")
            for pose_num_str in pose_num_list:
                for res in res_list:
                    for iternum in iternum_list:#range(499, 10000, 500):
                        save_trans_file_path = os.path.join(save_path, "iternum_{}_{}_{}pose_trans_voxelization_{}.npy".format(iternum, vis_root, pose_num_str, res))
                        if os.path.exists(save_trans_file_path):
                            if is_clear:
                                os.remove(save_trans_file_path)
                                print(">>> file {} remove and generate!".format(save_trans_file_path))
                            else:
                                print(">>> file {} exists skip!".format(save_trans_file_path))
                                continue
                        # load uniform gt
                        uniform_gt_data = np.load("/home/jianrenw/prox/tslam/assets/uniform_gt/uniform_{}_o3d.npz".format(voxel_cls))['pcd']
                        vis_gt_data = get_transform_gt_data(uniform_gt_data, voxel_cls, obj_relative_position_down if "down" in root else obj_relative_position_up)
                        # [load by iternum] load pcd each iter
                        recon_pcd_file = os.path.join(root, "iternum_{}_{}pose_alpha_pointcloud.npz".format(iternum, pose_num_str))
                        # ============================================
                        # [load by overlap] load pcd of max overlap with ground truth
                        # max_overlap = 0
                        # min_overlap = 1
                        # for file in files:
                        #     if ".npz" in file:
                        #         file_str = str(file)
                        #         previous_o
                        # ccup = 1 if "pose" in root else file_str[(file_str.index("-")+1):file_str.index(".npz")]
                        #         if not "random1" in root and float(previous_occup) > max_overlap:
                        #             max_overlap = float(previous_occup)
                        #             recon_pcd_file = os.path.join(root, file)
                        #         elif "random1" in root and float(previous_occup) <= min_overlap: # random
                        #             min_overlap = float(previous_occup)
                        #             recon_pcd_file = os.path.join(root, file)
                        # ============================================
                        if not os.path.exists(recon_pcd_file):
                            print(">>> recon_pcd_file{} not exists skip!".format(recon_pcd_file))
                            continue
                        file_path = recon_pcd_file
                        vis_exp_data = np.load(file_path)['pcd']
                        #=============== pcd covert to voxel grid : 0.25 * 0.25 * 0.25 grid [-0.125, -0.125+0.25] [-0.25 - 0.0125, -0.25+0.25] [0.16 - 0.075, 0.16 + 0.25]
                        resolution = 0.25 / res
                        x, y, z = np.indices((res, res, res))
                        gt_occupancies = np.zeros((res,res,res))
                        gt_map_list = []
                        gt_voxels = None
                        for idx,val in enumerate(vis_gt_data):
                            idx_x = math.floor((val[0] + 0.125) / resolution)
                            idx_y = math.floor((val[1] + 0.125) / resolution)
                            idx_z = math.floor((val[2] + 0.125) / resolution) 
                            name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
                            if name not in gt_map_list:
                                gt_map_list.append(name)
                            cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
                            gt_voxels = cube if gt_voxels is None else (gt_voxels + cube)
                            if idx_x < 32 and idx_y < 32 and idx_z < 32 and idx_x >= 0 and idx_y >= 0 and idx_z >= 0:
                                gt_occupancies[idx_x, idx_y, idx_z] = 1
                            else:
                                print("out of bound {} {} {}".format(idx_x, idx_y, idx_z))
                        #===============
                        if is_save_png:
                            gt_colors = np.empty(gt_voxels.shape, dtype=object)
                            gt_colors[gt_voxels] = 'cyan'
                            ax = plt.figure().add_subplot(projection='3d')
                            ax.set_zlim(1,res)
                            ax.voxels(gt_voxels, facecolors=gt_colors, edgecolor='g', alpha=.4, linewidth=.05)
                            plt.savefig(os.path.join(save_path, "uniform_gt_voxelization_{}.png".format(res)))
                            plt.close()
                        #===============
                        if vis_root != "uniform_gt":
                            exp_occupancies = np.zeros((res,res,res))
                            exp_map_list = []
                            exp_voxels = None
                            for idx,val in enumerate(vis_exp_data):
                                idx_x = math.floor((val[0] + 0.125) / resolution)
                                idx_y = math.floor((val[1] + 0.125) / resolution)
                                idx_z = math.floor((val[2] + 0.125) / resolution) 
                                #===============
                                name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
                                if name not in exp_map_list and name in gt_map_list:
                                    exp_map_list.append(name)
                                    cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
                                    # combine the objects into a single boolean array
                                    exp_voxels = cube if exp_voxels is None else (exp_voxels + cube)
                                    if idx_x < 32 and idx_y < 32 and idx_z < 32:
                                        exp_occupancies[idx_x, idx_y, idx_z] = 1
                            #===============
                            if is_save_png:
                                exp_colors = np.empty(exp_voxels.shape, dtype=object)
                                exp_colors[exp_voxels] = 'white'
                                ax = plt.figure().add_subplot(projection='3d')
                                ax.set_zlim(1,res)
                                ax.voxels(exp_voxels, facecolors=gt_colors, edgecolor='g', alpha=.4, linewidth=.05)
                                plt.savefig(os.path.join(save_path, "{}_voxelization_{}_iter{}_pose{}.png".format(vis_root, res, iternum, pose_num_str)))
                                plt.close()
                            # === write log
                            log_file.write(">>> gt_voxel: {} exp_voxel: {}".format(len(gt_occupancies[np.where(gt_occupancies == 1)]), len(exp_occupancies[np.where(exp_occupancies == 1)])))
                        #=============== save voxelized pcd
                        save_occupancies = exp_occupancies.copy() if vis_root != "uniform_gt" else gt_occupancies.copy()
                        if len(save_occupancies[np.where(save_occupancies == 1)]) <= 0:
                            log_file.write(">>> save_occupancies zero skip {}".format(save_trans_file_path) + "\n")
                            print(">>> save_occupancies zero skip {}".format(save_trans_file_path))
                            continue

                        # ========= save trans voxel
                        min = -0.5
                        max = 0.5
                        loc = ((min+max)/2, )*3
                        scale = max - min
                        mesh = VoxelGrid(save_occupancies, loc, scale).to_mesh()
                        # ========= mesh center (bias)
                        # total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
                        # centers = (mesh.bounds[1] + mesh.bounds[0]) /2
                        # mesh.apply_translation(-centers)
                        # mesh.apply_scale(1/total_size)
                        # centers = [0, 0.1, 0.23]#obj_relative_position_down
                        centers = [0, 0, 0.03]#obj_relative_position_down
                        mesh.apply_translation(centers)
                        
                        save_occupancies = voxels.VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
                        save_occupancies = np.reshape(save_occupancies, -1)
                        save_occupancies = np.packbits(save_occupancies)
                        print(">>> save {}".format(save_trans_file_path))
                        log_file.write(">>> save {}".format(save_trans_file_path) + "\n")
                        np.save(save_trans_file_path, save_occupancies)
                # log_file.close()
