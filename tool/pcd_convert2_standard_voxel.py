import os
import numpy as np
import math

from numpy.lib.npyio import save

res = 32
def get_transform_gt_data(origin_gt_data, obj_name):
    obj_to_scale_map = {'duck':1, 'watch':1, 'doorknob':1, 'headphones':1, 'bowl':1, 'cubesmall':1, 'spheremedium':1, 'train':1, 'piggybank':1, 'cubemedium':1, 'cubelarge':1, 'elephant':1, 'flute':1, 'wristwatch':1, 'pyramidmedium':1, 'gamecontroller':1, 'toothbrush':1, 'pyramidsmall':1, 'body':0.1, 'cylinderlarge':1, 'cylindermedium':1, 'cylindersmall':1, 'fryingpan':0.8, 'stanfordbunny':1, 'scissors':1, 'pyramidlarge':1, 'stapler':1, 'flashlight':1, 'mug':1, 'hand':1, 'stamp':1, 'rubberduck':1, 'binoculars':1, 'apple':1, 'mouse':1, 'eyeglasses':1, 'airplane':1, 'coffeemug':1, 'cup':1, 'toothpaste':1, 'torusmedium':1, 'cubemiddle':1, 'phone':1, 'torussmall':1, 'spheresmall':1, 'knife':1, 'banana':1, 'teapot':1, 'hammer':1, 'alarmclock':1, 'waterbottle':1, 'camera':1, 'table':0.05, 'wineglass':1, 'lightbulb':1, 'spherelarge':1, 'toruslarge':1, 'glass':0.015, 'heart':0.0006, 'donut':0.01}
    obj_scale = obj_to_scale_map[obj_name]
    data_scale = origin_gt_data * obj_scale
    obj_orientation = [0, 0, 0]
    obj_relative_position = [0, -0.14, 0.23]
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

save_root = "standard_voxel"
for root, dirs, files in os.walk("best_eval/"):
    if "voxel" in root and "airplane" in root:
        # glass_geneTrue_rotTrue_down_normal_cf0knn0voxel1 type
        voxel_cls = root[root.index('/')+1:root.index('/gene')]
        voxel_config = root[root.index('/gene')+1:].replace('/','_')
        save_path = os.path.join(save_root, voxel_cls, voxel_config)
        print(save_path)
        if os.path.isdir(save_path) == False: os.makedirs(save_path)
        # load pcd of max overlap with ground truth
        max_overlap = 0
        for file in files:
            if ".npz" in file:
                file_str = str(file)
                previous_occup = file_str[(file_str.index("-")+1):file_str.index(".npz")]
                if float(previous_occup) > max_overlap:
                    max_overlap = float(previous_occup)
                    recon_pcd_file = os.path.join(root, file)
        file_path = recon_pcd_file
        # pcd_data = np.load(file_path)['pcd']
        uniform_gt_data = np.load("/home/jianrenw/prox/tslam/assets/uniform_gt/uniform_{}_o3d.npz".format(voxel_cls))['pcd']
        uniform_gt_data = get_transform_gt_data(uniform_gt_data, voxel_cls)
        # pcd covert to voxel grid : 0.25 * 0.25 * 0.25
        # grid [-0.125, -0.125+0.25] [-0.25 + 0.0125, -0.25+0.25] [0.16 + 0.075, 0.16 + 0.25]
        resolution = 0.25 / res
        occupancies = np.zeros((res,res,res))
        for idx,val in enumerate(uniform_gt_data):
            idx_x = math.floor((val[0] + 0.125) / resolution)
            idx_y = math.floor((val[1] + 0.2375) / resolution)
            idx_z = math.floor((val[2] - 0.235) / resolution) 
            occupancies[idx_x, idx_y, idx_z] = 1
        # save voxelized pcd
        save_file_path = os.path.join(save_path, "uniformgt_voxelization_{}.npy".format(res))
        np.save(save_file_path, occupancies)
    