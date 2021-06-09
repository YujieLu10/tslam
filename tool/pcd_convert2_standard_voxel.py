import os
import numpy as np
import math
import voxels
from voxels import VoxelGrid

from numpy.lib.npyio import save
import matplotlib.pyplot as plt

vis_root = "exp"
res = 32
save_root = "../standard_voxel/{}".format(vis_root)

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

for root, dirs, files in os.walk("../best_eval/"):
    if "voxel" in root and "geneTrue_rotTrue_down" in root and "normal_cf0knn0voxel1" in root:
        # glass_geneTrue_rotTrue_down_normal_cf0knn0voxel1 type
        voxel_cls = root[root.index('best_eval/')+10:root.index('/gene')]
        voxel_config = root[root.index('/gene')+1:].replace('/','_')
        save_path = os.path.join(save_root, voxel_cls, voxel_config)

        if os.path.isdir(save_path) == False: os.makedirs(save_path)
        if vis_root == "uniform_gt":
            # load uniform gt
            uniform_gt_data = np.load("/home/jianrenw/prox/tslam/assets/uniform_gt/uniform_{}_o3d.npz".format(voxel_cls))['pcd']
            vis_data = get_transform_gt_data(uniform_gt_data, voxel_cls)
        else:
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
            vis_data = np.load(file_path)['pcd']
        # pcd covert to voxel grid : 0.25 * 0.25 * 0.25
        # grid [-0.125, -0.125+0.25] [-0.25 - 0.0125, -0.25+0.25] [0.16 - 0.075, 0.16 + 0.25]
        resolution = 0.25 / res
        x, y, z = np.indices((res, res, res))
        occupancies = np.zeros((res,res,res))
        gt_map_list = []
        gt_voxels = None
        for idx,val in enumerate(vis_data):
            # idx_x = math.floor((val[0] + 0.155) / resolution)
            # idx_y = math.floor((val[1] + 0.2625) / resolution)
            # idx_z = math.floor((val[2] - 0.215) / resolution) 
            idx_x = math.floor((val[0] + 0.125) / resolution)
            idx_y = math.floor((val[1] + 0.25) / resolution)
            idx_z = math.floor((val[2] - 0.16) / resolution) 
            #===============
            name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
            if name not in gt_map_list:
                gt_map_list.append(name)
            cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
            # combine the objects into a single boolean array
            gt_voxels = cube if gt_voxels is None else (gt_voxels + cube)
            #===============
            # print(">>> idx {} idx {} idx {}".format(idx_x, idx_y, idx_z))
            occupancies[idx_x, idx_y, idx_z] = 1
        #===============
        # gt_colors = np.empty(gt_voxels.shape, dtype=object)
        # gt_colors[gt_voxels] = 'white'
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.set_zlim(1,20)
        # ax.voxels(gt_voxels, facecolors=gt_colors, edgecolor='g', alpha=.4, linewidth=.05)
        # plt.savefig(os.path.join(save_path, "{}_voxelization_{}.png".format(vis_root, res)))
        # plt.close()
        #===============
        # save voxelized pcd
        # occupancies = np.zeros((res,res,res))
        # for i in range(10,20):
        #     for j in range(10,20):
        #         for k in range(15,20):
        #             occupancies[i,j,k] = 1
        # save_file_path = os.path.join(save_path, "{}_voxelization_{}.npy".format(vis_root, res))
        # # print(len(occupancies[np.where(occupancies == 1)]))
        # np.save(save_file_path, occupancies)

        # save reshaped voxel
        print(save_path)
        save_reshaped_file_path = os.path.join(save_path, "{}_trans_voxelization_{}.npy".format(vis_root, res))
        min = -0.5
        max = 0.5
        loc = ((min+max)/2, )*3
        scale = max - min

        mesh = VoxelGrid(occupancies, loc, scale).to_mesh()
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) /2
        mesh.apply_translation(-centers)
        # mesh.apply_scale(1/total_size)
        occupancies = voxels.VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
        occupancies = np.reshape(occupancies, -1)
        occupancies = np.packbits(occupancies)
        np.save(save_reshaped_file_path, occupancies)