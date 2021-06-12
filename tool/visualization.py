import matplotlib.pyplot as plt
import numpy as np
import math
import os

res = 32
# vis_data_tuple [obj_name, obj_orien, obj_pos, obj_scale, pc_frame, iternum]
def save_voxel_visualization(vis_data_tuple, save_path):#env_args, pc_frame, iternum, is_best_policy):
    obj_name, obj_orientation, obj_relative_position, obj_scale, pc_frame, iternum = vis_data_tuple[0], vis_data_tuple[1], vis_data_tuple[2], vis_data_tuple[3], vis_data_tuple[4], vis_data_tuple[5]
    uniform_gt_data = np.load("/home/jianrenw/prox/tslam/assets/uniform_gt/uniform_{}_o3d.npz".format(obj_name))['pcd']
    data_scale = uniform_gt_data * obj_scale
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
    data = pc_frame
    resolution = 0.25 / res
    x, y, z = np.indices((res, res, res))

    voxels = None
    gt_voxels = None

    # draw gt
    gt_map_list = []
    for idx,val in enumerate(uniform_gt_data):
        idx_x = math.floor((val[0] + 0.125) / resolution)
        idx_y = math.floor((val[1] + 0.25) / resolution)
        idx_z = math.floor((val[2] - 0.16) / resolution)
        name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
        if name not in gt_map_list:
            gt_map_list.append(name)
            cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
            # combine the objects into a single boolean array
            gt_voxels = cube if gt_voxels is None else (gt_voxels + cube)

    # =========== save gt
    if not os.path.exists(os.path.join(save_path, "voxel_gt.png")):
        gt_colors = np.empty(gt_voxels.shape, dtype=object)
        gt_colors[gt_voxels] = 'white'
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_zlim(1,res)
        ax.voxels(gt_voxels, facecolors=gt_colors, edgecolor='g', alpha=.4, linewidth=.05)
        plt.savefig(os.path.join(save_path, "voxel_gt.png"))
        plt.close()
    # draw cuboids in the top left and bottom right corners, and a link between them
    map_list = []
    for idx,val in enumerate(data):
        idx_x = math.floor((val[0] + 0.125) / resolution)
        idx_y = math.floor((val[1] + 0.25) / resolution)
        idx_z = math.floor((val[2] - 0.16) / resolution)
        name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
        if name not in map_list and name in gt_map_list:
            map_list.append(name)
            cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
            # combine the objects into a single boolean array
            voxels = cube if voxels is None else (voxels + cube)

    occupancy = len(map_list) / len(gt_map_list)
    # =========== save exp
    exp_colors = np.empty(voxels.shape, dtype=object)
    exp_colors[voxels] = 'cyan'
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_zlim(1,res)
    ax.voxels(voxels, facecolors=exp_colors, edgecolor='g', alpha=.4, linewidth=.05)
    plt.savefig(os.path.join(save_path, "iternum_" + str(iternum) + "_voxel_exp_overlap-{}.png".format(occupancy)))
    plt.close()
    # =========== save overlap
    # set the colors of each object
    vis_voxel = gt_voxels | voxels
    colors = np.empty(vis_voxel.shape, dtype=object)
    colors[gt_voxels] = 'white'
    colors[voxels] = 'cyan'
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_zlim(1,20)
    ax.voxels(vis_voxel, facecolors=colors, edgecolor='g', alpha=.4, linewidth=.05)
    plt.savefig(os.path.join(save_path, "iternum_" + str(iternum) + "_voxel_overlap-{}.png".format(occupancy)))
    plt.close()
    return occupancy