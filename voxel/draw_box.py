import matplotlib.pyplot as plt
import numpy as np
import math

np.set_printoptions(threshold=np.inf)

# prepare some coordinates
# [3, 4, 13, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 154, 155, 156, 157, 158]
# 2 2 2
# /home/jianrenw/prox/tslam/data/local/agent/20210506/resetnormal/gtsample/voxelFalse_rwFalse_obj4_orienup_[-1.57, 0, 0]_[0, -0.14, 0.22]_[-1.57, 0, 0]_[0, -0.7, 0.17]/cf0_knn0_vr1_lstd0.5_knnk5_vconf['3d', 0, 0.01, False]_sensorFalse/agent/run_0/2dnewpointcloud/obj4_step_299.npz
gtdata = np.load("/home/jianrenw/prox/tslam/data/local/agent/gt_pcloud/groundtruth_obj4.npz")['pcd']

uniform_gt_data = np.load("/home/jianrenw/prox/tslam/test_o3d.npz")['pcd']
# print(data['pcd'])
data_scale = uniform_gt_data * 0.0008

data_rotate = data_scale.copy()
x = data_rotate[:, 0].copy()
y = data_rotate[:, 1].copy()
z = data_rotate[:, 2].copy()
data_rotate[:, 0] = x
data_rotate[:, 1] = z
data_rotate[:, 2] = -y

data_trans = data_rotate.copy()
data_trans[:, 0] += 0
data_trans[:, 1] -= 0.14
data_trans[:, 2] += 0.22

uniform_gt_data = data_trans

for step in [49]:#, 99, 149, 249, 299]:
    data = np.load("/home/jianrenw/prox/tslam/voxel/2dnewpointcloud/obj4_orien__step_{}.npz".format(step))['pcd']
    resolution = 0.01
    sep_x = math.ceil(0.25 / resolution)
    sep_y = math.ceil(0.225 / resolution)
    sep_z = math.ceil(0.1 / resolution)
    x, y, z = np.indices((sep_x, sep_y, sep_z))

    cube1 = (x<0) & (y <1) & (z<1)
    gtcube = (x<0) & (y <1) & (z<1)
    voxels = cube1
    gt_voxels = gtcube
    # draw cuboids in the top left and bottom right corners, and a link between them
    map_list = []
    for idx,val in enumerate(data):
        idx_x = math.floor((val[0] + 0.125) / resolution)
        idx_y = math.floor((val[1] + 0.25) / resolution)
        idx_z = math.floor((val[2] - 0.16) / resolution)
        name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
        if name not in map_list:
            map_list.append(name)
        cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
        # combine the objects into a single boolean array
        voxels += cube

    # draw gt
    map_list = []
    for idx,val in enumerate(uniform_gt_data):
        idx_x = math.floor((val[0] + 0.125) / resolution)
        idx_y = math.floor((val[1] + 0.25) / resolution)
        idx_z = math.floor((val[2] - 0.16) / resolution)
        name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
        if name not in map_list:
            map_list.append(name)
        cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
        # combine the objects into a single boolean array
        gt_voxels += cube
    # gt 668
    # print(len(map_list) / 668)
    # print(len(map_list) / sep_x / sep_y / sep_z )

    # set the colors of each object
    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_zlim(1,20)
    vis_voxel = gt_voxels #| voxels
    colors = np.empty(vis_voxel.shape, dtype=object)
    colors[voxels] = 'cyan'
    colors[gt_voxels] = 'white'
    ax.voxels(vis_voxel, facecolors=colors, edgecolor='g', alpha=.4, linewidth=.05)

    plt.savefig('uniform_gtbox_{}.png'.format(step))
    plt.close()