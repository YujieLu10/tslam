import matplotlib.pyplot as plt
import numpy as np
import math
import os

np.set_printoptions(threshold=np.inf)

# prepare some coordinates
# [3, 4, 13, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 154, 155, 156, 157, 158]
# 2 2 2
# /home/yourpathname/prox/tslam/data/local/agent/20210506/resetnormal/gtsample/voxelFalse_rwFalse_obj4_orienup_[-1.57, 0, 0]_[0, -0.14, 0.22]_[-1.57, 0, 0]_[0, -0.7, 0.17]/cf0_knn0_vr1_lstd0.5_knnk5_vconf['3d', 0, 0.01, False]_sensorFalse/agent/run_0/2dnewpointcloud/obj4_step_299.npz
# gtdata = np.load("/home/yourpathname/prox/tslam/data/local/agent/gt_pcloud/groundtruth_obj4.npz")['pcd']

# uniform_gt_data = np.load("/home/yourpathname/prox/tslam/test_o3d.npz")['pcd']
# uniform_gt_data = np.load("/home/yourpathname/prox/tslam/uniform_glass_o3d.npz")['pcd']
uniform_gt_data = np.load("/home/yourpathname/prox/tslam/uniform_donut_o3d.npz")['pcd']
# print(data['pcd'])
# obj4:0.0008 obj1:0.015 obj2:0.01
data_scale = uniform_gt_data * 0.01

data_rotate = data_scale.copy()
# x = data_rotate[:, 0].copy()
# y = data_rotate[:, 1].copy()
# z = data_rotate[:, 2].copy()
# data_rotate[:, 0] = x
# data_rotate[:, 1] = z
# data_rotate[:, 2] = -y

data_trans = data_rotate.copy()
data_trans[:, 0] += 0
data_trans[:, 1] -= 0.24
data_trans[:, 2] += 0.23

uniform_gt_data = data_trans.copy()

for root, dirs, files in os.walk("/home/yourpathname/prox/tslam/data/local/train_adroit/20210516/resetnormal/gtsample/"):
    if "pointcloud_573.npz" in files and "obj2" in root:
        print(root)
        data = np.load(os.path.join(root, "pointcloud_597.npz"))['pcd']
# for step in [49]:#, 99, 149, 249, 299]:
# data = np.load("/home/yourpathname/prox/tslam/voxel/2dnewpointcloud/obj4_orien__step_{}.npz".format(step))['pcd']


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
    if idx_z > 6:
        continue
    name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
    if name not in map_list:
        map_list.append(name)
    cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
    # combine the objects into a single boolean array
    voxels += cube

# draw gt
gt_map_list = []
for idx,val in enumerate(uniform_gt_data):
    idx_x = math.floor((val[0] + 0.125) / resolution)
    idx_y = math.floor((val[1] + 0.25) / resolution)
    idx_z = math.floor((val[2] - 0.16) / resolution)
    if idx_z > 6:
        continue
    name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
    if name not in gt_map_list:
        gt_map_list.append(name)
    cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
    # combine the objects into a single boolean array
    gt_voxels += cube
# gt_obj4:668
print(len(map_list) / len(gt_map_list))
# print(len(map_list) / sep_x / sep_y / sep_z )

obj_name = "donut"
# set the colors of each object
vis_voxel = gt_voxels | voxels
colors = np.empty(vis_voxel.shape, dtype=object)
colors[gt_voxels] = 'white'
colors[voxels] = 'cyan'
# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.set_zlim(1,20)
ax.voxels(vis_voxel, facecolors=colors, edgecolor='g', alpha=.4, linewidth=.05)
# plt.savefig('uniform_gtbox_{}.png'.format(step))
plt.savefig('{}-overlap.png'.format(obj_name))
plt.close()

ax = plt.figure().add_subplot(projection='3d')
ax.set_zlim(1,20)
ax.voxels(gt_voxels, facecolors=colors, edgecolor='g', alpha=.4, linewidth=.05)
# plt.savefig('uniform_gtbox_{}.png'.format(step))
plt.savefig('{}-gt.png'.format(obj_name))
plt.close()

ax = plt.figure().add_subplot(projection='3d')
ax.set_zlim(1,20)
ax.voxels(voxels, facecolors=colors, edgecolor='g', alpha=.4, linewidth=.05)
# plt.savefig('uniform_gtbox_{}.png'.format(step))
plt.savefig('{}-exp.png'.format(obj_name))
plt.close()