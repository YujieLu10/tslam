import numpy as np
import matplotlib.pyplot as plt
import os

# for obj_idx in range(4,5):
#     if obj_idx == 7:
#         continue
#     for step_idx in range(99,200,100):
#         data_down = np.load("newpointcloud/obj{}_orien_down_step_{}.npz".format(obj_idx, step_idx))
#         data_up = np.load("newpointcloud/obj{}_orien_up_step_{}.npz".format(obj_idx, step_idx))
#         data_gt = np.load("gt_pcloud/obj{}.npz".format(obj_idx))
        
#         pc_frame = np.append(data_down['pcd'], data_up['pcd'], axis=0)
#         gt_pc_frame = data_gt['pcd']
#         ax = plt.axes(projection='3d')
#         ax.scatter(gt_pc_frame[:, 0], gt_pc_frame[:, 1], gt_pc_frame[:, 2], c='green', cmap='viridis', linewidth=0.5)
#         ax.scatter(pc_frame[:, 0], pc_frame[:, 1], pc_frame[:, 2], c='blue', cmap='viridis', linewidth=0.5)
        
#         # plt.savefig("visresult/obj{}_step{}.png".format(obj_idx, step_idx))
#         plt.savefig("visgt/obj{}_step{}.png".format(obj_idx, step_idx))
#         plt.close()

# obj_idx = 4
# pc_frame = None
# for path, dirlist, filelist in os.walk("newpointcloud"):
#     for file in filelist:
#         if "_9999.npz" in file:
#             data = np.load(os.path.join(path, file))
#             pc_frame = data['pcd'][:600] if pc_frame is None else np.append(pc_frame, data['pcd'][:600], axis=0)

# np.savez_compressed(os.path.join("/home/jianrenw/prox/tslam/data/local/agent", "gt_pcloud", "groundtruth_obj4.npz"), pcd=pc_frame)

# print(pc_frame.shape)
# ax = plt.axes(projection='3d')
# ax.scatter(pc_frame[:, 0], pc_frame[:, 1], pc_frame[:, 2], c='green', cmap='viridis', s=1)
# plt.savefig("gt_pcloud/new_sample_gt_obj{}.png".format(obj_idx))
# plt.close()

# data = np.load(os.path.join("/home/jianrenw/prox/tslam/data/local/agent", "gt_pcloud", "groundtruth_obj4.npz"))['pcd']
# data = np.load("/home/jianrenw/prox/tslam/test_o3d.npz")['pcd']
# data = np.load("/home/jianrenw/prox/tslam/uniform_donut_o3d.npz")['pcd']
data = np.load("/home/jianrenw/prox/tslam/uniform_glass_o3d.npz")['pcd']
# print(data['pcd'])
data_scale = data * 0.01

data_rotate = data_scale.copy()
x = data_rotate[:, 0].copy()
y = data_rotate[:, 1].copy()
z = data_rotate[:, 2].copy()
data_rotate[:, 0] = x
data_rotate[:, 1] = -z
data_rotate[:, 2] = -y

data_trans = data_rotate.copy()
data_trans[:, 0] += 0
data_trans[:, 1] -= 0.14
data_trans[:, 2] += 0.23

data = data_trans
ax = plt.axes()
ax.scatter(data[:, 0], data[:, 1], c='green', cmap='viridis', s=1)
plt.savefig("gt_pcloud/uniform_glass_find2dvoxel.png")
plt.close()