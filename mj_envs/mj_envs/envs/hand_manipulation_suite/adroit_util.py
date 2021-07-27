import math
import numpy as np


def is_in_voxel_bound(posx, posy):
    is_in_bound = False
    is_in_bound = posx > -0.125 and posx < 0.125 and posy > -0.25 and posy < -0.025
    return is_in_bound

def get_voxel_len(voxel_type, twod_sep):
    if voxel_type == '2d':
        sep_x, sep_y = 0, 0
        sep_x = 0.25 / twod_sep
        sep_y = 0.25 / twod_sep
        return math.ceil(sep_x) * math.ceil(sep_y)
    else:
        sep_x, sep_y, sep_z = 0, 0, 0
        sep_x = math.ceil(0.25 / twod_sep)
        sep_y = math.ceil(0.25 / twod_sep)
        sep_z = math.ceil(0.25 / twod_sep)
        return sep_x * sep_y * sep_z

def get_2d_voxel_idx(self, posx, posy, twod_sep):
    # for simple objectobj5(index in obj_list:4) only
    # here we use 14 pose each 600 sampled points as ground truth
    # center point xy(0, -0.14)
    # corner points (-0.125,-0.25) (-0.125,-0.025) (0.125, -0.25) (0.125, -0.025)
    # test left (0,-0.25) (0,-0.025) (0.125, -0.25) (0.125, -0.025)
    sep_x, sep_y, idx_x, idx_y = 0, 0, 0, 0
    sep_x = 0.25 / twod_sep
    sep_y = 0.25 / twod_sep
    idx_x = math.floor((posx + 0.125) / twod_sep)
    idx_y = math.floor((posy + 0.125) / twod_sep)
    voxel_idx = idx_y * math.ceil(sep_x) + idx_x + 1
    return voxel_idx

def generate_uniform_gt_voxel(obj_name, obj_scale, obj_orientation, obj_relative_position, twod_sep):
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

    x = data_rotate[:, 0].copy()
    y = data_rotate[:, 1].copy()
    z = data_rotate[:, 2].copy()
    y_theta = obj_orientation[1]
    data_rotate[:, 0] = x * math.cos(y_theta) + z * math.sin(y_theta)
    data_rotate[:, 1] = y
    data_rotate[:, 2] = z * math.cos(y_theta) - x * math.sin(y_theta)

    x = data_rotate[:, 0].copy()
    y = data_rotate[:, 1].copy()
    z = data_rotate[:, 2].copy()
    z_theta = obj_orientation[2]
    data_rotate[:, 0] = x * math.cos(z_theta) - y * math.sin(z_theta)
    data_rotate[:, 1] = x * math.sin(z_theta) + y * math.cos(z_theta)
    data_rotate[:, 2] = z

    data_trans = data_rotate.copy()
    data_trans[:, 0] += obj_relative_position[0]
    data_trans[:, 1] += obj_relative_position[1]
    data_trans[:, 2] += obj_relative_position[2]

    uniform_gt_data = data_trans.copy()

    resolution_x, resolution_y, resolution_z = 0.3 / twod_sep, 0.3 / twod_sep, 0.3 / twod_sep
    x, y, z = np.indices((twod_sep, twod_sep, twod_sep))

    gtcube = (x<0) & (y <1) & (z<1)
    gt_voxels = gtcube
    gt_map_list = []
    for idx,val in enumerate(uniform_gt_data):
        idx_x = math.floor((val[0] + 0.15) / resolution_x)
        idx_y = math.floor((val[1] + 0.15) / resolution_y)
        idx_z = math.floor((val[2]) / resolution_z)
        name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
        if name not in gt_map_list:
            gt_map_list.append(name)
        cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
        # combine the objects into a single boolean array
        gt_voxels += cube
    return gt_map_list.copy() # self.gt_map_list

def get_voxel_idx(posx, posy, posz, twod_sep, gt_map_list, need_gt):
    resolution_x, resolution_y, resolution_z = 0.3 / twod_sep, 0.3 / twod_sep, 0.3 / twod_sep
    idx_x = math.floor((posx + 0.15) / resolution_x)
    idx_y = math.floor((posy + 0.15) / resolution_y)
    idx_z = math.floor((posz) / resolution_z)
    voxel_idx = idx_z * twod_sep * twod_sep + idx_y * twod_sep + idx_x
    if need_gt:
        name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
        return voxel_idx if name in gt_map_list else -1
    else:
        return voxel_idx

def get_basic_reward(posA, posB):
    dist = np.linalg.norm(posA-posB)
    return -dist

def get_notouch_penalty(touched):
    return 0

def get_newpoints_reward(min_pos_dist):
    if min_pos_dist > 0.1:
        return 30
    else:
        return 10

def loss_transform(loss):
    # chamfer_dist loss normalization
    if loss >= 1e-6:
        loss = 1
    elif loss <= 1e-15:
        loss = 0
    else:
        loss = (loss - 1e-15) / (1e-6 - 1e-15)
    return loss

def get_chamfer_reward(chamfer_distance_loss, ground_truth_type):
    chamfer_reward = 0
    if "nope" not in ground_truth_type:
        chamfer_reward += (0.1-chamfer_distance_loss) * 10
    else:
        chamfer_reward += loss_transform(chamfer_distance_loss) * 10
    return chamfer_reward

def get_chamfer_distance_loss(is_touched, previous_pos_list, current_pos_list, ground_truth_type, previous_contact_points, obj_current_gt):
    chamfer_distance_loss = 0.0
    if "nope" not in ground_truth_type:
        if is_touched and previous_contact_points != [] and previous_pos_list != []:
            gt_dist1, gt_dist2 = chamfer_dist(torch.FloatTensor([obj_current_gt]), torch.FloatTensor([current_pos_list]))
            chamfer_distance_loss = (torch.mean(gt_dist1)) + (torch.mean(gt_dist2))
    else:
        if is_touched and previous_contact_points != [] and previous_pos_list != []:
            dist1, dist2 = chamfer_dist(torch.FloatTensor([previous_pos_list]), torch.FloatTensor([current_pos_list]))
            chamfer_distance_loss = (torch.mean(dist1)) + (torch.mean(dist2))
    return chamfer_distance_loss

def get_knn_reward(self):
    return 0

def get_penalty(self):
    return 0
