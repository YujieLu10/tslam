import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mj_envs.utils.quatmath import quat2euler, euler2quat
from mujoco_py import MjViewer
import os
import random
import matplotlib.pyplot as plt
import pyvista as pv
import torch
import math
import heapq

from xml.etree import ElementTree
from xml.dom import minidom

# from chamfer_distance import ChamferDistance
# chamfer_dist = ChamferDistance()

ADD_BONUS_REWARDS = True

class AdroitEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, 
            obj_orientation= [0, 0, 0], # object orientation
            obj_relative_position= [0, 0.5, 0], # object position related to hand (z-value will be flipped when arm faced down)
            goal_threshold= int(8e3), # how many points touched to achieve the goal
            new_point_threshold= 0.01, # minimum distance new point to all previous points
            forearm_orientation_name= "up", # ("up", "down")
            chamfer_r_factor= 0,
            mesh_p_factor= 0,
            mesh_reconstruct_alpha= 0.01,
            palm_r_factor= 0,
            untouch_p_factor= 0,
            newpoints_r_factor= 0,
            npoint_r_factor= 0,
            ntouch_r_factor= 0,
            random_r_factor= 0,
            knn_r_factor= 0,
            new_voxel_r_factor= 0,
            ground_truth_type= "nope",
            use_voxel= False,
            forearm_orientation= [0, 0, 0], # forearm orientation
            forearm_relative_position= [0, 0.5, 0], # forearm position related to hand (z-value will be flipped when arm faced down)
            reset_mode= "normal",
            knn_k= 1, # k setting
            voxel_conf= ['2d', 0.005], # 2d/3d, 2d_sep
            obj_scale= 0.01,
            obj_name= "airplane",
            generic= False,
            base_rotation= False,
            obs_type= [False, False],
        ):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # xml add object node
        self.generic = generic
        if base_rotation:
            model_prename = "DAPG_rotate_touch"
        else:
            model_prename = "DAPG_touch"
        if not generic and not os.path.isfile(os.path.join(curr_dir, "combine/{}_{}.xml".format(model_prename, obj_name))):
            root = ElementTree.parse(os.path.join(curr_dir, "assets/{}object.xml").format(model_prename)).getroot()
            root.find('include').attrib["file"] = "objects/{}.xml".format(obj_name)
            tree = ElementTree.ElementTree(root)
            tree.write(os.path.join(curr_dir, "combine/{}_{}.xml".format(model_prename, obj_name)), encoding="utf-8", xml_declaration=False)
        
        self.model_path_name = (curr_dir+'/combine/{}_{}.xml'.format(model_prename, obj_name)) if not generic else (curr_dir+'/combine/{}_{}.xml'.format(model_prename, "generic"))

        self.sim = mujoco_env.get_sim(model_path=self.model_path_name)

        self.obj_name = obj_name
        self.obj_scale = obj_scale
        self.obj_current_gt = None
        self.obj_orientation = obj_orientation
        self.obj_relative_position = obj_relative_position
        self.reset_mode = reset_mode
        self.knn_k = knn_k
        self.forearm_orientation = forearm_orientation
        self.forearm_relative_position = forearm_relative_position
        self.prev_a = None
        # reward comfigurations
        self.goal_threshold = goal_threshold
        self.new_point_threshold = new_point_threshold
        self.chamfer_r_factor = chamfer_r_factor
        self.mesh_p_factor = mesh_p_factor
        self.mesh_reconstruct_alpha = mesh_reconstruct_alpha
        self.palm_r_factor = palm_r_factor
        self.untouch_p_factor = untouch_p_factor
        self.newpoints_r_factor = newpoints_r_factor
        self.npoint_r_factor = npoint_r_factor
        self.ntouch_r_factor = ntouch_r_factor
        self.random_r_factor = random_r_factor
        self.knn_r_factor = knn_r_factor
        self.ground_truth_type = ground_truth_type
        self.use_voxel = use_voxel
        self.voxel_type = voxel_conf[0]
        self.twod_sep = voxel_conf[1]
        self.new_voxel_r_factor = new_voxel_r_factor
        self.obs_type = obs_type

        self.voxel_num = int(math.pow(self.twod_sep, 3))
        self.voxel_array = [0] * self.voxel_num
        self.gt_map_list = []

        self.touch_obj_bid = 0
        self.forearm_obj_bid = 0
        self.S_grasp_sid = 0
        self.ffknuckle_obj_bid = 0
        self.mfknuckle_obj_bid = 0
        self.rfknuckle_obj_bid = 0
        self.lfmetacarpal_obj_bid = 0
        self.thbase_obj_bid = 0
        self.sensor_rid_list = [0] * 27
        
        self.count_step = 0
        self.obj_iter = 0
        self.previous_contact_points = []
        self.new_current_pos_list = []
        # scales
        self.act_mid = 0
        self.act_rng = 0

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, self.model_path_name, 5)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.prev_a = None
        # scales
        self.act_mid = np.mean(self.sim.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.sim.model.actuator_ctrlrange[:,1]-self.sim.model.actuator_ctrlrange[:,0])

        utils.EzPickle.__init__(self)
        self.forearm_obj_bid = self.sim.model.body_name2id("forearm")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.ffknuckle_obj_bid = self.sim.model.body_name2id('ffknuckle')
        self.mfknuckle_obj_bid = self.sim.model.body_name2id('mfknuckle')
        self.rfknuckle_obj_bid = self.sim.model.body_name2id('rfknuckle')
        self.lfmetacarpal_obj_bid = self.sim.model.body_name2id('lfmetacarpal')
        self.thbase_obj_bid = self.sim.model.body_name2id('thbase')
        self.touch_obj_bid = self.sim.model.body_name2id('{}object'.format(self.obj_name if self.obj_name != "generic" else "airplane"))
        # self.grasp_sensor_rid, self.Tch_ffmetacarpal_sensor_rid, self.Tch_mfmetacarpal_sensor_rid, self.Tch_rfmetacarpal_sensor_rid, self.Tch_thmetacarpal_sensor_rid, self.Tch_palm_sensor_rid, self.Tch_ffproximal_sensor_rid, self.Tch_ffmiddle_sensor_rid, self.S_fftip_sensor_rid, self.Tch_fftip_sensor_rid, self.Tch_mfproximal_sensor_rid, self.Tch_mfmiddle_sensor_rid, self.S_mftip_sensor_rid, self.Tch_mftip_sensor_rid, self.Tch_rfproximal_sensor_rid, self.Tch_rfmiddle_sensor_rid, self.S_rftip_sensor_rid, self.Tch_rftip_sensor_rid, self.Tch_lfmetacarpal_sensor_rid, self.Tch_lfproximal_sensor_rid, self.Tch_lfmiddle_sensor_rid, self.S_lftip_sensor_rid, self.Tch_lftip_sensor_rid, self.Tch_thproximal_sensor_rid, self.Tch_thmiddle_sensor_rid, self.S_thtip_sensor_rid, self.Tch_thtip_sensor_rid 
        
        self.sensor_rid_list = [self.sim.model.sensor_name2id('S_grasp_sensor'), self.sim.model.sensor_name2id('Tch_ffmetacarpal_sensor'), self.sim.model.sensor_name2id('Tch_mfmetacarpal_sensor'), self.sim.model.sensor_name2id('Tch_rfmetacarpal_sensor'), self.sim.model.sensor_name2id('Tch_thmetacarpal_sensor'), self.sim.model.sensor_name2id('Tch_palm_sensor'), self.sim.model.sensor_name2id('Tch_ffproximal_sensor'), self.sim.model.sensor_name2id('Tch_ffmiddle_sensor'), self.sim.model.sensor_name2id('S_fftip_sensor'), self.sim.model.sensor_name2id('Tch_fftip_sensor'), self.sim.model.sensor_name2id('Tch_mfproximal_sensor'), self.sim.model.sensor_name2id('Tch_mfmiddle_sensor'), self.sim.model.sensor_name2id('S_mftip_sensor'), self.sim.model.sensor_name2id('Tch_mftip_sensor'), self.sim.model.sensor_name2id('Tch_rfproximal_sensor'), self.sim.model.sensor_name2id('Tch_rfmiddle_sensor'), self.sim.model.sensor_name2id('S_rftip_sensor'), self.sim.model.sensor_name2id('Tch_rftip_sensor'), self.sim.model.sensor_name2id('Tch_lfmetacarpal_sensor'), self.sim.model.sensor_name2id('Tch_lfproximal_sensor'), self.sim.model.sensor_name2id('Tch_lfmiddle_sensor'), self.sim.model.sensor_name2id('S_lftip_sensor'), self.sim.model.sensor_name2id('Tch_lftip_sensor'), self.sim.model.sensor_name2id('Tch_thproximal_sensor'), self.sim.model.sensor_name2id('Tch_thmiddle_sensor'), self.sim.model.sensor_name2id('S_thtip_sensor'), self.sim.model.sensor_name2id('Tch_thtip_sensor')]


    def is_in_voxel_bound(self, posx, posy):
        is_in_bound = False
        is_in_bound = posx > -0.125 and posx < 0.125 and posy > -0.25 and posy < -0.025
        return is_in_bound
    
    def get_voxel_len(self):
        if self.voxel_type == '2d':
            sep_x, sep_y = 0, 0
            sep_x = 0.25 / self.twod_sep
            sep_y = 0.225 / self.twod_sep
            return math.ceil(sep_x) * math.ceil(sep_y)
        else:
            sep_x, sep_y, sep_z = 0, 0, 0
            sep_x = math.ceil(0.25 / self.twod_sep)
            sep_y = math.ceil(0.225 / self.twod_sep)
            sep_z = math.ceil(0.1 / self.twod_sep)
            return sep_x * sep_y * sep_z

    def get_2d_voxel_idx(self, posx, posy):
        # for simple objectobj5(index in obj_list:4) only
        # here we use 14 pose each 600 sampled points as ground truth
        # center point xy(0, -0.14)
        # corner points (-0.125,-0.25) (-0.125,-0.025) (0.125, -0.25) (0.125, -0.025)
        # test left (0,-0.25) (0,-0.025) (0.125, -0.25) (0.125, -0.025)
        sep_x, sep_y, idx_x, idx_y = 0, 0, 0, 0
        sep_x = 0.25 / self.twod_sep
        sep_y = 0.225 / self.twod_sep
        idx_x = math.floor((posx + 0.125) / self.twod_sep)
        idx_y = math.floor((posy + 0.25) / self.twod_sep)
        voxel_idx = idx_y * math.ceil(sep_x) + idx_x + 1
        return voxel_idx

    def generate_uniform_gt_voxel(self):
        uniform_gt_data = np.load("/home/jianrenw/prox/tslam/assets/uniform_gt/uniform_{}_o3d.npz".format(self.obj_name))['pcd']
        data_scale = uniform_gt_data * self.obj_scale
        data_rotate = data_scale.copy()
        x = data_rotate[:, 0].copy()
        y = data_rotate[:, 1].copy()
        z = data_rotate[:, 2].copy()
        x_theta = self.obj_orientation[0]
        data_rotate[:, 0] = x
        data_rotate[:, 1] = y*math.cos(x_theta) - z*math.sin(x_theta)
        data_rotate[:, 2] = y*math.sin(x_theta) + z*math.cos(x_theta)
        data_trans = data_rotate.copy()
        data_trans[:, 0] += self.obj_relative_position[0]
        data_trans[:, 1] += self.obj_relative_position[1]
        data_trans[:, 2] += self.obj_relative_position[2]

        uniform_gt_data = data_trans.copy()

        resolution_x, resolution_y, resolution_z = math.ceil(0.25 / self.twod_sep), math.ceil(0.225 / self.twod_sep), math.ceil(0.1 / self.twod_sep)
        x, y, z = np.indices((self.twod_sep, self.twod_sep, self.twod_sep))

        gtcube = (x<0) & (y <1) & (z<1)
        gt_voxels = gtcube
        gt_map_list = []
        for idx,val in enumerate(uniform_gt_data):
            idx_x = math.floor((val[0] + 0.125) / resolution_x)
            idx_y = math.floor((val[1] + 0.25) / resolution_y)
            idx_z = math.floor((val[2] - 0.16) / resolution_z)
            name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
            if name not in gt_map_list:
                gt_map_list.append(name)
            cube = (x < idx_x + 1) & (y < idx_y + 1) & (z < idx_z + 1) & (x >= idx_x) & (y >= idx_y) & (z >= idx_z)
            # combine the objects into a single boolean array
            gt_voxels += cube
        self.gt_map_list = gt_map_list.copy()

    def get_voxel_idx(self, posx, posy, posz):
        resolution_x, resolution_y, resolution_z = math.ceil(0.25 / self.twod_sep), math.ceil(0.225 / self.twod_sep), math.ceil(0.1 / self.twod_sep)
        idx_x = math.floor((posx + 0.125) / resolution_x)
        idx_y = math.floor((posy + 0.25) / resolution_y)
        idx_z = math.floor((posz - 0.16) / resolution_z)
        name = str(idx_x) + '_' + str(idx_y) + '_' + str(idx_z)
        return self.gt_map_list.index(name) if name in self.gt_map_list else -1

    def get_basic_reward(self, posA, posB):
        dist = np.linalg.norm(posA-posB)
        return -dist

    def get_notouch_penalty(self, touched):
        return 0

    def get_newpoints_reward(self, min_pos_dist):
        if min_pos_dist > 0.1:
            return 30
        else:
            return 10

    def loss_transform(self, loss):
        # chamfer_dist loss normalization
        if loss >= 1e-6:
            loss = 1
        elif loss <= 1e-15:
            loss = 0
        else:
            loss = (loss - 1e-15) / (1e-6 - 1e-15)
        return loss

    def get_chamfer_reward(self, chamfer_distance_loss):
        chamfer_reward = 0
        if "nope" not in self.ground_truth_type:
            chamfer_reward += (0.1-chamfer_distance_loss) * 10
        else:
            chamfer_reward += self.loss_transform(chamfer_distance_loss) * 10
        return chamfer_reward

    # def get_chamfer_distance_loss(self, is_touched, previous_pos_list, current_pos_list):
    #     chamfer_distance_loss = 0.0
    #     if "nope" not in self.ground_truth_type:
    #         if is_touched and self.previous_contact_points != [] and previous_pos_list != []:
    #             gt_dist1, gt_dist2 = chamfer_dist(torch.FloatTensor([self.obj_current_gt]), torch.FloatTensor([current_pos_list]))
    #             chamfer_distance_loss = (torch.mean(gt_dist1)) + (torch.mean(gt_dist2))
    #     else:
    #         if is_touched and self.previous_contact_points != [] and previous_pos_list != []:
    #             dist1, dist2 = chamfer_dist(torch.FloatTensor([previous_pos_list]), torch.FloatTensor([current_pos_list]))
    #             chamfer_distance_loss = (torch.mean(dist1)) + (torch.mean(dist2))
    #     return chamfer_distance_loss

    def get_knn_reward(self):
        return 0

    def get_penalty(self):
        return 0

    def step(self, a):
        # uniform_samplegt = np.load('/home/jianrenw/prox/tslam/test_o3d.npz')['pcd']
        # apply action and step
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mid + a*self.act_rng
        self.do_simulation(a, self.frame_skip)
        self.count_step += 1
        obj_init_xpos  = self.data.body_xpos[self.touch_obj_bid].ravel()
        palm_xpos = self.data.site_xpos[self.S_grasp_sid].ravel()
        ffknuckle_xpos = self.data.site_xpos[self.ffknuckle_obj_bid].ravel()
        mfknuckle_xpos = self.data.site_xpos[self.mfknuckle_obj_bid].ravel()
        rfknuckle_xpos = self.data.site_xpos[self.rfknuckle_obj_bid].ravel()
        lfmetacarpal_xpos = self.data.site_xpos[self.lfmetacarpal_obj_bid].ravel()
        thbase_xpos = self.data.site_xpos[self.thbase_obj_bid].ravel()
        reward = 0.0
        untouched_p = 0.0
        # variant npoint and ntouch
        npoint_r = 0
        ntouch_r = 0
        # palm close to object reward
        palm_r = self.get_basic_reward(obj_init_xpos, palm_xpos) if self.palm_r_factor else 0
        palm_r += self.get_basic_reward(obj_init_xpos, ffknuckle_xpos) if self.palm_r_factor else 0
        palm_r += self.get_basic_reward(obj_init_xpos, mfknuckle_xpos) if self.palm_r_factor else 0
        palm_r += self.get_basic_reward(obj_init_xpos, rfknuckle_xpos) if self.palm_r_factor else 0
        palm_r += self.get_basic_reward(obj_init_xpos, lfmetacarpal_xpos) if self.palm_r_factor else 0
        palm_r += self.get_basic_reward(obj_init_xpos, thbase_xpos) if self.palm_r_factor else 0
        
        # pos of current obj's contacts
        current_pos_list = []
        is_touched = False
        new_points_cnt = 0
        # contacts of current obj
        for contact in self.data.contact:
            if self.sim.model.geom_id2name(contact.geom1) is None or self.sim.model.geom_id2name(contact.geom2) is None:
                continue
            if "contact" in self.sim.model.geom_id2name(contact.geom1) or "contact" in self.sim.model.geom_id2name(contact.geom2):
                current_pos_list.append(contact.pos.tolist())
                is_touched = True
                ntouch_r += 1

        # if is_touched:
        #     ntouch_r += 1
        
        # untouch penalty
        untouched_p -= 0.01 if self.untouch_p_factor and not is_touched else 0

        # dedup item
        current_pos_list = [item for item in current_pos_list if current_pos_list.count(item) == 1]

        new_pos_list = []
        min_pos_dist = None
        knn_r = 0
        newpoints_r = 0
        new_voxel_r = 0
        chamfer_r = 0
        chamfer_loss = 0

        previous_pos_list = self.previous_contact_points.copy()
        next_pos_list = self.previous_contact_points.copy()
        
        for pos in current_pos_list:
            if pos not in next_pos_list:
                next_pos_list.append(pos)  
            if self.npoint_r_factor:
                npoint_r += 1
            # new contact points
            if pos not in self.previous_contact_points and self.knn_r_factor:
                min_pos_dist = 1
                pos_dist_list = []
                for previous_pos in self.previous_contact_points:
                    pos_dist = np.linalg.norm(np.array(pos) - np.array(previous_pos))
                    min_pos_dist = pos_dist if min_pos_dist is None else min(min_pos_dist, pos_dist)
                    pos_dist_list.append(pos_dist)
                # new contact points that are not close to already touched points
                if min_pos_dist and min_pos_dist > 0.01: 
                    new_points_cnt += 1  
                    newpoints_r += self.get_newpoints_reward(min_pos_dist)
                    new_pos_list.append(pos)
                knn_r += sum(heapq.nsmallest(min(len(pos_dist_list), self.knn_k),pos_dist_list))
        
        self.new_current_pos_list = new_pos_list
        # similar points penalty
        # penalty_sim = similar_points_cnt * 5
        # penalty_sim = self.get_penalty()
        if self.chamfer_r_factor:
            chamfer_loss = 0 #self.get_chamfer_distance_loss(is_touched, previous_pos_list, next_pos_list)
            chamfer_r = 0 #1 / (chamfer_loss) if chamfer_loss > 0 else 0 # self.get_chamfer_reward(chamfer_loss)
            # chamfer_r -= 300

        self.previous_contact_points = next_pos_list.copy()
        # start computing reward
        # for simplicity goal_achieved depends on the nubmer of touched points
        goal_achieved = (len(self.previous_contact_points) > self.goal_threshold)
        if "nope" not in self.ground_truth_type and is_touched and self.previous_contact_points != [] and previous_pos_list != [] and chamfer_loss < 0.06:
            goal_achieved = True
        mesh_p = 0
        # voxel obs and new voxel reward
        if self.voxel_array is not None and len(self.voxel_array) == 0:
            self.voxel_array = [0] * self.voxel_num
        if len(self.previous_contact_points) > 0:
            for point in self.previous_contact_points:
                # if self.is_in_voxel_bound(point[0], point[1]):
                idx = self.get_2d_voxel_idx(point[0], point[1]) if self.voxel_type == '2d' else self.get_voxel_idx(point[0], point[1], point[2])
                if idx > 0 and self.voxel_array[min(idx, self.voxel_num-1)] == 0: # new voxel touched
                    new_voxel_r += 1
                    self.voxel_array[min(idx, self.voxel_num-1)] = 1
        denominator = len(np.array(self.voxel_array))
        voxel_occupancy = (len(np.where(np.array(self.voxel_array)>0)) / denominator) if denominator > 0 else 0
        reward += self.palm_r_factor * palm_r
        reward += self.untouch_p_factor * untouched_p
        reward += self.chamfer_r_factor * chamfer_r
        reward += self.newpoints_r_factor * newpoints_r
        reward += self.mesh_p_factor * mesh_p
        reward += self.knn_r_factor * knn_r
        reward += self.new_voxel_r_factor * new_voxel_r
        reward += self.npoint_r_factor * npoint_r
        reward += self.ntouch_r_factor * ntouch_r
        if self.random_r_factor > 0:
            reward = 0
        done = False
        info = dict(
            pointcloud= np.array(self.previous_contact_points), #np.array(uniform_samplegt),
            goal_achieved= goal_achieved,
            untouched_p= untouched_p,
            palm_r = palm_r,
            chamfer_r= chamfer_r,
            newpoints_r= newpoints_r,
            new_voxel_r= new_voxel_r,
            npoint_r= npoint_r,
            ntouch_r= ntouch_r,
            mesh_p= mesh_p,
            knn_r= knn_r,
            total_reward_r= reward,
            voxel_array= np.array(self.voxel_array),
            chamfer_loss_p= chamfer_loss,
            resolution= self.twod_sep,
            occupancy= voxel_occupancy,
        )
        return self.get_obs(), reward, done, info

    def get_obs(self):    
        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        palm_xpos = self.data.site_xpos[self.S_grasp_sid].ravel()
        # 6 extreme pos
        all_points = np.array(self.previous_contact_points)
        if len(all_points) > 0:
            extreme_points = np.concatenate((all_points[all_points[:, 0].argmin()], all_points[all_points[:, 1].argmin()], all_points[all_points[:, 2].argmin()], all_points[all_points[:, 0].argmax()], all_points[all_points[:, 1].argmax()], all_points[all_points[:, 2].argmax()]))
        else:
            extreme_points = (0.5 * np.random.randn(1, 18))
        
        new_touch_pos = random.sample(self.new_current_pos_list, min(6, len(self.new_current_pos_list))) if len(self.new_current_pos_list) > 0 else []
        old_touch_pos = random.sample(self.previous_contact_points, min(6 - len(new_touch_pos), len(self.previous_contact_points))) if 6 > len(self.new_current_pos_list) and len(self.previous_contact_points) > 0 else []
        random_touch_pos = (0.5 * np.random.randn(1, 3 *(7 - len(new_touch_pos) - len(old_touch_pos)))) if 7 - len(new_touch_pos) - len(old_touch_pos) > 0 else []
        touch_pos = np.array(palm_xpos)

        if len(new_touch_pos) > 0:
            touch_pos = np.append(touch_pos, new_touch_pos)
        if len(old_touch_pos) > 0:
            touch_pos = np.append(touch_pos, old_touch_pos)
        if len(random_touch_pos) > 0:
            touch_pos = np.append(touch_pos, random_touch_pos)

        impact_list = None
        for rid in self.sensor_rid_list:
            if impact_list is None:
                impact_list = np.clip(self.sim.data.sensordata[rid], [-1.0], [1.0])
            else:
                impact_list = np.append(impact_list, np.clip(self.sim.data.sensordata[rid], [-1.0], [1.0]))

        # use 3d fixed voxel grid
        voxel_obs = np.array(self.voxel_array)
        # use sensor obs
        if self.obs_type[1]:
            final_observation = np.concatenate([qp, qv, np.array(impact_list), voxel_obs]) if self.use_voxel else np.concatenate([qp, qv, np.array(impact_list), np.concatenate((np.array(touch_pos).flatten(), np.array(extreme_points).flatten()))])
        else:
            final_observation = np.concatenate([qp, qv, voxel_obs]) if self.use_voxel else np.concatenate([qp, qv, np.concatenate((np.array(touch_pos).flatten(), np.array(extreme_points).flatten()))])
        return final_observation

    def reset_model(self, num_traj_idx):
        # clear each round
        self.count_step = 0
        self.previous_contact_points = []
        self.new_current_pos_list = []
        # generate voxel
        name_map = ['duck', 'watch', 'doorknob', 'headphones', 'bowl', 'cubesmall', 'spheremedium', 'train', 'piggybank', 'cubemedium', 'cubelarge', 'elephant', 'flute', 'wristwatch', 'pyramidmedium', 'gamecontroller', 'toothbrush', 'pyramidsmall', 'body', 'cylinderlarge', 'cylindermedium', 'cylindersmall', 'fryingpan', 'stanfordbunny', 'scissors', 'pyramidlarge', 'stapler', 'flashlight', 'mug', 'hand', 'stamp', 'rubberduck', 'binoculars', 'apple', 'mouse', 'eyeglasses', 'airplane', 'coffeemug', 'cup', 'toothpaste', 'torusmedium', 'cubemiddle', 'phone', 'torussmall', 'spheresmall', 'knife', 'banana', 'teapot', 'hammer', 'alarmclock', 'waterbottle', 'camera', 'table', 'wineglass', 'lightbulb', 'spherelarge', 'toruslarge', 'glass', 'heart', 'donut']
        num_traj_idx = min(num_traj_idx, len(name_map) - 1)
        self.obj_name = name_map[num_traj_idx]

        self.touch_obj_bid = self.sim.model.body_name2id('{}object'.format(self.obj_name))
        self.generate_uniform_gt_voxel()

        if self.ground_truth_type == "sample":
            self.obj_current_gt = np.load(os.path.join("/home/jianrenw/prox/tslam/assets", "uniform_gt", "uniform_{}_o3d.npz".format(self.obj_name)))['pcd']

        current_reset = False
        if self.reset_mode == "random" and random.randint(0,9) > 6:
            current_reset = True
        elif self.reset_mode == "normal":
            current_reset = True

        # set target object pose
        for name in name_map:
            if name != self.obj_name:
                other_obj_bid = self.sim.model.body_name2id('{}object'.format(name))
                self.model.body_pos[other_obj_bid] = [0, 0, -10]

        self.model.body_pos[self.touch_obj_bid] = self.obj_relative_position
        # self.model.body_quat[self.touch_obj_bid] = euler2quat(self.obj_orientation)

        # set arm pose
        self.model.body_quat[self.forearm_obj_bid] = euler2quat(self.forearm_orientation)
        self.model.body_pos[self.forearm_obj_bid] = self.forearm_relative_position

        if not current_reset:
            self.sim.forward()
            return self.get_obs()

        # optional reset
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        forearm_orien = self.model.body_quat[self.forearm_obj_bid].ravel().copy()
        forearm_pos = self.model.body_pos[self.forearm_obj_bid].ravel().copy()
        obj_init_pos = self.model.body_pos[self.touch_obj_bid].ravel().copy()
        ffknuckle_pos = self.model.body_pos[self.ffknuckle_obj_bid].ravel().copy()
        mfknuckle_pos = self.model.body_pos[self.mfknuckle_obj_bid].ravel().copy()
        rfknuckle_pos = self.model.body_pos[self.rfknuckle_obj_bid].ravel().copy()
        lfmetacarpal_pos = self.model.body_pos[self.lfmetacarpal_obj_bid].ravel().copy()
        thbase_pos = self.model.body_pos[self.thbase_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, forearm_orien=forearm_orien, forearm_pos=forearm_pos, obj_init_pos=obj_init_pos, ffknuckle_pos=ffknuckle_pos, mfknuckle_pos=mfknuckle_pos,rfknuckle_pos=rfknuckle_pos,lfmetacarpal_pos=lfmetacarpal_pos,thbase_pos=thbase_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        forearm_orien = state_dict['forearm_orien']
        forearm_pos = state_dict['forearm_pos']
        obj_init_pos = state_dict['obj_init_pos']
        ffknuckle_pos = state_dict['ffknuckle_pos']
        mfknuckle_pos = state_dict['mfknuckle_pos']
        rfknuckle_pos = state_dict['rfknuckle_pos']
        lfmetacarpal_pos = state_dict['lfmetacarpal_pos']
        thbase_pos = state_dict['thbase_pos']
        self.set_state(qp, qv)
        self.model.body_quat[self.forearm_obj_bid] = forearm_orien
        self.model.body_pos[self.forearm_obj_bid] = forearm_pos
        self.model.body_pos[self.touch_obj_bid] = obj_init_pos
        self.model.body_pos[self.ffknuckle_obj_bid] = ffknuckle_pos
        self.model.body_pos[self.mfknuckle_obj_bid] = mfknuckle_pos
        self.model.body_pos[self.rfknuckle_obj_bid] = rfknuckle_pos
        self.model.body_pos[self.lfmetacarpal_obj_bid] = lfmetacarpal_pos
        self.model.body_pos[self.thbase_obj_bid] = thbase_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -45
        self.sim.forward()
        self.viewer.cam.distance = 0.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if pen within 15 degrees of target for 20 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 20:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
