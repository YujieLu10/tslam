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
from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

ADD_BONUS_REWARDS = True

class AdroitEnvV1(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, 
            obj_bid_idx= 2,
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
            knn_r_factor= 0,
            ground_truth_type= "nope",
            use_voxel= False,
            forearm_orientation= [0, 0, 0], # forearm orientation
            forearm_relative_position= [0, 0.5, 0], # forearm position related to hand (z-value will be flipped when arm faced down)
            reset_mode= "normal",
            knn_k= 1, # k setting
        ):

        self.forearm_orientation = forearm_orientation
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
        self.knn_r_factor = knn_r_factor
        self.ground_truth_type = ground_truth_type
        self.use_voxel = use_voxel
        
        self.obj_bid_idx = obj_bid_idx
        self.obj_name = [
            "plane",
            "glass",
            ["OShape1","OShape2","OShape3","OShape4","OShape5","OShape6"],
            "LShape",
            "simpleShape",
            "TShape",
            "thinShape",
            "VShape",
            "Egg",
            "Cylinder",
        ]
        self.mesh_name = [
            "plane",
            "glass",
            ["OShape1","OShape2","OShape3","OShape4","OShape5","OShape6"],
            "LShape",
            "simpleShape",
            "TShape",
            "thinShape",
            "VShape",
            "Egg",
            "Cylinder",
        ]
        # dumy obj_bid_list, updated later in this method
        self.obj_bid_list = [0 for _ in range(len(self.obj_name))]
        self.voxel_array = [0] * 280
        self.forearm_obj_bid = 0
        self.S_grasp_sid = 0
        self.ffknuckle_obj_bid = 0
        self.mfknuckle_obj_bid = 0
        self.rfknuckle_obj_bid = 0
        self.lfmetacarpal_obj_bid = 0
        self.thbase_obj_bid = 0
        
        self.count_step = 0
        self.previous_contact_points = []
        self.new_current_pos_list = []
        # scales
        self.act_mid = 0
        self.act_rng = 0

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_constrainedhand.xml', 5)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.prev_a
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

        # update obj_bid_list to make it actually working
        self.obj_bid_list = [
            (self.sim.model.body_name2id('Object{}'.format(idx+1)) if 'Object{}'.format(idx+1) in self.sim.model._body_name2id else 0) \
                for idx in range(len(self.obj_name))
        ]
        # TODO: mesh list and mesh name list
        if self.ground_truth_type == "mesh":
            self.obj_mesh_current_name = self.mesh_name[self.obj_bid_idx]
            self.obj_current_gt = self.model.mesh_vert[self.model.mesh_vertadr[self.sim.model.mesh_name2id(self.obj_mesh_current_name)]:self.model.mesh_vertadr[self.sim.model.mesh_name2id(self.obj_mesh_current_name)]+self.model.mesh_vertnum[self.sim.model.mesh_name2id(self.obj_mesh_current_name)]]
        elif self.ground_truth_type == "sample":
            self.obj_current_gt = np.load(os.path.join("/home/jianrenw/prox/tslam/data/local/agent", "gt_pcloud", "groundtruth_obj4.npz"))['pcd']
        # confB
        self.voxel_array = [0] * 280
        # confC
        # self.voxel_array = [0] * 280

    def get_voxel_idx(self, posx, posy, posz):
        # currently only suitable for obj4
        posx = max(min(posx, 0.1 - 1e-4), -0.1)
        posy = max(min(posy, 0.16 - 1e-4), -0.12)
        posz = max(min(posz, 0.2 - 1e-4), 0.16)
        # unit: 0.01 x 0.01 x 0.01
        # confA: 1x1x1 -> 20x28x4
        # confB: 2x2x2 -> 10x14x2
        unit = 0.02
        # confC: 4x4x4 -> 5x7x1
        # unit = 0.04
        idx_x = math.floor((posx + 0.1) / unit)
        idx_y = math.floor((posy + 0.12) / unit)
        idx_z = math.floor((posz - 0.16) / unit)
        # confB
        voxel_idx = idx_z * 10 * 14 + idx_y * 10 + idx_x
        return voxel_idx

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

    def get_chamfer_distance_loss(self, is_touched, previous_pos_list, current_pos_list):
        chamfer_distance_loss = 0.0
        if "nope" not in self.ground_truth_type:
            if is_touched and self.previous_contact_points != [] and previous_pos_list != []:
                gt_dist1, gt_dist2 = chamfer_dist(torch.FloatTensor([self.obj_current_gt]), torch.FloatTensor([current_pos_list]))
                chamfer_distance_loss = (torch.mean(gt_dist1)) + (torch.mean(gt_dist2))
        else:
            if is_touched and self.previous_contact_points != [] and previous_pos_list != []:
                dist1, dist2 = chamfer_dist(torch.FloatTensor([previous_pos_list]), torch.FloatTensor([current_pos_list]))
                chamfer_distance_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return chamfer_distance_loss

    def get_knn_reward(self):
        return 0

    def get_penalty(self):
        return 0

    def step(self, a):
        # apply action and step
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mid + a*self.act_rng
        # all_points = np.array(self.previous_contact_points)
        # if len(all_points) > 0:
        #     extreme_points = np.concatenate((all_points[all_points[:, 0].argmin()], all_points[all_points[:, 1].argmin()], all_points[all_points[:, 2].argmin()], all_points[all_points[:, 0].argmax()], all_points[all_points[:, 1].argmax()], all_points[all_points[:, 2].argmax()]))
        #     if (all_points[:, 1].max() + 0.14) > (-0.14 - all_points[:, 1].min()):
        #         print(">>> r")
        #         a[0] = self.act_mid[0] + a[0] * self.act_rng[0]
        #     else:
        #         print(">>> l")
        #         a[0] = self.act_mid[0] - a[0] * self.act_rng[0]

        #     print(extreme_points)
        # if self.prev_a is None:
        #     self.prev_a = a
        # else:
        #     self.prev_a = a
        
        self.do_simulation(a, self.frame_skip)

        self.count_step += 1
        obj_init_xpos  = self.data.body_xpos[self.obj_bid_list[self.obj_bid_idx]].ravel()
        palm_xpos = self.data.site_xpos[self.S_grasp_sid].ravel()
        ffknuckle_xpos = self.data.site_xpos[self.ffknuckle_obj_bid].ravel()
        mfknuckle_xpos = self.data.site_xpos[self.mfknuckle_obj_bid].ravel()
        rfknuckle_xpos = self.data.site_xpos[self.rfknuckle_obj_bid].ravel()
        lfmetacarpal_xpos = self.data.site_xpos[self.lfmetacarpal_obj_bid].ravel()
        thbase_xpos = self.data.site_xpos[self.thbase_obj_bid].ravel()
        reward = 0.0
        untouched_p = 0.0
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
            if self.sim.model.geom_id2name(contact.geom1) in self.obj_name[self.obj_bid_idx] or self.sim.model.geom_id2name(contact.geom2) in self.obj_name[self.obj_bid_idx]:#["handle", "neck", "head"]:
                current_pos_list.append(contact.pos.tolist())
                is_touched = True
        
        # untouch penalty
        untouched_p -= 0.01 if self.untouch_p_factor and not is_touched else 0

        # dedup item
        current_pos_list = [item for item in current_pos_list if current_pos_list.count(item) == 1]

        new_pos_list = []
        min_pos_dist = None
        knn_r = 0
        newpoints_r = 0

        previous_pos_list = self.previous_contact_points.copy()
        next_pos_list = self.previous_contact_points.copy()
        
        for pos in current_pos_list:
            if pos not in next_pos_list:
                next_pos_list.append(pos)  
            # new contact points
            if pos not in self.previous_contact_points:
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
      
        chamfer_loss = self.get_chamfer_distance_loss(is_touched, previous_pos_list, next_pos_list)
        chamfer_r = 1 / (chamfer_loss) if chamfer_loss > 0 else 0 # self.get_chamfer_reward(chamfer_loss)
        chamfer_r -= 300
        self.previous_contact_points = next_pos_list.copy()
        #
        # start computing reward
        # for simplicity goal_achieved depends on the nubmer of touched points
        goal_achieved = (len(self.previous_contact_points) > self.goal_threshold)
        if "nope" not in self.ground_truth_type and is_touched and self.previous_contact_points != [] and previous_pos_list != [] and chamfer_loss < 0.06:
            goal_achieved = True

        if self.mesh_p_factor and len(self.previous_contact_points) > 0:
            pv_cloud = pv.PolyData(np.array(self.previous_contact_points))
            pv_volume = pv_cloud.delaunay_3d(alpha= self.mesh_reconstruct_alpha)
            pv_shell = pv_volume.extract_geometry()
            reconstruct_points = pv_shell.points
            mesh_p = chamfer_dist(
                torch.FloatTensor([reconstruct_points]),
                torch.FloatTensor([self.previous_contact_points]),
            )
            mesh_p = (-1) * (torch.mean(mesh_p[0]) + torch.mean(mesh_p[1]))
        else:
            mesh_p = 0
        
        reward += self.palm_r_factor * palm_r
        reward += self.untouch_p_factor * untouched_p
        reward += self.chamfer_r_factor * chamfer_r
        reward += self.newpoints_r_factor * newpoints_r
        reward += self.mesh_p_factor * mesh_p
        reward += self.knn_r_factor * knn_r
        done = False
        info = dict(
            pointcloud= np.array(self.previous_contact_points),
            goal_achieved= goal_achieved,
            untouched_p= untouched_p,
            palm_r = palm_r,
            chamfer_r= chamfer_r,
            newpoints_r= newpoints_r,
            mesh_p= mesh_p,
            knn_r= knn_r,
            voxel_array= np.array(self.voxel_array),
            chamfer_loss_p= chamfer_loss,
        )
        return self.get_obs(), reward, done, info

    def get_obs(self):
        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        obj_init_xpos = self.data.body_xpos[self.obj_bid_list[self.obj_bid_idx]].ravel()
        palm_xpos = self.data.site_xpos[self.S_grasp_sid].ravel()
        # touch_points = len(self.previous_contact_points)
        # touch_pos = self.previous_contact_points[touch_points-3:] if touch_points >= 3 else [0,0,0,0,0,0,0,0,0]

        # 6 extreme pos
        all_points = np.array(self.previous_contact_points)
        if len(all_points) > 0:
            extreme_points = np.concatenate((all_points[all_points[:, 0].argmin()], all_points[all_points[:, 1].argmin()], all_points[all_points[:, 2].argmin()], all_points[all_points[:, 0].argmax()], all_points[all_points[:, 1].argmax()], all_points[all_points[:, 2].argmax()]))
        else:
            extreme_points = (0.5 * np.random.randn(1, 18))
        # 3 random pos
        # touch_points = len(self.new_current_pos_list)
        # touch_pos = self.new_current_pos_list[touch_points-3:] if touch_points >= 3 else [0,0,0,0,0,0,0,0,0]
        # if touch_points >= 3:
        #     touch_pos = random.sample(self.new_current_pos_list, 3)
        # elif len(self.previous_contact_points) >= 3:
        #     touch_pos = random.sample(self.previous_contact_points, 3)
        # else:
        #     touch_pos = (0.5 * np.random.randn(1, 9))
        
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
        # touch_pos = np.array(touch_pos)
        
        ffknuckle_xpos = self.data.body_xpos[self.ffknuckle_obj_bid].ravel()
        mfknuckle_xpos = self.data.body_xpos[self.mfknuckle_obj_bid].ravel()
        rfknuckle_xpos = self.data.body_xpos[self.rfknuckle_obj_bid].ravel()
        lfmetacarpal_xpos = self.data.body_xpos[self.lfmetacarpal_obj_bid].ravel()
        thbase_xpos = self.data.body_xpos[self.thbase_obj_bid].ravel()
        # return np.concatenate([qp[:-6], palm_xpos, obj_init_xpos, palm_xpos-obj_init_xpos, ffknuckle_xpos, mfknuckle_xpos, rfknuckle_xpos, lfmetacarpal_xpos, thbase_xpos,np.concatenate((np.array(touch_pos).flatten(), np.array(extreme_points).flatten()))])
        
        if len(self.previous_contact_points) > 0:
            for point in self.previous_contact_points:
                idx = self.get_voxel_idx(point[0], point[1], point[2])
                # confB
                if len(self.voxel_array) == 0:
                    self.voxel_array = [-1] * 280
                self.voxel_array[min(idx, 279)] = 1

        # final_observation = np.concatenate([qv[:-6], qv[:-6], np.array(self.voxel_array)]) if self.use_voxel else np.concatenate([qv[:-6], qv[:-6], np.concatenate((np.array(touch_pos).flatten(), np.array(extreme_points).flatten()))])
        final_observation = np.concatenate([qv[:-6], np.array(self.voxel_array)]) if self.use_voxel else np.concatenate([qp[:-6], np.concatenate((np.array(touch_pos).flatten(), np.array(extreme_points).flatten()))])
        return final_observation

    def reset_model(self, obj_bid_idx= None):
        current_reset = False
        if self.reset_mode == "random" and random.randint(0,9) > 7:
            current_reset = True

        obj_pos = [0,0,-2.0]
        for obj_bid in self.obj_bid_list:
            self.model.body_pos[obj_bid] = np.array(obj_pos)
        # set target object pose
        self.model.body_pos[self.obj_bid_list[self.obj_bid_idx]] = self.obj_relative_position
        self.model.body_quat[self.obj_bid_list[self.obj_bid_idx]] = euler2quat(self.obj_orientation)

        if not current_reset:
            self.sim.forward()
            return self.get_obs()

        # reset model
        if obj_bid_idx is not None:
            assert 0 <= obj_bid_idx and obj_bid_idx < len(self.obj_bid_list)
            self.obj_bid_idx = obj_bid_idx
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        if self.reset_mode == "normal":
            # clear each episode
            self.count_step = 0
            self.previous_contact_points = []
            self.new_current_pos_list = []

        # self.obj_current_gt = list(np.array(self.obj_current_gt) + np.array([self.obj_relative_position]).repeat(len(self.obj_current_gt), axis=0))

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
        obj_init_pos = self.model.body_pos[self.obj_bid_list[self.obj_bid_idx]].ravel().copy()
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
        self.model.body_pos[self.obj_bid_list[self.obj_bid_idx]] = obj_init_pos
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
