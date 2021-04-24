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
from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()
from pathlib import Path

ADD_BONUS_REWARDS = True

class AdroitEnvV2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, 
            obj_bid_idx= 2,
            obj_orientation= [0, 0, 0], # object orientation
            obj_relative_position= [0, 0.5, 0], # object position related to hand (z-value will be flipped when arm faced down)
            goal_threshold= int(8e3), # how many points touched to achieve the goal
            new_point_threshold= 0.01, # minimum distance new point to all previous points
            forearm_orientation= "up", # ("up", "down")
            chamfer_r_factor= 0,
            mesh_p_factor= 0,
            mesh_reconstruct_alpha= 0.01,
            palm_r_factor= 0,
            untouch_p_factor= 0,
            newpoints_r_factor= 0,
            knn_r_factor= 0,
            chamfer_use_gt= False,
        ):
        # self.touched_points = list() # record all valid points since reset
        self.forearm_orientation = forearm_orientation
        self.obj_orientation = obj_orientation
        self.obj_relative_position = obj_relative_position

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
        self.chamfer_use_gt = chamfer_use_gt
        
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

        self.forearm_obj_bid = 0
        self.S_grasp_sid = 0
        self.ffknuckle_obj_bid = 0
        self.mfknuckle_obj_bid = 0
        self.rfknuckle_obj_bid = 0
        self.lfmetacarpal_obj_bid = 0
        self.thbase_obj_bid = 0
        self.ratio = 1
        self.count_step = 0
        self.previous_contact_points = []
        self.new_current_pos_list = []
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_constrainedhand.xml', 5)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

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
        self.obj_mesh_current_name = self.mesh_name[self.obj_bid_idx]
        self.obj_current_gt = self.model.mesh_vert[self.model.mesh_vertadr[self.sim.model.mesh_name2id(self.obj_mesh_current_name)]:self.model.mesh_vertadr[self.sim.model.mesh_name2id(self.obj_mesh_current_name)]+self.model.mesh_vertnum[self.sim.model.mesh_name2id(self.obj_mesh_current_name)]]

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

    def get_chamfer_reward(self, is_touched, previous_pos_list, current_pos_list):
        chamfer_reward = 0
        if self.chamfer_use_gt:
            if is_touched and self.previous_contact_points != [] and previous_pos_list != []:
                gt_dist1, gt_dist2 = chamfer_dist(torch.FloatTensor([self.obj_current_gt]), torch.FloatTensor([current_pos_list]))
                gt_loss = (torch.mean(gt_dist1)) + (torch.mean(gt_dist2))
                chamfer_reward += (0.1-gt_loss) * 10
        else:
            if is_touched and self.previous_contact_points != [] and previous_pos_list != []:
                dist1, dist2 = chamfer_dist(torch.FloatTensor([previous_pos_list]), torch.FloatTensor([current_pos_list]))
                loss = (torch.mean(dist1)) + (torch.mean(dist2))
                chamfer_reward += self.loss_transform(loss) * 10
        return chamfer_reward

    def get_knn_reward(self):
        return 0

    def get_penalty(self):
        return 0

    def step(self, a):
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
            if self.sim.model.geom_id2name(contact.geom1) in self.obj_name[self.obj_bid_idx] or self.sim.model.geom_id2name(contact.geom2) in self.obj_name[self.obj_bid_idx]:#["handle", "neck", "head"]:
                current_pos_list.append(contact.pos.tolist())
                is_touched = True
        
        # untouch penalty
        untouched_p = -1 if self.untouch_p_factor and not is_touched else -1

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
                for previous_pos in self.previous_contact_points:
                    pos_dist = np.linalg.norm(np.array(pos) - np.array(previous_pos))
                    min_pos_dist = pos_dist if min_pos_dist is None else min(min_pos_dist, pos_dist)
                # new contact points that are not close to already touched points
                if min_pos_dist and min_pos_dist > 0.01: 
                    new_points_cnt += 1  
                    newpoints_r += self.get_newpoints_reward(min_pos_dist)
                    new_pos_list.append(pos)
                knn_r += min_pos_dist
        
        self.new_current_pos_list = new_pos_list
        # similar points penalty
        # penalty_sim = similar_points_cnt * 5
        # penalty_sim = self.get_penalty()
      

        chamfer_r = self.get_chamfer_reward(is_touched, previous_pos_list, next_pos_list)
        self.previous_contact_points = next_pos_list.copy()
        #
        # start computing reward
        # for simplicity goal_achieved depends on the nubmer of touched points
        goal_achieved = (len(self.previous_contact_points) > self.goal_threshold)
        if self.chamfer_use_gt:
            if is_touched and self.previous_contact_points != [] and previous_pos_list != []:
                gt_dist1, gt_dist2 = chamfer_dist(torch.FloatTensor([self.obj_current_gt]), torch.FloatTensor([next_pos_list]))
                gt_loss = (torch.mean(gt_dist1)) + (torch.mean(gt_dist2))
                if gt_loss < 0.06:
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
            untouched_p= untouched_p * self.untouch_p_factor,
            palm_r = palm_r * self.palm_r_factor ,
            chamfer_r= chamfer_r * self.chamfer_r_factor,
            newpoints_r= newpoints_r * self.newpoints_r_factor,
            mesh_p= mesh_p * self.mesh_p_factor ,
            knn_r= knn_r * self.knn_r_factor,
        )
        return self.get_obs(), reward, done, info

    def get_obs(self):
        qp = self.data.qpos.ravel()
        obj_init_xpos = self.data.body_xpos[self.obj_bid_list[self.obj_bid_idx]].ravel()
        palm_xpos = self.data.site_xpos[self.S_grasp_sid].ravel()
        # touch_points = len(self.previous_contact_points)
        # touch_pos = self.previous_contact_points[touch_points-3:] if touch_points >= 3 else [0,0,0,0,0,0,0,0,0]
        touch_points = len(self.new_current_pos_list)
        touch_pos = self.new_current_pos_list[touch_points-3:] if touch_points >= 3 else (0.1 * np.random.randn(1, 9)) # [0,0,0,0,0,0,0,0,0]
        ffknuckle_xpos = self.data.body_xpos[self.ffknuckle_obj_bid].ravel()
        mfknuckle_xpos = self.data.body_xpos[self.mfknuckle_obj_bid].ravel()
        rfknuckle_xpos = self.data.body_xpos[self.rfknuckle_obj_bid].ravel()
        lfmetacarpal_xpos = self.data.body_xpos[self.lfmetacarpal_obj_bid].ravel()
        thbase_xpos = self.data.body_xpos[self.thbase_obj_bid].ravel()
        
        return 0.5 * np.random.randn(1, 66)
        # return np.concatenate([qp[:-6], palm_xpos, obj_init_xpos, palm_xpos-obj_init_xpos, ffknuckle_xpos, mfknuckle_xpos, rfknuckle_xpos, lfmetacarpal_xpos, thbase_xpos, np.array(touch_pos).flatten()])

    def reset_model(self, obj_bid_idx= None):
        self.touched_points = list()
        if obj_bid_idx is not None:
            assert 0 <= obj_bid_idx and obj_bid_idx < len(self.obj_bid_list)
            self.obj_bid_idx = obj_bid_idx
        # clear each episode
        self.count_step = 0
        self.previous_contact_points = []
        self.new_current_pos_list = []
        self.ratio = random.randint(-5, 5)
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        # set all objects position
        obj_pos = [0,0,-2.0]
        for obj_bid in self.obj_bid_list:
            self.model.body_pos[obj_bid] = np.array(obj_pos)

        # set forearm pose
        # four pos and orien
        # pos_list = [[0, -0.7, 0.1], [0, -0.7, 0.33], [0.12, -0.69, 0.23], [-0.14, -0.69, 0.23]]
        # orien_list = [[-1.57, 0, 0], [-1.57, 0, 3], [-1.57, 0, 4.5], [-1.57, 0, 2]]
        # only top|bottom
        pos_list = [[0, -0.7, 0.16],[0, -0.7, 0.26]]
        orien_list = [[-1.57, 0, 0],[-1.57, 0, 3]]
        idx = 0 if self.forearm_orientation == "up" else 1
        forearm_orien = np.array(orien_list[idx])
        forearm_pos = np.array(pos_list[idx])
        self.model.body_quat[self.forearm_obj_bid] = euler2quat(forearm_orien)
        self.model.body_pos[self.forearm_obj_bid] = forearm_pos
            
        # set target object pose
        if self.forearm_orientation == "up":
            obj_abs_position = np.array(self.obj_relative_position) + forearm_pos
        else:
            obj_abs_position = np.array([
                self.obj_relative_position[0],
                self.obj_relative_position[1],
                -self.obj_relative_position[2],
            ]) + forearm_pos
        self.model.body_pos[self.obj_bid_list[self.obj_bid_idx]] = obj_abs_position
        self.model.body_quat[self.obj_bid_list[self.obj_bid_idx]] = euler2quat(self.obj_orientation)
        self.obj_current_gt = list(np.array(self.obj_current_gt) + np.array([obj_abs_position]).repeat(len(self.obj_current_gt), axis=0))
        # gt_pc_frame = np.array(self.obj_current_gt) + np.array([obj_abs_position]).repeat(len(self.obj_current_gt), axis=0)
        # ax = plt.axes(projection='3d')
        # ax.scatter(gt_pc_frame[:, 0], gt_pc_frame[:, 1], gt_pc_frame[:, 2], c='green', cmap='viridis', linewidth=0.5)

        # file_name = os.path.join("/home/jianrenw/prox/tslam/data/local/agent/gt_pcloud", "obj" + str(self.obj_bid_idx) + ".npz")
        # if not Path(file_name).is_file():
        #     np.savez_compressed(file_name, pcd=gt_pc_frame)

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
