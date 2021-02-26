import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mj_envs.utils.quatmath import quat2euler, euler2quat
from mujoco_py import MjViewer
import os
import random
import torch
from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

ADD_BONUS_REWARDS = True

class AdroitEnv2V4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.forearm_obj_bid = 0
        self.S_grasp_sid = 0
        self.ffknuckle_obj_bid = 0
        self.mfknuckle_obj_bid = 0
        self.rfknuckle_obj_bid = 0
        self.lfmetacarpal_obj_bid = 0
        self.thbase_obj_bid = 0
        self.obj_bid_idx = 2
        self.obj1_bid = 0
        self.obj2_bid = 0
        self.obj3_bid = 0
        self.obj4_bid = 0
        self.obj5_bid = 0
        self.obj6_bid = 0
        self.obj7_bid = 0
        self.obj8_bid = 0
        self.obj_bid_list = [self.obj1_bid, self.obj2_bid, self.obj3_bid, self.obj4_bid, self.obj5_bid, self.obj6_bid, self.obj7_bid, self.obj8_bid]
        self.obj_name = ["plane", "glass", ["OShape1","OShape2","OShape3","OShape4","OShape5","OShape6"], "LShape", "simpleShape", "TShape", "thinShape", "VShape"]
        self.ratio = 1
        self.count_step = 0
        self.previous_contact_points = []
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
        self.obj1_bid = 0#self.sim.model.body_name2id('Object1')
        self.obj2_bid = 0#self.sim.model.body_name2id('Object2')
        self.obj3_bid = self.sim.model.body_name2id('Object3')
        self.obj4_bid = 0#self.sim.model.body_name2id('Object4')
        self.obj5_bid = 0#self.sim.model.body_name2id('Object5')
        self.obj6_bid = 0#self.sim.model.body_name2id('Object6')
        self.obj7_bid = 0#self.sim.model.body_name2id('Object7')
        self.obj8_bid = 0#self.sim.model.body_name2id('Object8')
        self.obj_bid_list = [self.obj1_bid, self.obj2_bid, self.obj3_bid, self.obj4_bid, self.obj5_bid, self.obj6_bid, self.obj7_bid, self.obj8_bid]

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

    def step(self, a):
        # a = np.clip(a, -1.0, 1.0)
        # try:
        #     starting_up = False
        #     a = self.act_mid + a*self.act_rng # mean center and scale
        # except:
        #     starting_up = True
        #     a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        self.count_step += 1
        obj_init_xpos  = self.data.body_xpos[self.obj_bid_list[self.obj_bid_idx]].ravel()
        palm_xpos = self.data.site_xpos[self.S_grasp_sid].ravel()
        
        # palm close to object reward
        # dist = np.linalg.norm(obj_init_xpos-palm_xpos)
        # reward = -dist
        reward = 0

        # pos of current obj's contacts
        current_pos_list = []
        is_touched = False
        new_points_cnt = 0
        # contacts of current obj
        for contact in self.data.contact:
            if self.sim.model.geom_id2name(contact.geom1) in self.obj_name[self.obj_bid_idx] or self.sim.model.geom_id2name(contact.geom2) in self.obj_name[self.obj_bid_idx]:#["handle", "neck", "head"]:
                current_pos_list.append(contact.pos.tolist())
                is_touched = True
        
        # not touched punish
        if not is_touched:
            reward -= 20

        # dedup item
        current_pos_list = [item for item in current_pos_list if current_pos_list.count(item) == 1]
        new_pos_list = []
        min_pos_dist = None
        for pos in current_pos_list:
            # new contact points
            if pos not in self.previous_contact_points:
                min_pos_dist = 1
                for previous_pos in self.previous_contact_points:
                    pos_dist = np.linalg.norm(np.array(pos) - np.array(previous_pos))
                    min_pos_dist = pos_dist if min_pos_dist is None else min(min_pos_dist, pos_dist)
                # new contact points that are not close to already touched points
                # print(min_pos_dist)
                if min_pos_dist and min_pos_dist > 0.01: 
                    new_points_cnt += 1  
                    reward += 2
                    new_pos_list.append(pos)
                    # bonus
                    if min_pos_dist > 0.1:
                        reward += 3

        # new contact points bonus
        if new_points_cnt > 10:
            reward += 5
        elif new_points_cnt > 5:
            reward += 2
        previous_pos_list = self.previous_contact_points.copy()
        current_pos_list = self.previous_contact_points.copy()
        for item in new_pos_list:
            if item not in current_pos_list:
                current_pos_list.append(item)        
        # self.previous_contact_points = previous_pos_list + current_pos_list
        # dedup
        # self.previous_contact_points = [item for item in self.previous_contact_points if self.previous_contact_points.count(item) == 1]
        # print(">>> count_step:{} reward:{}".format(self.count_step, reward))
        chamfer_reward = 0
        if is_touched and self.previous_contact_points != [] and previous_pos_list != []:
            dist1, dist2 = chamfer_dist(torch.FloatTensor([previous_pos_list]), torch.FloatTensor([current_pos_list]))
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            # print(">>> loss:{}".format(loss))
            # chamfer_dist loss normalization
            if loss >= 1e-6:
                loss = 1
            elif loss <= 1e-15:
                loss = 0
            else:
                loss = (loss - 1e-15) / (1e-6 - 1e-15)
            chamfer_reward += loss * 10
            # print(">>> chamfer_reward:{}".format(chamfer_reward))

            # try ground truth
            # dist1, dist2 = chamfer_dist(torch.FloatTensor([self.obj3_gt]), torch.FloatTensor([self.previous_contact_points]))
            # loss = (torch.mean(dist1)) + (torch.mean(dist2))

        # with open('/home/jianrenw/prox/mj_envs/mjrl/train_vis_material/debug/exp8_adroit_v4_2_contacts.txt', 'a') as pf:
        #     pf.write("step:" + str(self.count_step) +'\n')
        #     for line in self.previous_contact_points:
        #         pf.write(str(line) + '\n')

        reward += chamfer_reward
        done = False
        # for simplicity goal_achieved depends on the nubmer of touched points
        # TODO: use reconstruction loss
        goal_achieved = True if len(self.previous_contact_points) > 8000 else False
        # print(self.get_obs())
        self.previous_contact_points = current_pos_list.copy()
        return self.get_obs(), reward, done, dict(goal_achieved=goal_achieved,recon_reward=chamfer_reward)

    def get_obs(self):
        qp = self.data.qpos.ravel()
        obj_init_xpos = self.data.body_xpos[self.obj_bid_list[self.obj_bid_idx]].ravel()
        palm_xpos = self.data.site_xpos[self.S_grasp_sid].ravel()
        touch_points = len(self.previous_contact_points)
        touch_pos = self.previous_contact_points[touch_points-3:] if touch_points >= 3 else [0,0,0,0,0,0,0,0,0]
        # print(">>> len {} touch_pos {}".format(touch_points, touch_pos))
        ffknuckle_xpos = self.data.body_xpos[self.ffknuckle_obj_bid].ravel()
        mfknuckle_xpos = self.data.body_xpos[self.mfknuckle_obj_bid].ravel()
        rfknuckle_xpos = self.data.body_xpos[self.rfknuckle_obj_bid].ravel()
        lfmetacarpal_xpos = self.data.body_xpos[self.lfmetacarpal_obj_bid].ravel()
        thbase_xpos = self.data.body_xpos[self.thbase_obj_bid].ravel()

        # return np.concatenate([qp[:-6], palm_xpos, obj_init_xpos, palm_xpos-obj_init_xpos, ffknuckle_xpos, mfknuckle_xpos, rfknuckle_xpos, lfmetacarpal_xpos, thbase_xpos])
        return np.concatenate([qp[:-6], palm_xpos, obj_init_xpos, palm_xpos-obj_init_xpos, ffknuckle_xpos, mfknuckle_xpos, rfknuckle_xpos, lfmetacarpal_xpos, thbase_xpos, np.array(touch_pos).flatten()])

    def reset_model(self):
        self.obj_bid_idx = 2 #(self.obj_bid_idx + 1) % len(self.obj_bid_list)
        # clear each episode
        self.count_step = 0
        self.previous_contact_points = []
        self.ratio = random.randint(-5, 5)
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        obj_pos = [0,0,-2.0]
        # current learned obj initiated near forearm
        obj_init_pos = [-0.00,-0.22,0.215]
        for obj_bid in self.obj_bid_list:
            self.model.body_pos[obj_bid] = np.array(obj_pos)
        self.model.body_pos[self.obj_bid_list[self.obj_bid_idx]] = np.array(obj_init_pos)
        # four pos and orien
        # pos_list = [[0, -0.7, 0.1], [0, -0.7, 0.33], [0.12, -0.69, 0.23], [-0.14, -0.69, 0.23]]
        # orien_list = [[-1.57, 0, 0], [-1.57, 0, 3], [-1.57, 0, 4.5], [-1.57, 0, 2]]
        # only top|bottom
        pos_list = [[0, -0.7, 0.16],[0, -0.7, 0.26]]
        orien_list = [[-1.57, 0, 0],[-1.57, 0, 3]]

        idx = random.randint(0,len(orien_list) - 1)
        forearm_orien = np.array(orien_list[idx])
        forearm_pos = np.array(pos_list[idx])
        self.model.body_quat[self.forearm_obj_bid] = euler2quat(forearm_orien)
        self.model.body_pos[self.forearm_obj_bid] = forearm_pos

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
        self.viewer.cam.distance = 1.0

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if pen within 15 degrees of target for 20 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 20:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
