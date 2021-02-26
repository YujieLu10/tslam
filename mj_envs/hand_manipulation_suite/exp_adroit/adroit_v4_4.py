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

class AdroitEnv4V4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_bid = 0
        self.forearm_obj_bid = 0
        self.S_grasp_sid = 0
        self.eps_ball_sid = 0
        self.obj_bid_idx = 0
        self.obj1_bid = 0
        self.obj2_bid = 0
        self.obj3_bid = 0
        self.obj4_bid = 0
        self.obj5_bid = 0
        self.obj6_bid = 0
        self.obj7_bid = 0
        self.obj8_bid = 0
        self.obj_bid_list = [self.obj1_bid, self.obj2_bid, self.obj3_bid, self.obj4_bid, self.obj5_bid, self.obj6_bid, self.obj7_bid, self.obj8_bid]
        self.obj_name = ["plane", "glass", "OShape", "LShape", "simpleShape", "TShape", "thinShape", "VShape"]
        self.obj_t_sid = 0
        self.obj_b_sid = 0
        self.tar_t_sid = 0
        self.tar_b_sid = 0
        self.pen_length = 1.0
        self.tar_length = 1.0
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
        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.forearm_obj_bid = self.sim.model.body_name2id("forearm")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        # self.obj_bid = self.sim.model.body_name2id('Object')
        self.obj1_bid = self.sim.model.body_name2id('Object1')
        self.obj2_bid = self.sim.model.body_name2id('Object2')
        self.obj3_bid = self.sim.model.body_name2id('Object3')
        self.obj4_bid = self.sim.model.body_name2id('Object4')
        self.obj5_bid = self.sim.model.body_name2id('Object5')
        self.obj6_bid = self.sim.model.body_name2id('Object6')
        self.obj7_bid = self.sim.model.body_name2id('Object7')
        self.obj8_bid = self.sim.model.body_name2id('Object8')
        self.obj_bid_list = [self.obj1_bid, self.obj2_bid, self.obj3_bid, self.obj4_bid, self.obj5_bid, self.obj6_bid, self.obj7_bid, self.obj8_bid]
        self.eps_ball_sid = self.sim.model.site_name2id('eps_ball')
        self.obj_t_sid = self.sim.model.site_name2id('object_top')
        self.obj_b_sid = self.sim.model.site_name2id('object_bottom')
        self.tar_t_sid = self.sim.model.site_name2id('target_top')
        self.tar_b_sid = self.sim.model.site_name2id('target_bottom')

        self.pen_length = np.linalg.norm(self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

    def step(self, a):
        self.count_step += 1
        a = np.clip(a, -1.0, 1.0)
        try:
            starting_up = False
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            starting_up = True
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)

        obj_init_pos  = self.data.body_xpos[self.obj_bid_list[self.obj_bid_idx]].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()

        # pos cost
        dist = np.linalg.norm(obj_init_pos-palm_pos) * 10
        reward = -dist
        current_pos_list = []
        is_touched = False
        for contact in self.data.contact:
            if self.sim.model.geom_id2name(contact.geom1) in [self.obj_name[self.obj_bid_idx]] or self.sim.model.geom_id2name(contact.geom2) in [self.obj_name[self.obj_bid_idx]]:#["handle", "neck", "head"]:
                current_pos_list.append(contact.pos.tolist())
                is_touched = True
        # exp6 is_touched reward -> new contact points reward
        current_pos_list = [item for item in current_pos_list if current_pos_list.count(item) == 1]
        new_points_cnt = 0
        for pos in current_pos_list:
            if pos not in self.previous_contact_points:
                new_points_cnt += 1
                reward += 0.05
        # contact points bonus
        if new_points_cnt > 10:
            reward += 1
        elif new_points_cnt > 5:
            reward += 0.3
        previous_pos_list = self.previous_contact_points
        self.previous_contact_points = previous_pos_list + current_pos_list
        if not starting_up and is_touched and self.previous_contact_points != [] and previous_pos_list != []:
            dist1, dist2 = chamfer_dist(torch.FloatTensor([previous_pos_list]), torch.FloatTensor([self.previous_contact_points]))
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            # we want current contact points discover significantly different reconstructed shape
            if loss >= 1e-6:
                loss = 1
            elif loss <= 1e-15:
                loss = 0
            else:
                loss = (loss - 1e-15) / (1e-6 - 1e-15)
            reward += loss

        if self.count_step % 200 == 0:
            print(">>> self.count_step {} reward {}".format(self.count_step, reward))
        # with open('../mjrl/train_vis_material/adroitV1_train/adroit_random_point_cloud_all.txt', 'a') as pf:
        #     if self.count_step % 100 == 0:
        #         pf.write("step"+str(self.count_step)+'\n')
        #     for line in current_pos_list:
        #         pf.write(str(line) + '\n')
        done = False
        goal_achieved = False

        return self.get_obs(), reward, done, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qp = self.data.qpos.ravel()
        # obj_init_pos = self.data.body_xpos[self.obj_bid_list[self.obj_bid_idx]].ravel()
        # palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        # return np.concatenate([qp[:-6], palm_pos, palm_pos-obj_init_pos])
        qp = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.body_xpos[self.obj_bid_list[self.obj_bid_idx]].ravel()
        desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length
        return np.concatenate([qp[:-6], obj_pos, obj_vel, obj_orien, desired_orien, obj_pos-desired_pos, obj_orien-desired_orien])

    def reset_model(self):
        self.obj_bid_idx = (self.obj_bid_idx + 1) % len(self.obj_bid_list)
        # clear each episode
        self.previous_contact_points = []
        self.ratio = random.randint(-5, 5)
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        # desired_orien = np.zeros(3)
        # desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        # desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        # self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        obj_pos = [0,0,-2.0]
        obj_init_pos = [-0.00,-0.18,0.215]
        for obj_bid in self.obj_bid_list:
            self.model.body_pos[obj_bid] = np.array(obj_pos)
        self.model.body_pos[self.obj_bid_list[self.obj_bid_idx]] = np.array(obj_init_pos)
        # for eight objects
        # pos_list = [[0, -0.7, 0.1], [0, -0.7, 0.33], [0.12, -0.69, 0.23], [-0.14, -0.69, 0.23]]
        # orien_list = [[-1.57, 0, 0], [-1.57, 0, 3], [-1.57, 0, 4.5], [-1.57, 0, 2]]
        pos_list = [[0, -0.7, 0.16],[0, -0.7, 0.26]]
        orien_list = [[-1.57, 0, 0],[-1.57, 0, 3]]
        # for ball
        # pos_list = [[0, -0.7, 0.2], [0, -0.7, 0.23], [0.02, -0.69, 0.23], [-0.02, -0.69, 0.23]]
        # orien_list = [[-1.57, 0, 0], [-1.57, 0, 3], [-1.57, 0, 4.5], [-1.57, 0, 2]]
        # for hammer 
        # pos_list = [[0.1, -0.69, 0.23], [-0.025, -0.69, 0.23], [0, -0.7, 0.185], [0, -0.7, 0.245]]
        # orien_list = [[-1.57, 0, 4.5], [-1.57, 0, 2], [-1.57, 0, 0], [-1.57, 0, 3]]

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
        # desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        forearm_orien = self.model.body_quat[self.forearm_obj_bid].ravel().copy()
        forearm_pos = self.model.body_pos[self.forearm_obj_bid].ravel().copy()
        obj_init_pos = self.model.body_pos[self.obj_bid_list[self.obj_bid_idx]].ravel().copy()
        return dict(qpos=qp, qvel=qv, forearm_orien=forearm_orien, forearm_pos=forearm_pos, obj_init_pos=obj_init_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        forearm_orien = state_dict['forearm_orien']
        forearm_pos = state_dict['forearm_pos']
        obj_init_pos = state_dixt['obj_init_pos']
        self.set_state(qp, qv)
        self.model.body_quat[self.forearm_obj_bid] = forearm_orien
        self.model.body_pos[self.forearm_obj_bid] = forearm_pos
        self.model.body_pos[self.obj_bid_list[self.obj_bid_idx]] = obj_init_pos

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
        print(">>> evaluate_success")
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 20:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
