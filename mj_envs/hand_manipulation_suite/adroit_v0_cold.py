import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os
import random

ADD_BONUS_REWARDS = True

class RelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.Touch_palm_sid = 0
        self.list = []
        self.ratio = 1
        self.ratioa = 1
        self.ratiob = 1
        self.ratioc = 1
        self.ratiod = 1
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_relocate_yjl.xml', 5)
        
        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.target_obj_sid = self.sim.model.site_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.Touch_palm_sid = self.sim.model.site_name2id('Tch_palm')
        self.list = []
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        # print(">>> a {}".format(a))
        # a = [ 0.00177033,  0.00969992,  0.10566138, -0.07078566, -0.03343381,  0.02396764,
        #     0.07418799, -0.0550275,  -0.06356242,  0.87415451,  0.99072902,  1.15819077,
        #     0.13383595,  0.80240414,  0.92409155,  0.80283885, -0.08666901,  0.45729976,
        #     1.10996752,  1.13094401,  0.7,        -0.24165964,  0.86459684,  1.52325935,
        #     1.08266764,  1.0,          1.21674265,  0.21181869, -0.47744665, -0.49434806]
        # a = [-0.01095778,  0.0, 0.09735764, -0.07605794,  0.05527397,  0.00730973,
        #     0.06250753, -0.06769375, -0.0407761,   0.74619363,  0.72874663,  0.85902658,
        #     0.12549262,  0.61291656,  0.71764565,  0.67651453, -0.08525737,  0.4982008,
        #     1.14214749,  1.09163322,  0.60468282, -0.25035654,  0.65608048,  1.59690161,
        #     1.06413233,  0.49626786,  1.16367719,  0.20195321, -0.37851378, -0.50428095]
        # ratio : (-1, 1)
        ammend_a = [ 0,  0.004,  0.004,  -0.004,
            0.01,  0.05,  0,  0,
            0,  0,  0,  0,
            0,  0,  0,  0,
            0,  0,  0,  0,
            0,  0,  0,  0,
            0,  0,  0,  0,
            0,  0]

        ammend_a = [i * self.ratio for i in ammend_a]
        self.do_simulation(a + ammend_a, self.frame_skip)
        # random_ammend_a = np.random.rand(30)
        # rotate_a = [ 0.0,  0.02,  0.00,  0.0,
        #     0.0,  0.0,  0,  0,
        #     0,  0,  0,  0,
        #     0,  0,  0,  0,
        #     0,  0,  0,  0,
        #     0,  0,  0,  0,
        #     0,  0,  0,  0,
        #     0,  0]
        # rotate_b = [ 0.0,  0.0,  0.00,  -0.01, 0.0,  0.0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0]
        # rotate_c = [ 0.0,  0.0,  0.00,  0.0, 0.01,  0.0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0]
        # rotate_d = [ 0.0,  0.0,  0.00,  0.0, 0.0,  1,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0]            
        # rotate_a = [i * self.ratioa for i in rotate_a]
        # rotate_b = [i * self.ratiob for i in rotate_b]
        # rotate_c = [i * self.ratioc for i in rotate_c]
        # rotate_d = [i * self.ratiod for i in rotate_d]
        ob = self.get_obs()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        touch_palm_pos = self.data.site_xpos[self.Touch_palm_sid].ravel()
        
        # print(self.obj_bid)
        # print(">>> touch_palm_pos {}".format(touch_palm_pos))
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

        reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
        # if np.linalg.norm(palm_pos-obj_pos) < 0.05:
        #     reward += 5.0
        #     # self.do_simulation(a + rotate_a + rotate_b + rotate_c + rotate_d, self.frame_skip)
        #     self.do_simulation(a + ammend_a, self.frame_skip)
        # else:
        #     self.do_simulation(a, self.frame_skip)

        current_pos_list = []
        for contact in self.data.contact:
            # if contact.geom2 != 0:
            #     print(self.sim.model.geom_id2name(contact.geom2))
            # ball contacts  or self.sim.model.geom_id2name(contact.geom2) == "sphere":
            # if self.sim.model.geom_id2name(contact.geom2) == "sphere":
            #     print(self.sim.model.geom_id2name(contact.geom1))
            if self.sim.model.geom_id2name(contact.geom1) == "sphereball":
                current_pos_list.append(contact.pos)
            #     print(self.sim.model.geom_id2name(contact.geom2))
            # hammer contacts
            # if self.sim.model.geom_id2name(contact.geom1) in ["handle", "neck", "head"] or self.sim.model.geom_id2name(contact.geom2) in ["handle", "neck", "head"]:
                
                # print(contact.pos)
        with open('adroit_random_point_cloud_ball_10.txt', 'a') as pf:
            for line in current_pos_list:
                pf.write(str(line) + '\n')
            # pf.write(str(current_pos_list))
        # self.list.append(current_pos_list)
        
        # print(">>> np.linalg.norm(palm_pos-obj_pos) {}".format(np.linalg.norm(palm_pos-obj_pos)))
        # if obj_pos[2] > 0.04:                                       # if object off the table
        #     reward += 1.0                                           # bonus for lifting the object
        #     reward += -0.5*np.linalg.norm(palm_pos-target_pos)      # make hand go to target
        #     reward += -0.5*np.linalg.norm(obj_pos-target_pos)       # make object go to target

        # if ADD_BONUS_REWARDS:
        #     if np.linalg.norm(obj_pos-target_pos) < 0.1:
        #         reward += 10.0                                          # bonus for object close to target
        #     if np.linalg.norm(obj_pos-target_pos) < 0.05:
        #         reward += 20.0                                          # bonus for object "very" close to target

        goal_achieved = True if np.linalg.norm(obj_pos-palm_pos) < 0.1 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])
       
    def reset_model(self):
        self.ratio = random.randint(-5, 5)
        # self.ratioa = random.randint(-5, 5)
        # self.ratiob = random.randint(-5, 5)
        # self.ratioc = random.randint(-5, 5)
        # self.ratiod = random.randint(-5, 5)
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        # self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15, high=0.15)
        # self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15, high=0.3)
        # self.model.site_pos[self.target_obj_sid, 0] = self.model.body_pos[self.obj_bid,0]
        # self.model.site_pos[self.target_obj_sid,1] = self.model.body_pos[self.obj_bid,1]

        # self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        # self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2, high=0.2)
        # self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)

        obj_init_pos_incre = [0.00,0.05,0.00]
        self.model.body_pos[self.obj_bid] = self.model.body_pos[self.obj_bid] + np.array(obj_init_pos_incre)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
            qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        palm_pos = state_dict['palm_pos']
        target_pos = state_dict['target_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.S_grasp_sid] = palm_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
