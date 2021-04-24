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

class AdroitEnvV4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # self.default_config = {'use_gt':False, 'coef_br':1.0, 'coef_cr':1.0, 'coef_kr':1.0}
        self.config = {}
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
        self.obj3_gt = [[5.962460041046143,-0.22303539514541626,3.714493989944458],[6.618359088897705,-0.005236822180449963,1.092887043952942],[5.743826866149902,0.6500856280326843,3.9327199459075928],[-1.2454999685287476,-0.4414748549461365,6.990780830383301],[-1.2454999685287476,0.43164512515068054,6.990780830383301],[-1.2454999685287476,-0.4414753019809723,4.150370121002197],[-1.2454999685287476,-1.7527601718902588,5.460885524749756],[6.399725914001465,0.8685252070426941,1.0928869247436523],[5.088697910308838,-1.3152378797531128,1.092887282371521],[3.777669906616211,0.21320417523384094,1.092887043952942],[2.686044931411743,1.305406928062439,5.679111003875732],[3.777669906616211,0.4316449463367462,5.897914886474609],[5.307331085205078,1.086966633796692,3.9327197074890137],[4.652202129364014,1.5241683721542358,1.9669477939605713],[2.0316851139068604,0.21320509910583496,6.771399021148682],[-0.5896009802818298,1.3054068088531494,5.023853778839111],[-0.8082330226898193,1.0869667530059814,4.586822986602783],[3.996303081512451,-0.8783568143844604,1.0928871631622314],[-1.2454999685287476,1.08696711063385,6.553173065185547],[-1.0268659591674805,-1.5336782932281494,4.586823463439941],[0.06552799791097641,-1.0967968702316284,6.771399021148682],[2.9039080142974854,-1.5336780548095703,5.897915363311768],[5.52596378326416,-1.533678650856018,1.9669482707977295],[5.962460041046143,-0.6599164605140686,3.277462959289551],[4.433568954467773,1.0869662761688232,1.3116909265518188],[2.9039080142974854,-1.7527601718902588,5.460885524749756],[4.214935779571533,-1.3152371644973755,5.023853778839111],[0.06552799791097641,-1.7527600526809692,6.116142272949219],[-1.2454999685287476,0.4316447079181671,4.368597030639648],[3.777669906616211,-0.4414750337600708,5.897914886474609],[1.5951889753341675,1.305406928062439,6.1161417961120605],[4.870065212249756,-1.5336787700653076,1.5299181938171387],[-1.2462689876556396,-1.5342131853103638,5.242786407470703],[-5.3969597816467285,-1.534213900566101,1.0923092365264893],[-4.522342205047607,-1.3156657218933105,1.3113702535629272],[-1.2462689876556396,-1.5342130661010742,6.116805553436279],[-1.2462689876556396,-0.660322368144989,4.150262832641602],[-1.2462689876556396,1.3057068586349487,4.805777072906494],[-1.4649239778518677,0.8686121106147766,6.7723188400268555],[-4.0861029624938965,0.6500639319419861,5.679794788360596],[-2.339005947113037,-0.4417749047279358,6.7723188400268555],[-5.615078926086426,1.3057063817977905,2.403892755508423],[-4.0861029624938965,1.3057068586349487,4.805777072906494],[-6.2705078125,0.43181556463241577,3.4964170455932617],[-6.707817077636719,0.2132682055234909,1.3113700151443481],[-5.8331990242004395,-1.534213900566101,1.0923092365264893],[-6.707817077636719,-0.8788707852363586,1.5298751592636108],[-5.615078926086426,-1.0971183776855469,3.932314157485962],[-4.741530895233154,1.305706262588501,1.5298748016357422],[-4.30422306060791,0.8686111569404602,1.0923088788986206],[-4.959650993347168,-0.44177520275115967,5.024281978607178],[-3.2125539779663086,-0.8788700103759766,6.116805076599121],[-6.0523881912231445,0.8686111569404602,1.0923088788986206],[-1.2462689876556396,1.3057069778442383,5.8983001708984375],[-4.0861029624938965,-0.8788707852363586,1.3113701343536377],[-4.0861029624938965,0.43181517720222473,1.0923088788986206],[-4.741530895233154,-1.752760410308838,3.932314157485962],[-5.177771091461182,-1.5342133045196533,4.15026330947876],[-6.0523881912231445,1.087158441543579,2.840902805328369],[-3.868518114089966,-1.752760410308838,3.496417284011841],[-2.9944350719451904,-1.3156650066375732,5.8983001708984375],[-1.2462689876556396,-0.005278897006064653,6.7723188400268555],[-2.9944350719451904,1.3057069778442383,5.679794788360596],[-1.4649239778518677,0.6500636339187622,4.150262832641602],[5.962716102600098,-0.8786574602127075,-3.0583808422088623],[6.39959716796875,0.43142959475517273,-2.621500015258789],[6.6178669929504395,-0.8786572813987732,-1.7477378845214844],[6.83682107925415,-0.22331421077251434,-1.3108570575714111],[6.83682107925415,0.21288278698921204,-1.3108570575714111],[6.83682107925415,0.21288317441940308,1.0923099517822266],[4.871026992797852,-1.3157519102096558,1.0923101902008057],[3.778995990753174,-0.005065822042524815,1.0923099517822266],[3.342113971710205,-1.5342997312545776,-4.149620532989502],[3.342113971710205,0.6502767205238342,-1.7477381229400635],[3.342113971710205,1.3056195974349976,-2.4027392864227295],[3.342113971710205,-1.0972048044204712,-5.4615478515625],[6.6178669929504395,-1.0972039699554443,0.21726517379283905],[4.215877056121826,1.086773157119751,1.092309832572937],[6.6178669929504395,0.868226170539856,1.092309832572937],[6.180984973907471,1.0867726802825928,-2.184619188308716],[5.089295864105225,1.5241681337356567,0.6541457772254944],[4.215877056121826,-0.22331489622592926,-5.4615478515625],[4.434145927429199,-1.5342997312545776,-4.149620532989502],[6.180984973907471,-1.3157522678375244,-1.7477377653121948],[3.997265100479126,1.305619239807129,-4.36902379989624],[3.342113971710205,0.4314292073249817,-5.023384094238281],[4.6527581214904785,0.8682251572608948,-5.023384094238281],[5.525835037231445,-1.5342991352081299,-0.43773576617240906],[4.215877056121826,1.5241674184799194,-3.712740182876587],[3.342113971710205,-0.6604092121124268,-1.5302598476409912],[6.180984973907471,1.3056198358535767,-0.4377362132072449],[3.997265100479126,-0.6604087948799133,1.0923100709915161],[5.744104862213135,-1.3157519102096558,1.0923101902008057],[3.342113971710205,-1.534299612045288,-3.4952616691589355],[5.307223796844482,-0.6604096293449402,-4.14962100982666],[4.6527581214904785,1.5241681337356567,0.6541457772254944],[3.341430902481079,1.3057053089141846,-4.587699890136719],[3.341430902481079,-1.3158378601074219,-5.461334228515625],[3.341430902481079,1.0870721340179443,-5.023810863494873],[3.341430902481079,0.4318576455116272,-2.1846189498901367],[3.341430902481079,-1.5341285467147827,-3.495774745941162],[-0.15396000444889069,1.0870722532272339,-4.369880199432373],[-0.15396000444889069,0.4318569004535675,-6.772018909454346],[-0.15396000444889069,-1.5341287851333618,-5.023810386657715],[3.341430902481079,1.3057054281234741,-3.71406626701355],[0.7198879718780518,-1.5341285467147827,-3.932356834411621],[3.1224560737609863,-0.878913402557373,-2.4033796787261963],[0.7198879718780518,-0.8789135217666626,-3.2770137786865234],[1.3757870197296143,-1.534129023551941,-6.7720184326171875],[2.904165029525757,-1.7527618408203125,-4.8055195808410645],[0.28296399116516113,-1.752761960029602,-5.679624557495117],[2.6858739852905273,1.3057055473327637,-3.2770142555236816],[0.9385210275650024,1.7426282167434692,-5.023811340332031],[1.3757870197296143,-1.3158382177352905,-6.990780830383301],[-0.15396000444889069,-1.3158377408981323,-4.369880199432373],[-0.15396000444889069,-0.44199106097221375,-6.553729057312012],[3.1224560737609863,0.43185701966285706,-6.116206169128418],[1.1571539640426636,-0.4419911205768585,-6.990780830383301],[3.1224560737609863,0.4318576455116272,-2.1846189498901367],[-0.15396000444889069,0.8687809705734253,-6.334968090057373],[0.9385210275650024,-1.3158382177352905,-6.990780830383301],[-0.15396000444889069,1.3057053089141846,-4.587699890136719],[0.9385210275650024,1.7426280975341797,-5.679625511169434],[1.3757870197296143,1.523995041847229,-5.898386478424072],[0.7198879718780518,-0.44199052453041077,-3.2770140171051025],[0.28296399116516113,-1.7527618408203125,-5.023810386657715],[0.28296399116516113,0.4318569004535675,-6.772018909454346],[2.6858739852905273,-0.44199103116989136,-6.334968090057373],[-5.178197860717773,0.6500844359397888,-3.2769289016723633],[-5.178197860717773,0.6500843167304993,-3.9327640533447266],[-5.396317958831787,0.43164336681365967,-3.9327640533447266],[-5.615507125854492,-0.22303654253482819,-3.2769289016723633],[-5.615507125854492,-0.4414765238761902,-3.2769289016723633],[-3.867341995239258,-1.7527614831924438,-3.276928663253784],[-0.15396000444889069,1.3054052591323853,-4.587913990020752],[-0.15396000444889069,-0.005237676203250885,-4.1510329246521],[-0.15396000444889069,0.21320289373397827,-6.771635055541992],[-3.2119131088256836,0.8685244917869568,-3.2769291400909424],[-0.15396000444889069,-1.0967986583709717,-4.1510329246521],[-4.522769927978516,0.43164312839508057,-5.242722988128662],[-2.557554006576538,1.0869650840759277,-5.8978729248046875],[-2.1202449798583984,0.21320292353630066,-6.553366184234619],[-3.649221897125244,-0.8783579468727112,-5.6796040534973145],[-2.3394339084625244,-1.3152389526367188,-6.116485118865967],[-4.741425037384033,-1.5336796045303345,-3.9327638149261475],[-2.1202449798583984,-0.8783575296401978,-3.276928663253784],[-4.304649829864502,1.0869654417037964,-3.4958832263946533],[-1.0280430316925049,1.5241671800613403,-5.24272346496582],[-4.741425037384033,-0.6599177718162537,-4.806526184082031],[-2.3394339084625244,0.43164345622062683,-3.2769289016723633],[-0.15396000444889069,-1.3152387142181396,-4.369645118713379],[-0.8093879818916321,-1.5336799621582031,-5.897872447967529],[-0.3726139962673187,-1.3152389526367188,-5.8978729248046875],[-0.15396000444889069,0.6500838994979858,-6.553366184234619],[-2.1202449798583984,-0.6599180698394775,-6.553366184234619],[-0.15396000444889069,-0.6599180698394775,-6.553366184234619],[-3.649221897125244,1.0869652032852173,-5.024453163146973],[-4.304649829864502,-1.7527616024017334,-3.7141518592834473],[-0.8093879818916321,1.3054050207138062,-5.8978729248046875],[-1.6834709644317627,0.6500838994979858,-6.553366184234619],[-6.270102024078369,-1.3156658411026,1.0923092365264893],[-4.9593729972839355,-1.534213900566101,0.6552152633666992],[-4.305228233337402,-1.097118854522705,1.0923091173171997],[-2.775502920150757,-0.00528053380548954,-3.276927947998047],[-2.775502920150757,-1.0971194505691528,-3.2769277095794678],[-3.868046998977661,-1.7527614831924438,-3.2769277095794678],[-6.052323818206787,-1.5342140197753906,0.2181202471256256],[-5.396553993225098,-1.3156664371490479,-2.8398327827453613],[-5.615143775939941,-0.22352853417396545,-3.276927947998047],[-6.707687854766846,-0.8788708448410034,1.0923091173171997],[-4.086637020111084,-0.8788708448410034,1.0923091173171997],[-4.086637020111084,0.43181517720222473,1.0923088788986206],[-6.926279067993164,0.43181514739990234,0.8737619519233704],[-6.489099025726318,1.087157964706421,-1.7712052624574426E-07],[-6.707687854766846,0.6500627398490906,-1.3104290962219238],[-2.775502920150757,-0.005280427169054747,-2.6208579540252686],[-5.615143775939941,1.0871580839157104,1.0923088788986206],[-4.74240779876709,1.0871580839157104,0.8737618327140808],[-6.489099025726318,-0.4417763948440552,-2.402311086654663],[-6.052323818206787,0.6500625014305115,-2.8398330211639404],[-3.6494569778442383,1.0871574878692627,-2.8398332595825195],[-5.177962779998779,1.3057055473327637,-2.8398332595825195],[-4.9593729972839355,1.3057059049606323,-0.43709519505500793],[-2.775502920150757,-1.0971194505691528,-2.8398327827453613],[-4.9593729972839355,0.6500624418258667,-3.276927947998047],[-6.489099025726318,-1.0971190929412842,-0.6556417942047119],[-2.9940929412841797,0.43181461095809937,-2.402311086654663],[-5.833734035491943,1.3057057857513428,-1.7470961809158325],[-6.926279067993164,-0.22352789342403412,0.6552150845527649],[-3.868046998977661,-1.7527614831924438,-2.6208577156066895],[-5.615143775939941,-0.8788715600967407,-3.2769277095794678],[-6.052323818206787,1.3057060241699219,-2.1272651906656392E-07]]
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
        self.obj1_bid = self.sim.model.body_name2id('Object1')
        self.obj2_bid = self.sim.model.body_name2id('Object2')
        self.obj3_bid = self.sim.model.body_name2id('Object3')
        self.obj4_bid = self.sim.model.body_name2id('Object4')
        self.obj5_bid = self.sim.model.body_name2id('Object5')
        self.obj6_bid = self.sim.model.body_name2id('Object6')
        self.obj7_bid = self.sim.model.body_name2id('Object7')
        self.obj8_bid = self.sim.model.body_name2id('Object8')
        self.obj_bid_list = [self.obj1_bid, self.obj2_bid, self.obj3_bid, self.obj4_bid, self.obj5_bid, self.obj6_bid, self.obj7_bid, self.obj8_bid]

    def custom_init(self, config):
        self.config = config

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
        if 'use_gt' in self.config.keys() and self.config['use_gt']:
            if is_touched and self.previous_contact_points != [] and previous_pos_list != []:
                gt_dist1, gt_dist2 = chamfer_dist(torch.FloatTensor([self.obj3_gt]), torch.FloatTensor([current_pos_list]))
                gt_loss = (torch.mean(gt_dist1)) + (torch.mean(gt_dist2))
                chamfer_reward += self.loss_transform(gt_loss) * 10
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
        
        # palm close to object reward
        basic_reward = self.get_basic_reward(obj_init_xpos, palm_xpos)

        # pos of current obj's contacts
        current_pos_list = []
        is_touched = False
        new_points_cnt = 0
        # contacts of current obj
        for contact in self.data.contact:
            if self.sim.model.geom_id2name(contact.geom1) in self.obj_name[self.obj_bid_idx] or self.sim.model.geom_id2name(contact.geom2) in self.obj_name[self.obj_bid_idx]:#["handle", "neck", "head"]:
                current_pos_list.append(contact.pos.tolist())
                is_touched = True

        # dedup item
        current_pos_list = [item for item in current_pos_list if current_pos_list.count(item) == 1]
        new_pos_list = []
        min_pos_dist = None
        knn_reward = 0
        for pos in current_pos_list:
            # new contact points
            if pos not in self.previous_contact_points:
                min_pos_dist = 1
                for previous_pos in self.previous_contact_points:
                    pos_dist = np.linalg.norm(np.array(pos) - np.array(previous_pos))
                    min_pos_dist = pos_dist if min_pos_dist is None else min(min_pos_dist, pos_dist)
                # new contact points that are not close to already touched points
                if min_pos_dist and min_pos_dist > 0.01: 
                    new_points_cnt += 1  
                    basic_reward += self.get_newpoints_reward(min_pos_dist)
                    new_pos_list.append(pos)
                knn_reward += min_pos_dist

        # similar points penalty
        # penalty_sim = similar_points_cnt * 5
        # penalty_sim = self.get_penalty()
        previous_pos_list = self.previous_contact_points.copy()
        current_pos_list = self.previous_contact_points.copy()
        for item in new_pos_list:
            if item not in current_pos_list:
                current_pos_list.append(item)        

        chamfer_reward = self.get_chamfer_reward(is_touched, previous_pos_list, current_pos_list)
        # knn_reward = self.get_knn_reward()

        if 'coef_' in self.config.keys():
            basic_reward *= self.config['coef_br']
            chamfer_reward *= self.config['coef_cr']
            knn_reward *= self.config['coef_kr']

        reward = basic_reward + chamfer_reward + knn_reward
        done = False
        # for simplicity goal_achieved depends on the nubmer of touched points
        goal_achieved = False
        self.previous_contact_points = current_pos_list.copy()
        # temporarily penalyty use knn reward
        return self.get_obs(), reward, done, dict(goal_achieved=goal_achieved,basic_reward=basic_reward,recon_reward=chamfer_reward,penalty_sim=knn_reward)

    def get_obs(self):
        qp = self.data.qpos.ravel()
        obj_init_xpos = self.data.body_xpos[self.obj_bid_list[self.obj_bid_idx]].ravel()
        palm_xpos = self.data.site_xpos[self.S_grasp_sid].ravel()
        touch_points = len(self.previous_contact_points)
        touch_pos = self.previous_contact_points[touch_points-3:] if touch_points >= 3 else [0,0,0,0,0,0,0,0,0]
        ffknuckle_xpos = self.data.body_xpos[self.ffknuckle_obj_bid].ravel()
        mfknuckle_xpos = self.data.body_xpos[self.mfknuckle_obj_bid].ravel()
        rfknuckle_xpos = self.data.body_xpos[self.rfknuckle_obj_bid].ravel()
        lfmetacarpal_xpos = self.data.body_xpos[self.lfmetacarpal_obj_bid].ravel()
        thbase_xpos = self.data.body_xpos[self.thbase_obj_bid].ravel()

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
