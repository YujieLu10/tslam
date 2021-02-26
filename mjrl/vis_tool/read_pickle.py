import pickle
import mjrl.policies

with open('/Users/yujie/Desktop/project/mj_envs/hand_dapg/dapg/demonstrations/relocate-v0_demos.pickle', 'rb') as f:
    x = pickle.load(f)
    with open('output_relocate.txt', 'wb') as outputf:
        pickle.dump(x, outputf)
    print(x)
