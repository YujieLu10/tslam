from voxels import VoxelGrid
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import glob
import os
import argparse

vis_root = "uniform_gt"
pose_num_str_list = ["four", "eight"]
policy_type_list = ["ours", "disagreef1", "heuristic", "random1", "npoint1", "ntouch1", "knn1", "cf1"]

def create_voxel_off(path):

    for iternum in [199]:
        for pose_num_str in pose_num_str_list:
            voxel_path = path + "iternum_{}_{}_{}pose_trans_voxelization_{}.npy".format(iternum, vis_root, pose_num_str, res)
            off_path = path + "uniform_gt_trans_voxelization_32.off"
            if os.path.exists(off_path):
                print(">>> {} exists! skip".format(off_path))
                continue
            try:
                if unpackbits:
                    occ = np.unpackbits(np.load(voxel_path))
                    voxels = np.reshape(occ, (res,)*3)
                else:
                    voxels = np.reshape(np.load(voxel_path)['occupancies'], (res,)*3)

                loc = ((min+max)/2, )*3
                scale = max - min
                if os.path.exists(off_path):
                    print(">>> off_path {} exists".format(off_path))
                    continue
                VoxelGrid(voxels, loc, scale).to_mesh().export(off_path)
                print('Finished: {}'.format(path))
            except:
                print('Voxel not exist: {}'.format(voxel_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxalization to off'
    )
    parser.add_argument('-res', type=int)

    args = parser.parse_args()

    # ROOT = '/home/yourpathname/prox/tslam/data/result/agent_eval_standard_voxel/exp'
    ROOT = '/home/yourpathname/prox/tslam/data/result/agent_eval_standard_voxel/uniform_gt'
    # ROOT = '/home/yourpathname/ifnet/objs/grab'

    unpackbits = True
    res = args.res
    min = -0.5
    max = 0.5

    p = Pool(mp.cpu_count())
    p.map(create_voxel_off, glob.glob(ROOT + '/*/'))

# trajectory prepare
# from voxels import VoxelGrid
# import numpy as np
# import multiprocessing as mp
# from multiprocessing import Pool
# import glob
# import os
# import argparse

# def create_voxel_off(path):

#     for file in os.listdir(path):
#         voxel_path = os.path.join(path, file)
#         off_path = os.path.join("/home/yourpathname/prox/tslam/data/result/glass_trajectory/voxel_off", "traj_{}.off".format((int(file.replace("eight","pointcloud")[file.index("_")+1:file.replace("eight","pointcloud").index("_pointcloud")]) + 1) / int(100)))

#         if unpackbits:
#             occ = np.unpackbits(np.load(voxel_path))
#             voxels = np.reshape(occ, (res,)*3)
#         else:
#             voxels = np.reshape(np.load(voxel_path)['occupancies'], (res,)*3)

#         loc = ((min+max)/2, )*3
#         scale = max - min
#         if os.path.exists(off_path):
#             print(">>> off_path {} exists".format(off_path))
#             continue
#         VoxelGrid(voxels, loc, scale).to_mesh().export(off_path)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Run voxalization to off'
#     )
#     parser.add_argument('-res', type=int)

#     args = parser.parse_args()

#     ROOT = '/home/yourpathname/prox/tslam/data/result/glass_trajectory/pcd_raw'
#     # ROOT = '/home/yourpathname/ifnet/objs/grab'

#     unpackbits = True
#     res = args.res
#     min = -0.5
#     max = 0.5

#     p = Pool(mp.cpu_count())
#     p.map(create_voxel_off, glob.glob(ROOT))

