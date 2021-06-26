import trimesh
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import glob
import os
import argparse

pose_num_str_list = ["four", "eight"]
policy_type_list = ["ours", "disagreef1", "heuristic", "random1", "npoint1", "ntouch1", "knn1", "cf1"]

def create_voxel_off(path):

    for iternum in [199]:
        for pose_num_str in pose_num_str_list:
            pc_path = path + 'iternum_{}_{}pose_alpha_pointcloud.npz'.format(iternum, pose_num_str)
            off_path = path + 'iternum_{}_{}pose_alpha_pointcloud.off'.format(iternum, pose_num_str)

            if not os.path.exists(pc_path):
                print(" {} not exist".format(pc_path))
                continue
            pc = np.load(pc_path)['pcd']


            trimesh.Trimesh(vertices = pc , faces = []).export(off_path)
            print('Finished: {}'.format(off_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create off visualization from point cloud.'
    )

    parser.add_argument('-res', type=int)
    parser.add_argument('-num_points', type=int)

    args = parser.parse_args()

    ROOT = '/home/yourpathname/prox/tslam/data/result/agent_eval_standard_voxel/exp'

    p = Pool(mp.cpu_count())
    p.map(create_voxel_off, glob.glob(ROOT + '/*/*/'))

# import trimesh
# import numpy as np
# import multiprocessing as mp
# from multiprocessing import Pool
# import glob
# import os
# import argparse


# def create_voxel_off(path):
#     # pc_path = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(args.res, args.num_points)
#     # off_path = path + '/voxelized_point_cloud_{}res_{}points.off'.format(args.res, args.num_points)

#     # pc = np.load(pc_path)['point_cloud']
#     print(path)
#     for file in os.listdir(path):
#         pc_path = os.path.join(path, file)
#         off_path = os.path.join("/home/yourpathname/prox/tslam/data/result/glass_trajectory/pc_off", "traj_{}.off".format((int(file.replace("eight","pointcloud")[file.index("_")+1:file.replace("eight","pointcloud").index("_pointcloud")]) + 1) / int(100)))
#         pc = np.load(pc_path)['pcd']


#         trimesh.Trimesh(vertices = pc , faces = []).export(off_path)
#         print('Finished: {}'.format(off_path))



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Create off visualization from point cloud.'
#     )

#     parser.add_argument('-res', type=int)
#     parser.add_argument('-num_points', type=int)

#     args = parser.parse_args()

#     ROOT = '/home/yourpathname/prox/tslam/data/result/glass_trajectory/iter_pcdnpz'

#     p = Pool(mp.cpu_count())
#     p.map(create_voxel_off, glob.glob(ROOT))