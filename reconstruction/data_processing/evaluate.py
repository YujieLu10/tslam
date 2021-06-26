from evaluation import eval_mesh, eval_pointcloud
import trimesh
import pickle as pkl
import os
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import argparse
from glob import glob
import traceback
import random
from voxels import VoxelGrid
import numpy as np

pose_num_list = ["four", "eight"]
iter_num = 199
def eval(path):
    if not "heuristic" in path:
        return
    for pose_num_str in pose_num_list:
        if args.reconst:
            eval_file_name = "/eval_pose{}.pkl".format(pose_num_str)
        elif args.voxels:
            eval_file_name = "/eval_voxelization_{}_pose{}.pkl".format(args.res, pose_num_str)
        else:
            eval_file_name = "/eval_pointcloud_{}_pose{}.pkl".format(args.points, pose_num_str)

        try:
            if os.path.exists(path + eval_file_name):
                # os.remove(path + eval_file_name)
                print('File exists. Done.')
                return
            else:
                path = os.path.normpath(path)
                folder = path.split(os.sep)[-3]
                file_name = path.split(os.sep)[-1]

                if args.reconst:
                    # pred_mesh_path = path + '/surface_reconstruction.off'
                    pred_mesh_path = path + '/iternum_{}_exp_trans_{}_{}_pose{}_surface_reconstruction.off'.format(iter_num, args.model, args.res, pose_num_str)
                    pred_mesh = trimesh.load(pred_mesh_path, process=False)

                    # gt_mesh_path = data_path + '/{}/{}/isosurf_scaled.off'.format(folder, file_name)
                    gt_mesh_path = data_path + '/{}/{}/uniform_gt_trans_{}_{}_surface_reconstruction.off'.format(folder, file_name, "ShapeNet32Vox", args.res)
                    gt_mesh = trimesh.load(gt_mesh_path, process=False)

                    eval = eval_mesh(pred_mesh, gt_mesh, min, max)

                elif args.voxels:
                    voxel_path = path + '/iternum_{}_exp_{}pose_trans_voxelization_{}.npy'.format(iter_num, pose_num_str, args.res)
                    occ = np.unpackbits(np.load(voxel_path))
                    voxels = np.reshape(occ, (args.res,) * 3)

                    off_path = path + '/iternum_{}_exp_{}pose_trans_voxelization_{}.off'.format(iter_num, pose_num_str, args.res)
                    input_mesh = VoxelGrid(voxels, [0,0,0], 1).to_mesh()
                    input_mesh.export(off_path)

                    gt_mesh_path = data_path + '/{}/generation/uniform_gt_trans_{}_{}_surface_reconstruction.off'.format(path.split(os.sep)[-2], "ShapeNet32Vox", args.res)
                    gt_mesh = trimesh.load(gt_mesh_path, process=False)

                    eval = eval_mesh(input_mesh, gt_mesh, min, max)

                else:
                    input_points_path = path + '/voxelized_point_cloud_128res_{}points.npz'.format(args.points)
                    input_points = np.load(input_points_path)['point_cloud'].astype(np.float32)
                    gt_mesh_path = data_path + '/{}/{}/isosurf_scaled.off'.format(folder, file_name)

                    gt_mesh = trimesh.load(gt_mesh_path, process=False)
                    pointcloud_gt, idx = gt_mesh.sample(100000, return_index=True)
                    pointcloud_gt = pointcloud_gt.astype(np.float32)

                    eval = eval_pointcloud(input_points, pointcloud_gt)

                pkl.dump( eval ,open(path + eval_file_name, 'wb'))
                print('Finished {}'.format(path))

        except Exception as err:

            print('Error with {}: {}'.format(path, traceback.format_exc()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run input evaluation'
    )

    parser.add_argument('-model', default='ShapeNet3000Points', type=str) # ShapeNet32Vox | ShapeNet3000Points
    parser.add_argument('-voxels', dest='voxels', action='store_true')
    parser.add_argument('-pc', dest='voxels', action='store_false')
    parser.add_argument('-res', default=32, type=int)
    parser.add_argument('-points',type=int)
    parser.set_defaults(voxels=True)
    parser.add_argument('-reconst', action='store_true')
    parser.set_defaults(reconst=False)
    parser.add_argument('-generation_path', type=str) # *500fixpose*voxel1 *fixpose*voxel1 *fixpose*random1


    args = parser.parse_args()

    # uniform_gt if-net reconstruction
    data_path = '../prox/tslam/data/result/agent_eval_standard_voxel/uniform_gt'

    min = -0.5
    max = 0.5

    p = Pool(mp.cpu_count())
    if args.reconst:
        paths = glob(args.generation_path + '/*/*/*/')
    elif args.voxels:
        paths = glob(args.generation_path + '/*/*/')
    else:
        paths = glob(data_path + '/*/*/*/')

    # enabeling to run te script multiple times in parallel: shuffling the data
    random.shuffle(paths)
    p.map(eval, paths)
    p.close()
    p.join()
