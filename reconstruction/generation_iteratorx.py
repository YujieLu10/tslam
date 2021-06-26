import os
import trimesh
from data_processing.evaluation import eval_mesh
import traceback
import pickle as pkl
import numpy as np
from tqdm import tqdm
import itertools
import multiprocessing as mp
from multiprocessing import Pool
import torch
import shutil

pose_num_list = ["four", "eight"]

# def gen_iterator(out_path, dataset, gen_p , buff_p, start,end):
def gen_iterator(out_path, dataset, gen_p, vis_type, res, model_type, is_pcloud, pc_samples):
    print(">>> gen_iteratorx {}".format(out_path))
    global gen
    gen = gen_p

    if not os.path.exists(os.path.join(out_path, 'generation')):
        os.makedirs(os.path.join(out_path, 'generation'))
    for pose_num_str in pose_num_list:
        # for iternum in range(499, 10000, 500):
        for iternum in [499]:#, 999, 1499]:
            # export_file_path = os.path.join(out_path, 'generation', 'iternum_{}_{}_{}_{}_pose{}_surface_reconstruction.off'.format(iternum, vis_type, model_type, res, pose_num_str))
            export_file_path = os.path.join(out_path, 'generation', 'uniform_gt_trans_{}_{}_surface_reconstruction.off'.format(model_type, res))
            if os.path.exists(export_file_path):
                # shutil.rmtree(export_file_path)
                print('File Path exists skip! {}'.format(export_file_path))
                continue
                # return
            # if not is_pcloud:
            #     voxel_path = os.path.join(out_path, "iternum_{}_exp_{}pose_trans_voxelization_{}.npy".format(iternum, pose_num_str, res))
            # else:
            #     voxel_path = os.path.join(out_path, "iternum_{}_exp_{}pose_trans_voxelization_{}.npy".format(iternum, pose_num_str, res))
            voxel_path = os.path.join(out_path, "iternum_{}_uniform_gt_{}pose_trans_voxelization_{}.npy".format(iternum, pose_num_str, res))
            if not os.path.exists(voxel_path):
                print('Voxel not exists - skip! {}'.format(voxel_path))
                return

            occupancies = np.load(voxel_path)
            occupancies = np.unpackbits(occupancies)

            input = np.reshape(occupancies, (res,)*3)
            data = {'inputs': torch.tensor([np.array(input, dtype=np.float32)])}
            try:
                data_tupels = []
                logits = gen.generate_mesh(data)
                data_tupels.append((logits, export_file_path))
                create_meshes(data_tupels)
            except Exception as err:
                print('Error with {}: {}'.format(voxel_path, traceback.format_exc()))


def save_mesh(data_tupel):
    logits, export_file_path = data_tupel
    mesh = gen.mesh_from_logits(logits)

    # if not os.path.exists(export_file_path):
    #     os.makedirs(export_file_path)
    print(">>> save mesh {}".format(export_file_path))
    mesh.export(export_file_path)


def create_meshes(data_tupels):
    p = Pool(mp.cpu_count())
    p.map(save_mesh, data_tupels)
    p.close()