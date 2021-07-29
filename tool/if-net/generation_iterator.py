import os
import trimesh
from data_processing.evaluation import eval_mesh
import traceback
import pickle as pkl
import numpy as np
from tqdm import tqdm
import itertools
import multiprocessing as mp
from multiprocessing import Pool, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import torch

# def gen_iterator(out_path, dataset, gen_p , buff_p, start,end):
def gen_iterator(out_path, dataset, gen_p):
    print(">>> gen_iterator {}".format(out_path))
    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)

    voxel_path = out_path + '/uniform_gt_voxelization_{}.npy'.format(32)
    occupancies = np.load(voxel_path)
    input = np.reshape(occupancies, (32,)*3)
    data = {'inputs': torch.tensor([np.array(input, dtype=np.float32)])}
    try:
        data_tupels = []
        logits = gen.generate_mesh(data)
        data_tupels.append((logits, data, out_path))
        create_meshes(data_tupels)
    except Exception as err:
        print('Error with {}: {}'.format(voxel_path, traceback.format_exc()))


def save_mesh(data_tupel):
    logits, data, out_path = data_tupel

    mesh = gen.mesh_from_logits(logits)
    export_path = out_path + '/generation'

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    mesh.export(export_path + '/surface_reconstruction.off')

def create_meshes(data_tupels):
    p = Pool(mp.cpu_count())
    p.map(save_mesh, data_tupels)
    p.close()