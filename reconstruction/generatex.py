import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
import os
import numpy as np
import argparse
from models.generation import Generator
from generation_iteratorx import gen_iterator

parser = argparse.ArgumentParser(
    description='Run generation'
)


parser.add_argument('-pointcloud', dest='pointcloud', action='store_true')
parser.add_argument('-voxels', dest='pointcloud', action='store_false')
parser.set_defaults(pointcloud=False)
parser.add_argument('-pc_samples' , default=3000, type=int)
parser.add_argument('-dist','--sample_distribution', default=[0.5,0.5], nargs='+', type=float)
parser.add_argument('-std_dev','--sample_sigmas',default=[], nargs='+', type=float)
parser.add_argument('-res' , default=32, type=int)
parser.add_argument('-decoder_hidden_dim' , default=256, type=int)
parser.add_argument('-mode' , default='test', type=str)
parser.add_argument('-retrieval_res' , default=256, type=int)
parser.add_argument('-checkpoint', type=int)
parser.add_argument('-batch_points', default=1000000, type=int)
parser.add_argument('-m','--model' , default='LocNet', type=str)
parser.add_argument('-path' , default='agent_eval_standard_voxel/uniform_gt', type=str)
parser.add_argument('-vis_type' , default='gt_trans', type=str)
parser.add_argument('-policy_type' , default='voxel1', type=str)
parser.add_argument('-combine' , default=2, type=int)

args = parser.parse_args()

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]


if args.model ==  'ShapeNet32Vox':
    net = model.ShapeNet32Vox()

if args.model ==  'ShapeNet128Vox':
    net = model.ShapeNet128Vox()

if args.model == 'ShapeNet300Points':
    net = model.ShapeNetPoints()

if args.model == 'ShapeNet3000Points':
    net = model.ShapeNetPoints()
if args.model == 'SVR':
    net = model.SVR()

test_obj_list = ["body", "airplane"]# "cup", "lightbulb"] #, "spherelarge", "body", "fryingpan"]

dataset = voxelized_data.VoxelizedDataset(args.mode, voxelized_pointcloud= args.pointcloud , pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                          sample_sigmas=args.sample_sigmas ,num_sample_points=100, batch_size=1, num_workers=0)


# exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format(  'PC' + str(args.pc_samples) if args.pointcloud else 'Voxels',
#                                     ''.join(str(e)+'_' for e in args.sample_distribution),
#                                        ''.join(str(e) +'_'for e in args.sample_sigmas),
#                                                                 args.res,args.model)

exp_name = '{}'.format(args.model)


gen = Generator(net,0.5, exp_name, checkpoint=args.checkpoint ,resolution=args.retrieval_res, batch_points=args.batch_points)

# out_path = 'experiments/{}/evaluation_{}_@{}/'.format(exp_name,args.checkpoint, args.retrieval_res)
# out_path = ''
for root, dirs, files in os.walk("../prox/tslam/data/result/{}".format(args.path)):
    is_obj_test = True
    # for obj_test in test_obj_list:
    #     if obj_test in root: is_obj_test = True
    is_combine_type = True if (args.combine == 2 and "10kpose" in root) or (args.combine == 0 and "pose" not in root) or args.vis_type == "gt_trans" else False
    if is_obj_test and is_combine_type and not "generation" in root:# and args.policy_type in root:
        out_path = root
        gen_iterator(out_path, dataset, gen, args.vis_type, args.res, args.model, args.pointcloud, args.pc_samples)
# out_path = 'standard_voxel/uniform_gt/airplane/geneTrue_rotTrue_down_normal_cf0knn0voxel1'
