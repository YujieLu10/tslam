import open3d as o3d
import numpy as np
import os

for dir in os.listdir("result/ply"):
    obj_name = dir
    if "DS_Store" in obj_name:
        continue
    
    np_pcd = np.load("result/pcloud/{}/{}.npz".format(obj_name, obj_name))['pcd']

    with open("result/ply/{}/{}.ply".format(obj_name, obj_name), "w") as f:
        header_line = "ply\nformat ascii 1.0\ncomment PCL generated\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".format(len(np_pcd))
        content_line = ""
        for vertex in np_pcd:
            content_line += str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + " 255 0 0\n"
        f.write(header_line + content_line)

    
for dir in os.listdir("result/ply"):
    obj_name = dir
    if "DS_Store" in obj_name:
        continue
    pcd = o3d.io.read_point_cloud("result/ply/{}/{}.ply".format(obj_name, obj_name))
    pcd.estimate_normals()

    # pcd = np.load("result/pcloud/{}/{}.npz".format(obj_name, obj_name))['pcd']
    # print(pcd)
    o3d.visualization.draw_geometries([pcd])
    # pcd = mesh.sample_points_poisson_disk(750)
    # o3d.visualization.draw_geometries([pcd])
    alpha = 0.03
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)