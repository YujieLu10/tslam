import open3d as o3d

pcd = o3d.io.read_point_cloud("hammer_new_11.ply")
pcd.estimate_normals()

alpha = 0.03
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


export PATH=/home/jianrenw/cuda-11.1/bin:$PATH
export CPATH=/home/jianrenw/cuda-11.1/include:$CPATH
export LD_LIBRARY_PATH=/home/jianrenw/cuda-11.1/lib64:$LD_LIBRARY_PATH