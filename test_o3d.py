import open3d as o3d
import numpy
# mesh = o3d.geometry.TriangleMesh.create_sphere()
# mesh = o3d.io.read_triangle_mesh("mj_envs/mj_envs/envs/hand_manipulation_suite/assets/stl/Simple_Shape.stl")
# mesh = o3d.io.read_triangle_mesh("mj_envs/mj_envs/envs/hand_manipulation_suite/assets/stl/C_Shape_2.stl")
mesh = o3d.io.read_triangle_mesh("mj_envs/mj_envs/envs/hand_manipulation_suite/assets/stl/D_Shape.stl")
# pcd = mesh.sample_points_poisson_disk(number_of_points=1000, init_factor=5)
# o3d.visualization.draw_geometries([pcd])

pcd = mesh.sample_points_uniformly(number_of_points=2500)
pcd = mesh.sample_points_poisson_disk(number_of_points=1000, pcl=pcd)
# o3d.visualization.draw_geometries([pcd])

numpy.savez_compressed("uniform_donut_o3d.npz",pcd=numpy.asarray(pcd.points))