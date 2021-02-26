import open3d as o3d

pcd = o3d.io.read_point_cloud("ball_20.ply")
pcd.estimate_normals()

alpha = 0.03
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)