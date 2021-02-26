import open3d as o3d

mesh = o3d.io.read_triangle_mesh("D_shape.stl")
pointcloud = mesh.sample_points_poisson_disk(1000)

print(pointcloud)
# # you can plot and check
# o3d.visualization.draw_geometries([mesh])
# o3d.visualization.draw_geometries([pointcloud])
