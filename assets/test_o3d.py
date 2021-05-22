import open3d as o3d
import numpy
import os

for dir in os.listdir("assets/meshes/grab"):
    obj_name = dir
    mesh = o3d.io.read_triangle_mesh("assets/meshes/grab/{}/{}.stl".format(obj_name, obj_name))

    pcd = mesh.sample_points_uniformly(number_of_points=2500)
    pcd = mesh.sample_points_poisson_disk(number_of_points=1000, pcl=pcd)
    # o3d.visualization.draw_geometries([pcd])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("assets/uniform_vis/uniform_vis_{}.png".format(obj_name))
    vis.destroy_window()
    numpy.savez_compressed("assets/uniform_gt/uniform_{}_o3d.npz".format(obj_name),pcd=numpy.asarray(pcd.points))