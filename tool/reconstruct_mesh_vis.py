import open3d as o3d
import os


mesh = o3d.io.read_triangle_mesh("export_pathsurface_reconstruction.off")
o3d.visualization.draw_geometries([mesh])