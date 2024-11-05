import open3d as o3d
import sys
import numpy as np

pcd_path = sys.argv[1]
pcd = o3d.io.read_point_cloud(pcd_path)
print(pcd)
print(np.asarray(pcd.points))
#mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd, mesh_frame])
