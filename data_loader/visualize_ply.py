import open3d as o3d
import numpy as np
import sys, os


if __name__ == "__main__":
    print("Load a ply point cloud, print it, and render it")
    ply_point_cloud = o3d.data.PLYPointCloud()

    pcds = []
    for i in range(1, len(sys.argv)):
        pcd = o3d.io.read_point_cloud(sys.argv[i])
        pcds.append(pcd)
        print("loaded pcd ", pcd)
        print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries(pcds)
