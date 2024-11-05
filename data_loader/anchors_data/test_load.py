import trimesh
import sys
print(trimesh.load(sys.argv[1]).vertices.shape)
