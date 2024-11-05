import pickle
import sys

scene_info = pickle.load(open(sys.argv[1], 'rb'))[0] 
print("scene_info len is ",len(scene_info))
dataindex_info = pickle.load(open(sys.argv[2], 'rb'))[0] 
print("dataindex_info len is ",len(dataindex_info))
