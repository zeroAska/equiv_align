import pickle
import sys, os
import copy
import ipdb

def regen_scene_cache_img(source, target=None):
    with open(source, "rb") as f:
        scene_cache = pickle.load(f)
        data_dict = {}

        for old_key, old_key_seq in scene_cache[0].items():
            
            key_old_list = old_key.split("/")
            key_new = "/".join(key_old_list[3:])
            print("change {} to {}".format(key_old_list, key_new))
            
            key_seq = copy.deepcopy(old_key_seq)
            images_old = old_key_seq['images']
            images_new = []
            for image in images_old:
                image_old_list = image.split("/")
                image_new = "/".join(image_old_list[3:])
                images_new.append(image_new)
            key_seq['images'] = images_new
            data_dict[key_new] = key_seq
                
        data = (data_dict,)

        if target == None:
            return

        with open(target, 'wb') as fw:
            # Pickle the object and write it to the file
            pickle.dump(data, fw)

        with open(target, 'rb') as fr:
            new_index_cache = pickle.load(fr)
            print("first new element is ",)
            print(new_index_cache[0]['data/eth3d/training/table_3'])

def regen_scene_cache_depth(source, target=None):
    print("source: ", source)
    with open(source, "rb") as f:
        scene_cache = pickle.load(f)
        data_dict = {}

        for old_key, old_key_seq in scene_cache[0].items():
            
            #key_old_list = old_key.split("/")
            #key_new = "/".join(key_old_list[3:])
            #print("change {} to {}".format(key_old_list, key_new))
            
            key_seq = copy.deepcopy(old_key_seq)
            depths_old = old_key_seq['depths']
            depths_new = []
            for depth in depths_old:
                depth_old_list = depth.split("/")
                depth_new = "/".join(depth_old_list[3:])
                depths_new.append(depth_new)
            key_seq['depths'] = depths_new
            data_dict[old_key] = key_seq
                
        data = (data_dict,)

        if target == None:
            return

        with open(target, 'wb') as fw:
            # Pickle the object and write it to the file
            pickle.dump(data, fw)

        with open(target, 'rb') as fr:
            new_index_cache = pickle.load(fr)
            for key, key_seq in new_index_cache[0].items():
                print("key is ", key)
                print(new_index_cache[0][key]['images'][0])
                print(new_index_cache[0][key]['depths'][0])



if __name__ == "__main__":
    old_dir = "../index_cache/eth3d.scene.backup/"
    for i in os.listdir(old_dir):
        old_fname = old_dir + "/" + i
        new_fname = i
        regen_scene_cache_depth(old_fname, new_fname)
        #regen_dataindex_cache()
        
