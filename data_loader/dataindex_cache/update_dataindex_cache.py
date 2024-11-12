import pickle
import sys, os
import shutil
import ipdb
from collections import OrderedDict

def group_dataindex_cache(source, target=None):
    with open(source, "rb") as f:
        dataindex_cache = pickle.load(f)

        grouped_data = OrderedDict()

        for tup in dataindex_cache[0]:
            ipdb.set_trace()
            path = tup[0]
            pair = tup[1]
            if path not in grouped_data:
                grouped_data[path] = []
            else:
                grouped_data[path].append((path, pair))
        
        data = (grouped_data,)
        if target is not None:
            with open(target, 'wb') as fw:
                # Pickle the object and write it to the file
                pickle.dump(data, fw)


def regen_dataindex_cache(source, target=None):
    with open(source, "rb") as f:
        dataindex_cache = pickle.load(f)
        data_list = []
        for tup in dataindex_cache[0]:
            path = tup[0]
            pair = tup[1]
            path_list = path.split("/")
            path_new = "/".join(path_list[3:])
            print("change {} to {}".format(path, path_new))
            tup_new = (path_new, pair)
            data_list.append(tup_new)
        data = (data_list,)
        if target == None:
            return
        with open(target, 'wb') as fw:
            # Pickle the object and write it to the file
            pickle.dump(data, fw)

        with open(target, 'rb') as fr:
            new_index_cache = pickle.load(fr)
            print("first new element is ",)
            print(new_index_cache[0][0])



if __name__ == "__main__":
    #old_dir = "../index_cache/eth3d/"
    for i in os.listdir("."):
        if not i.endswith("pickle"):
            continue
        old_fname = i +".backup" #old_dir + "/" + i
        os.rename(i, old_fname)
        new_fname = i #old_fname[:-7]
        assert(old_fname != new_fname)
        print("process ",old_fname," to " ,new_fname)
        
        #group_dataindex_cache(old_fname, new_fname)
        regen_dataindex_cache()
        
