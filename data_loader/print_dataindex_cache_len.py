# This file is part of EquivAlign.
# 
# Copyright [2024] [Authors of Paper: Correspondence-free SE(3) point cloud registration in RKHS via unsupervised equivariant learning]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Author Email: <Ray Zhang rzh@umich.edu>
import pickle
import sys
import ipdb

with open(sys.argv[1], 'rb') as f:
    a = pickle.load(f)
    print(len(a[0]))
    ipdb.set_trace()
    print("First elem is ", a[0][a[0]])

