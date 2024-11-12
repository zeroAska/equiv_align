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
for i in $(ls log)
do
   echo checking $i
   if [ ! -d "log/${i}/checkpoints" ]; then
      echo "will rm log/${i}/checkpoints"
      rm -rf log/${i}
   elif [[ ! -f "log/${i}/checkpoints/best_model.pth" ]] && [[  ! -f "log/${i}/checkpoints/0_model.pth"  ]]; then
      #echo "will rm log/${i}/checkpoints"
      rm -rf log/${i}
   else
	   echo "log/${i} survives"
   fi 
    
done


