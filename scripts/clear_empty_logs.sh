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


