docker build --build-arg USER=$USER --build-arg USER_ID=$(id -u)  --build-arg GROUP_ID=$(id -g)  --tag slam/equiv_vo .
