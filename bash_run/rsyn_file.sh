set -e

REMOTE_IP="34.87.77.5"
REMOTE_PORT="34375"
REMOTE_TENSORBOARD_PORT="6006"
LOCAL_PORT="6009"
REMOTE_DIR="/root/data/save_dir"
LOCAL_DIR="/home/tanlm/Downloads/lgg-mri-segmentation/remote_save_dir"
TIME_UPDATE=10 #second

#--------------------------------------
echo "ssh -N -f -L ${LOCAL_PORT}:127.0.0.1:${REMOTE_TENSORBOARD_PORT} -p ${REMOTE_PORT} root@${REMOTE_IP}"
sh -c "ssh -N -f -L ${LOCAL_PORT}:127.0.0.1:${REMOTE_TENSORBOARD_PORT} -p ${REMOTE_PORT} root@${REMOTE_IP}" &
echo "rsync -avzhe ssh -p ${REMOTE_PORT} root@${REMOTE_IP}:${REMOTE_DIR} ${LOCAL_DIR}"
while [ true ]
do
  echo "Rsyn from remote server"
  sh -c "rsync -avzhe \"ssh -p ${REMOTE_PORT}\" root@${REMOTE_IP}:${REMOTE_DIR} ${LOCAL_DIR}"
  echo "Sleeping"
  sleep $TIME_UPDATE
done