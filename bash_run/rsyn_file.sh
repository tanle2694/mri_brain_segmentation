#!/bin/bash
set -e
REMOTE_IP="38499"
REPORT_PORT=""
REMOTE_DIR=""
LOCAL_DIR=""

while [ true ]
do
  echo "Rsyn from remote server"
  rsync -avzhe "ssh -p 38499" root@34.87.77.5:/root/Project/pytorch-deeplab-xception/run /data2/tanlm/Project/pytorch-deeplab-xception/update
  echo "Sleeping"
  sleep 10
done