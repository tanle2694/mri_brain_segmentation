set -e
ROOT_FOLDER="/home/tanlm/Downloads/covid_data/rotation_data"
EXP_NAME="fold_3"
TRAIN_DATA="/home/tanlm/Downloads/covid_data/kfold/3/train.txt"
VALIDATION_DATA="/home/tanlm/Downloads/covid_data/kfold/3/val.txt"
TRAIN_SAVE_DIR="/home/tanlm/Downloads/save_dir_rotation/save_dir"


python train.py --train_data $TRAIN_DATA \
                   --validation_data $VALIDATION_DATA \
                   --root_folder $ROOT_FOLDER \
                   --exper_name $EXP_NAME \
                   --trainer_save_dir $TRAIN_SAVE_DIR
