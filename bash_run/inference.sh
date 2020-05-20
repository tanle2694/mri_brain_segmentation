set -e

VALIDATION_DATA="/home/tanlm/Downloads/covid_data/kfold/3/val.txt"
ROOT_FOLDER="/home/tanlm/Downloads/covid_data/data"
DEVICE="cuda:1"
PREDICT_RESULT="/home/tanlm/Downloads/covid_data/data/predict_fold3.txt"
RESUME="/home/tanlm/Downloads/covid_data/save_dir/models/fold_3/0513_203829/model_best.pth"


python inference_images.py --validation_data $VALIDATION_DATA \
                           --root_folder $ROOT_FOLDER \
                           --device $DEVICE \
                           --predict_result $PREDICT_RESULT \
                           --resume $RESUME
