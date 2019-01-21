# put credentials here
export GOOGLE_APPLICATION_CREDENTIALS="credentials.json"
TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.model"
PACKAGE_STAGING_PATH="gs://cbis-ddsm-cnn"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="Brian_Nguyen_$now"
MODEL_NAME='complete_train_inceptionNet_channelsfirst_concat_613'
JOB_DIR="gs://cbis-ddsm-cnn"
REGION="us-central1"
MODE="LOCAL"

gcloud ml-engine local train \
--module-name=$MAIN_TRAINER_MODULE \
--package-path=$TRAINER_PACKAGE_PATH \
--job-dir=$JOB_DIR \
-- \
--mode=$MODE \
--train FALSE \
--model_name=$MODEL_NAME
