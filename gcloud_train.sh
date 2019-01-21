
TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.model"
PACKAGE_STAGING_PATH="gs://cbis-ddsm-cnn"

now=$(date +"%m%d%Y_%H%M%S")
JOB_NAME="train_inceptionNet_channelsfirst_concat_$now"
MODEL_NAME=$JOB_NAME.h5
JOB_DIR="gs://cbis-ddsm-cnn"
REGION="us-central1"
MODE='CLOUD'

gcloud ml-engine jobs submit training $JOB_NAME \
--module-name=$MAIN_TRAINER_MODULE \
--package-path=$TRAINER_PACKAGE_PATH \
--job-dir=$JOB_DIR \
--region=$REGION \
--python-version 3.5 \
--config config.yaml \
--runtime-version 1.12 \
-- \
--mode=$MODE \
--train TRUE \
--model_name=$MODEL_NAME