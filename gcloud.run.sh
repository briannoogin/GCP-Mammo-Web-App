export GOOGLE_APPLICATION_CREDENTIALS="CBIS DDSM CNN-ed9e66db905b.json"
TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.model"
PACKAGE_STAGING_PATH="gs://cbis-ddsm-cnn"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="Brian_Nguyen_$now"
JOB_DIR="gs://cbis-ddsm-cnn"
REGION="us-central1"

GCP_FLOW='trainer/gcp_flow.py'

#gcloud ml-engine local train \
gcloud ml-engine jobs submit training $JOB_NAME \
--module-name=$MAIN_TRAINER_MODULE \
--package-path=$TRAINER_PACKAGE_PATH \
--job-dir=$JOB_DIR \
--region=$REGION \
--python-version 3.5 \
--config config.yaml \
--runtime-version 1.12 \
#--packages=$GCP_FLOW \