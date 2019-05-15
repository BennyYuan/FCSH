#!/bin/bash
set -e
# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/yun/work/data/checkpoints

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=vgg_16

# The hash code length.
hash_code_len=48

# The dataset name.
dataset_name=cifar10

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/yun/work/data/${dataset_name}-models/${hash_code_len}/${MODEL_NAME}

# Where the dataset is saved to.
DATASET_DIR=/home/yun/work/data/${dataset_name}


# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt ]; then
  wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
  tar -xvf vgg_16_2016_08_28.tar.gz
  mv vgg_16.ckpt ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt
  rm vgg_16_2016_08_28.tar.gz
fi

cd  ..


## Fine-tune only the new layers for 1000 steps.
#CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
#  --train_dir=${TRAIN_DIR} \
#  --dataset_name=${dataset_name} \
#  --dataset_split_name=train \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=${MODEL_NAME} \
#  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt \
#  --checkpoint_exclude_scopes=vgg_16/hash_layer,vgg_16/fc8 \
#  --trainable_scopes=vgg_16/hash_layer,vgg_16/fc8 \
#  --hash_code_num=${hash_code_len} \
#  --max_number_of_steps=5000 \
#  --batch_size=64 \
#  --learning_rate=0.01 \
#  --learning_rate_decay_type=fixed \
#  --save_interval_secs=60 \
#  --save_summaries_secs=60 \
#  --log_every_n_steps=10 \
#  --optimizer=rmsprop \
#  --weight_decay=0.00004
#
## Run evaluation.
#CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR} \
#  --eval_dir=${TRAIN_DIR} \
#  --dataset_name=${dataset_name} \
#  --dataset_split_name=test \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=${MODEL_NAME} \
#  --hash_code_num=${hash_code_len} \
#  --batch_size=120 # 根据显存大小设计


# Fine-tune all the new layers for 500 steps.
CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=${dataset_name} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${TRAIN_DIR} \
  --hash_code_num=${hash_code_len} \
  --max_number_of_steps=100000 \
  --batch_size=64 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=${dataset_name} \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --hash_code_num=${hash_code_len} \
  --batch_size=120

##################
## Produce code ##
##################

# Create database hashcode.
CUDA_VISIBLE_DEVICES=0 python hash_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=${dataset_name} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --hash_code_num=${hash_code_len} \
  --batch_size=200

# Create test hashcode.
CUDA_VISIBLE_DEVICES=0 python hash_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=${dataset_name} \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --hash_code_num=${hash_code_len} \
  --batch_size=200

##################
##  Eval model  ##
##################

# eval mAp.
python mAP.py \
  --dataset_dir=${DATASET_DIR} \
  --hash_bit=${hash_code_len} \
