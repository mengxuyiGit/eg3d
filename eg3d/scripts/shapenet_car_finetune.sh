# Train with Shapenet finetune, using 1 GPUs.
##-------- common settings ---------------------
CUDA_VISIBLE_DEVICES=1
GPUS=2
BATCH_SIZE=4
BASE_DIR=/home/xuyi/Repo/eg3d
# BASE_DIR=/xuyi-fast-vol/Repo-fast/eg3d

# ##-------- abo/shapenet with triplane -----------
# # DATA=${BASE_DIR}/dataset_preprocessing/shapenet_cars/cars_128_copy.zip
# DATA=${BASE_DIR}/dataset_preprocessing/abo/abo_128_copy.zip
# PRETRAINED_MODEL=${BASE_DIR}/eg3d/pretrained_models/shapenetcars128-64.pkl

# python train.py --outdir=${BASE_DIR}/try-runs --cfg=shapenet --data=${DATA} \
#   --resume=${PRETRAINED_MODEL} \
#   --gpus=${GPUS} --batch=${BATCH_SIZE} --gamma=0.3

# ##---------abo with 3D volume -----------
# DATA=${BASE_DIR}/dataset_preprocessing/abo/abo_128_copy.zip
# # DATA=${BASE_DIR}/dataset_preprocessing/shapenet_cars/cars_128_copy.zip
# PRETRAINED_MODEL=${BASE_DIR}/eg3d/pretrained_models/shapenetcars128-64.pkl

# python train.py --outdir=${BASE_DIR}/try-runs --cfg=abo_dataset --data=${DATA} \
#   --resume=${PRETRAINED_MODEL} \
#   --gpus=${GPUS} --batch=${BATCH_SIZE} --gamma=0.3 \
#   --backbone volume


##---------abo with 3D volume + no pretraining (because feature channel is down to 8)-----------
# DATA=${BASE_DIR}/dataset_preprocessing/shapenet_cars/cars_128_copy.zip
# DATA=${BASE_DIR}/dataset_preprocessing/abo/abo_128_copy.zip
# DATA=${BASE_DIR}/dataset_preprocessing/abo/abo_128_completed.zip
# DATA=${BASE_DIR}/dataset_preprocessing/abo/abo_128_completed_white.zip
# DATA=${BASE_DIR}/dataset_preprocessing/abo/abo_512_completed_white.zip
# DATA=${BASE_DIR}/dataset_preprocessing/abo/abo_512_completed_white_small.zip
DATA=${BASE_DIR}/dataset_preprocessing/abo/debug_one_obj_one_view.zip
CUDA_VISIBLE_DEVICES=[1]
GPUS=1
BATCH_SIZE=1

DROP=True
L1=False
# always not use patch

python train.py --outdir=${BASE_DIR}/try-runs --cfg=abo_dataset --data=${DATA} \
  --gpus=${GPUS} --batch=${BATCH_SIZE} --gamma=0.3 \
  --backbone volume --decoder_dim 8 \
  --noise_strength 1 --snap 1 \
  --use_patch False --z_from_pc False \
  --use_l1 $L1 --l1_reg 100 \
  --discriminator_condition_on_real $DROP --drop_pixel_ratio 0.8
  


  # --use_perception True --perception_reg 1 \
  # --use_l1 True \
  # --use_chamfer True \