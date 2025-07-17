export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# 480*832像素, size选632
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --main_process_port=12340 --num_processes=8 --use_deepspeed \
  --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard \
  train.py --config="config/train_wan_a2v.yaml" 