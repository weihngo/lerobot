# lerobot-train \
#     --dataset.repo_id=/mnt/data/yzh/dataset/lekiwi_pick_and_put_banana \
#     --policy.type=pi05 \
#     --output_dir=./outputs/pi05_training \
#     --job_name=pi05_training \
#     --policy.repo_id=lwh \
#     --policy.pretrained_path=lerobot/pi05_base \
#     --policy.compile_model=false \
#     --policy.gradient_checkpointing=true \
#     --wandb.enable=false \
#     --policy.dtype=bfloat16 \
#     --policy.freeze_vision_encoder=false \
#     --policy.train_expert_only=false \
#     --steps=30000 \
#     --save_freq=1000
#     --policy.device=cuda \
#     --batch_size=32



lerobot-train --config_path=outputs/pi05_training/checkpoints/025000/pretrained_model/train_config.json --resume=true