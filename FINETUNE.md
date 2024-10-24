## Finetuning ViTs with low-rank

python -m torch.distributed.launch \
--master_port 12346 --nproc_per_node=1 \
--use_env main_finetune_chestxray.py \
--output_dir /path/to/output_dir \
--log_dir /path/to/output_dir \
--dataset chestxray \
--batch_size 512 --retrain /path/to/base_model \
--epochs 20 --blr 2.5e-5 --layer_decay 0.55 --weight_decay 0 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /path/to/dataset --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-6 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--softRank 38 --train_all --tn_loss_factor 0.00125
