#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29052 scripts/train.py \
#	--config config/dgx.py:aesthetic \
#	--config.seed=42 \
#	--config.entity_name=diffusion_cornell \
#	--config.project_name=full_aesthetic \
#	--config.run_name=ddpo_aesthetic_42 \
#	--config.sample.num_batches_per_epoch=8 \
#	--config.train.gradient_accumulation_steps=16 \
#	--config.num_epochs=125 \
#	--config.train.batch_size=2
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29052 scripts/train.py \
#	--config config/dgx.py:aesthetic \
#	--config.seed=55513 \
#	--config.entity_name=diffusion_cornell \
#	--config.project_name=full_aesthetic \
#	--config.run_name=ddpo_aesthetic_55513 \
#	--config.sample.num_batches_per_epoch=8 \
#	--config.train.gradient_accumulation_steps=16 \
#	--config.num_epochs=125 \
#	--config.train.batch_size=2

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29052 scripts/train.py \
	--config config/dgx.py:aesthetic \
	--config.seed=66624 \
	--config.entity_name=diffusion_cornell \
	--config.project_name=full_aesthetic \
	--config.run_name=ddpo_aesthetic_66624 \
	--config.sample.num_batches_per_epoch=8 \
	--config.train.gradient_accumulation_steps=16 \
	--config.num_epochs=125 \
	--config.train.batch_size=2
