#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 29058 scripts/train.py \
	--config config/dgx.py:incompressibility \
	--config.seed=42 \
	--config.entity_name=diffusion_cornell \
	--config.project_name=incompression_final \
	--config.run_name=ddpo_incompression_42 \
	--config.sample.num_batches_per_epoch=8 \
	--config.train.gradient_accumulation_steps=8 \
	--config.train.batch_size=2

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 29058 scripts/train.py \
	--config config/dgx.py:incompressibility \
	--config.seed=55513 \
	--config.entity_name=diffusion_cornell \
	--config.project_name=incompression_final \
	--config.run_name=ddpo_incompression_55513 \
	--config.sample.num_batches_per_epoch=8 \
	--config.train.gradient_accumulation_steps=8 \
	--config.train.batch_size=2

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 29058 scripts/train.py \
	--config config/dgx.py:incompressibility \
	--config.seed=66624 \
	--config.entity_name=diffusion_cornell \
	--config.project_name=incompression_final \
	--config.run_name=ddpo_incompression_66624 \
	--config.sample.num_batches_per_epoch=8 \
	--config.train.gradient_accumulation_steps=8 \
	--config.train.batch_size=2
