#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29051 --num_processes 4 scripts/train.py \
	--config config/dgx.py:incompressibility \
	--config.seed=42 \
	--config.entity_name=diffusion_cornell \
	--config.project_name=incompression_final \
	--config.run_name=ddpo_incompression_42 \
	--config.sample.num_batches_per_epoch=8 \
	--config.train.gradient_accumulation_steps=4

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29051 --num_processes 4 scripts/train.py \
	--config config/dgx.py:incompressibility \
	--config.seed=55513 \
	--config.entity_name=diffusion_cornell \
	--config.project_name=incompression_final \
	--config.run_name=ddpo_incompression_55513 \
	--config.sample.num_batches_per_epoch=8 \
	--config.train.gradient_accumulation_steps=4

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29051 --num_processes 4 scripts/train.py \
	--config config/dgx.py:incompressibility \
	--config.seed=66624 \
	--config.entity_name=diffusion_cornell \
	--config.project_name=incompression_final \
	--config.run_name=ddpo_incompression_66624 \
	--config.sample.num_batches_per_epoch=8 \
	--config.train.gradient_accumulation_steps=4
