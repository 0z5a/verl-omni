#!/usr/bin/env bash
set -euo pipefail
: "${WORKSPACE:?Set WORKSPACE to the prepared OCR workspace}"
: "${MODEL_PATH:?Set MODEL_PATH to pinned Qwen-Image weights}"
: "${REWARD_MODEL_PATH:?Set REWARD_MODEL_PATH to pinned Qwen3-VL-8B-Instruct weights}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export CC=${CC:-gcc} CXX=${CXX:-g++} CUDAHOSTCXX=${CUDAHOSTCXX:-g++}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NUM_GPUS_ACTOR_ROLLOUT_REWARD=2 ROLLOUT_TP=2 REWARD_TP=2 CUDA_VISIBLE_DEVICES=0,1,2,3
export GENRM_OCR_TEMPERATURE=0 GENRM_OCR_TOP_P=1 GENRM_OCR_SEED=42
OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE/h100-results}
mkdir -p "$OUTPUT_DIR"
python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd
directory = Path(os.environ['WORKSPACE']) / 'data/ocr/qwen_image'
frame = pd.read_parquet(directory / 'test.parquet')
assert len(frame) >= 32 and {'prompt', 'reward_model'}.issubset(frame.columns)
frame.iloc[:32].to_parquet(directory / 'test32.parquet')
PY
algorithm=flow
recipe=flowdppo_trainer
responses=2
for mode in v0 v1; do
  suffix=
  if [ "$mode" = v1 ]; then suffix=_v1; fi
  cpu_count=32
  if [ "$algorithm" = flow ] && [ "$mode" = v0 ]; then cpu_count=16; fi
  label=$algorithm-$mode-full100
  ray_tmp=$(mktemp -d /tmp/vfg.XXXXXX)
  extra=(trainer.resume_mode=disable trainer.val_only=false)
  bash "examples/$recipe/qwen_image/run_qwen_image_ocr_lora$suffix.sh" \
   actor_rollout_ref.model.path="$MODEL_PATH" reward.reward_model.model_path="$REWARD_MODEL_PATH" \
   actor_rollout_ref.model.tokenizer_path="$MODEL_PATH/tokenizer" actor_rollout_ref.model.attn_backend=native \
   data.train_batch_size=4 data.val_batch_size=2 data.val_files="$WORKSPACE/data/ocr/qwen_image/test32.parquet" \
   actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.enable_timestep_staging=true actor_rollout_ref.actor.optim.lr=3e-5 actor_rollout_ref.rollout.n="$responses" \
   actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
   actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=10 \
   actor_rollout_ref.rollout.pipeline.height=512 actor_rollout_ref.rollout.pipeline.width=512 \
   actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
   actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.max_num_seqs=4 \
   reward.num_workers=1 reward.reward_model.enable_resource_pool=true reward.reward_model.rollout.free_cache_engine=false \
   reward.reward_model.n_gpus_per_node=2 reward.reward_model.nnodes=1 \
   reward.reward_model.rollout.max_model_len=8192 reward.reward_model.rollout.max_num_seqs=8 \
   reward.reward_model.rollout.gpu_memory_utilization=0.6 \
   reward.reward_model.rollout.temperature=0 reward.reward_model.rollout.do_sample=false \
   trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.agent.num_workers=1 \
   actor_rollout_ref.rollout.tensor_model_parallel_size=2 reward.reward_model.rollout.tensor_model_parallel_size=2 \
   ray_kwargs.ray_init.num_cpus="$cpu_count" \
   trainer.logger=console trainer.total_training_steps=100 trainer.total_epochs=100 \
   trainer.test_freq=20 trainer.save_freq=20 trainer.val_before_train=true \
   trainer.max_actor_ckpt_to_keep=1 actor_rollout_ref.actor.checkpoint.save_lora_only=true \
   trainer.experiment_name="$label" \
   trainer.default_local_dir="$OUTPUT_DIR/checkpoints/$label" \
   +ray_kwargs.ray_init.object_store_memory=2147483648 \
   +ray_kwargs.ray_init._temp_dir="$ray_tmp" "${extra[@]}" > "$OUTPUT_DIR/$label.log" 2>&1
done
