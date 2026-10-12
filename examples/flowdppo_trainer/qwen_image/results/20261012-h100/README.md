# Flow-DPPO V0/V1 H100 validation

Last updated: 10/12/2026

Both modes completed 100 distinct updates, with finite logged metrics, validation at
0/20/40/60/80/100, and a step-100 checkpoint. Both LoRA model shards in each mode
contain finite tensors and nonzero trained LoRA B values.

V0 completed all 100 updates, step-100 validation and checkpoint saving before its own process monitor misclassified exiting workers and canceled the Slurm step (143). The original exit is preserved. Loading its original step-100 checkpoint and running validation only on the same 32 prompts exited 0 (reward 0.8176391032757238), with no extra training. V1 training exited 0. V0 is a qualified recovery pass.


| Step | V0 OCR reward | V1 sync OCR reward |
| ---: | ---: | ---: |
| 0 | 0.835786 | 0.814357 |
| 20 | 0.777085 | 0.825861 |
| 40 | 0.816461 | 0.858843 |
| 60 | 0.866351 | 0.834163 |
| 80 | 0.836867 | 0.830012 |
| 100 | 0.830533 | 0.812736 |

The experiment uses four H100 GPUs (65247 MiB each): two for actor/rollout and two
for Qwen3-VL OCR reward, with tensor parallelism 2. FSDP2 uses timestep staging,
parameter/optimizer CPU offload, BF16 LoRA rank 64 / alpha 128, learning rate `3e-5`,
train batch 4 and 2 images per prompt. Images are 512×512 with 10 train/evaluation
denoising steps, CFG 4, SDE noise 1.2, window 2 over [0,5]. The same first 32 test
prompts are evaluated in batches of 2; checkpoints and validation run every 20 steps.
Flow V0 used 16 Ray CPUs; V1 used 32. V1 needs enough CPUs for TransferQueue plus actor/reward pools.
Reward `free_cache_engine=false`; actor/rollout and reward use native/TORCH_SDPA attention.

The runtime baseline is `70bb551bf1de17f4a2ec0e5b17bb0be1b05e18b1`, plus the two V1 recipe overlays.
Flow uses the unmodified frozen runtime.
The new documentation and regression tests were not part of the original H100 source snapshot.
Model revisions are Qwen-Image `75e0b4be04f60ec59a75f475837eced720f823b6`
and Qwen3-VL-8B-Instruct `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
Python 3.12.15, PyTorch 2.13.0+cu130, vLLM 0.28.0, vLLM-Omni
`ded8934626aaad1a3e816c3a1d9d742efc012d93`, verl
`fefb080262e1c015a0ea05f958822a6a512dc795`, Transformers 5.14.1,
Diffusers 0.40.0, Ray 2.59.0 and TransferQueue 0.1.9 were used.

The original runtime passed 79 CPU cases for diffusion core algorithms, registry, V1 parameter sync and checkpoint recovery.

The command used for that CPU run was:

```bash
python -m pytest \
  tests/trainer/diffusion/test_diffusion_core_algos_on_cpu.py \
  tests/trainer/diffusion/test_diffusion_registry_on_cpu.py \
  tests/trainer/diffusion/test_v1_parameter_sync_on_cpu.py \
  tests/trainer/diffusion/test_v1_checkpoint_recovery_on_cpu.py -q
```


The [portable launch script](run_100_steps.sh) carries the recorded Hydra overrides
with caller-supplied model, dataset and output paths. Run it from the repository root
after preparing the models, dataset and pinned runtime:

```bash
bash examples/flowdppo_trainer/qwen_image/results/20261012-h100/run_100_steps.sh
```

These are runtime checks for this configuration. One run per mode and six validation
means do not establish convergence, statistical equivalence, speedup, or behavior at
the recipes' default 50-step denoising schedule. Historical 5090 measurements remain
separate. Full logs/checkpoints have verified private backups; no model weights,
images or private infrastructure details are included here. AI assistance was used
for the changes and validation tooling.
