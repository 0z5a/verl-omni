# GRPO-Guard V0/V1 H100 validation

Last updated: 10/12/2026

Both modes completed 100 distinct updates, with finite logged metrics, validation at
0/20/40/60/80/100, and a step-100 checkpoint. Both LoRA model shards in each mode
contain finite tensors and nonzero trained LoRA B values.

Both final runs started from pretrained weights and exited 0. An earlier unpatched V0 attempt stopped after 19 updates because its singleton `actor/ratio_std` was NaN; that failed attempt is preserved and excluded from the 100-step results. The final runs use `ratio.std(unbiased=False)` only in the Guard diagnostic under `torch.no_grad()`.


| Step | V0 OCR reward | V1 sync OCR reward |
| ---: | ---: | ---: |
| 0 | 0.838428 | 0.812844 |
| 20 | 0.835081 | 0.849931 |
| 40 | 0.857558 | 0.841119 |
| 60 | 0.839344 | 0.856016 |
| 80 | 0.887079 | 0.893772 |
| 100 | 0.882561 | 0.843319 |

The experiment uses four H100 GPUs (65247 MiB each): two for actor/rollout and two
for Qwen3-VL OCR reward, with tensor parallelism 2. FSDP2 uses timestep staging,
parameter/optimizer CPU offload, BF16 LoRA rank 64 / alpha 128, learning rate `3e-5`,
train batch 4 and 4 images per prompt. Images are 512×512 with 10 train/evaluation
denoising steps, CFG 4, SDE noise 1.2, window 2 over [0,5]. The same first 32 test
prompts are evaluated in batches of 2; checkpoints and validation run every 20 steps.
Both Guard modes used 32 Ray CPUs. V1 needs enough CPUs for TransferQueue plus actor/reward pools.
Reward `free_cache_engine=false`; actor/rollout and reward use native/TORCH_SDPA attention.

The runtime baseline is `70bb551bf1de17f4a2ec0e5b17bb0be1b05e18b1`, plus the two V1 recipe overlays.
Guard adds only the population-std diagnostic fix; the loss and gradient computations are unchanged.
The new documentation and regression tests were not part of the original H100 source snapshot.
Model revisions are Qwen-Image `75e0b4be04f60ec59a75f475837eced720f823b6`
and Qwen3-VL-8B-Instruct `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
Python 3.12.15, PyTorch 2.13.0+cu130, vLLM 0.28.0, vLLM-Omni
`ded8934626aaad1a3e816c3a1d9d742efc012d93`, verl
`fefb080262e1c015a0ea05f958822a6a512dc795`, Transformers 5.14.1,
Diffusers 0.40.0, Ray 2.59.0 and TransferQueue 0.1.9 were used.

The patched runtime passed 31 existing diffusion-core CPU cases. Twelve FP32/BF16 probes (batch 1/2/8, with/without importance weights) gave 36 bitwise-equal loss and gradient comparisons. The new regression cases were also executed with the exact production Guard class isolated from trainer imports and real CPU PyTorch 2.14.1: 12 expected failures before the fix, 12 passes after it, and another 36 exact loss/gradient comparisons. This isolated check does not replace a full trainer-import test run.

The existing core run used the following command before the 12 new cases were added:

```bash
python -m pytest tests/trainer/diffusion/test_diffusion_core_algos_on_cpu.py -q
```

The [portable launch script](run_100_steps.sh) carries the recorded Hydra overrides
with caller-supplied model, dataset and output paths. Run it from the repository root
after preparing the models, dataset and pinned runtime:

```bash
bash examples/grpoguard_trainer/qwen_image/results/20261012-h100/run_100_steps.sh
```

These are runtime checks for this configuration. One run per mode and six validation
means do not establish convergence, statistical equivalence, speedup, or behavior at
the recipes' default 50-step denoising schedule. Historical 5090 measurements remain
separate. Full logs/checkpoints have verified private backups; no model weights,
images or private infrastructure details are included here. AI assistance was used
for the changes and validation tooling.
