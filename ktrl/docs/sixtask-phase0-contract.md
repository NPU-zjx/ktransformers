# SixTask Phase 0 Frozen Contract (`KTRL1.6.0`)

Frozen on 2026-11-27 (UTC+8) on base commit `f1e4b2bbe2c74948125aba21bc9e5d0b9212e82c` (`kt-rl`). No runtime source code was modified in Phase 0.

## 1. Verified 3-Node Baseline & Model Equivalence

- **Model**: `Qwen3-30B-A3B` (`Qwen3MoeForCausalLM`, BF16, 48 layers, 32 attn heads, 4 KV heads, `head_dim=128`, 128 experts top-8, `hidden_size=2048`, `moe_intermediate_size=768`).
- Verified identical across `sapphire2`, `sapphire1`, and `qjh005` (`/mnt/qjh007/models/Qwen3-30B-A3B`):
  - `tokenizer.json`: `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
  - `model.safetensors.index.json`: `df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24`
  - `model-00001-of-00016.safetensors`: `454e77b346a61bfb201d54df60e15158838cf930617ee135113556204f2802b5`
- **Hardware Defaults**:
  - `sapphire2`: 4 × RTX 4090 D (24 GiB) — profiling, optimization, and 4-card gate validation.
  - `sapphire1`: 8 × RTX 4090 (24 GiB) — 8-card formal regression after 4-card gate passes.
  - `qjh005`: 8 × NVIDIA H20 (96 GiB, NV18 NVLink) — pure GPU no-offload veRL FSDP2 & Megatron baselines (`verl 0.10.0.dev0`, `vllm 0.23.0`, `megatron-core 0.16.1`, `mbridge 0.15.1`).

## 2. Unified KV Cache Contract (Zero-Byte Drift Verified Live)

- Per-token KV footprint: `96 KiB/token` across all 4 KV heads (`24 KiB/token/GPU` under TP4, TP8, or TP4×DP2).
- **Frozen Target**: **`3.000000 GiB/GPU`** (`3,221,225,472` bytes) = **`131,072` logical tokens** (`16.00×` 8,192-token context, `8,192` vLLM blocks of 16).
  - **SGLang (4×4090 D `a02` live verified)**: `--max-total-tokens 131072 --mem-fraction-static 0.35` → allocated `131,072` tokens (`1.50 GiB K + 1.50 GiB V = 3.00 GiB/GPU`, `12.00 GiB` total TP4), rollout NVML `5.08–5.38 GiB/GPU` (down ~4.4 GiB/GPU from `9.55–9.83 GiB/GPU`).
  - **vLLM 0.23.0 (8×H20 `a03` live verified)**: `kv_cache_memory_bytes=3221225472` → allocated `8,192` blocks × `16` = `131,072` tokens (`3.00 GiB/GPU`, `24.00 GiB` physical across 8 GPUs with 2× KV-head replication), NVML `13.03 GiB/GPU`.

## 3. Rollout Concurrency & Short-Probe SQL Dataset

- **Frozen Rollout Concurrency**: **`concurrency=4`** with `prompt_grouped` (1 prompt × 4 answers) submission.
  - Live SynSQL probe (`sapphire2` 4×4090 D, `max_new_tokens=2048`): on the identical 2-prompt × 4-answer (8 req) set, `concurrency=4` cut wall clock from `321.71 s` (`c=1`) to **`177.33 s`** (1.81× faster), boosted overall TPS from `43.15` to **`73.90 tok/s`** (+71.3%) and Extend TPS from `580.2` to **`2,415.0 tok/s`**, while keeping TTFT P50/P95 at `696 / 1,026 ms` (16-req case: `126 / 1,120 ms`, max `1,181 ms`) and TPOT P50 at `50.1–51.4 ms/tok` (vs `8.78–10.15 s` TTFT and `141 ms` TPOT at `concurrency=32`), with `75%` prefix cache hit.
- **Frozen 24-Prompt Non-Zero Advantage Short-Probe Queue**: built from `synsql_qwen3_v1/train.parquet` on `sapphire2` (`/mnt/data/zjx/ktrl-fifth/checks/sixtask-phase0-20261127/`): 14 proven non-zero advantage prompts from FifthTask formal runs (3 double-proven in both KTRL and GPU veRL: `783930`, `687152`, `1012306`) + 10 distinct-`db_id` `Highly Complex` backup prompts.

`SIXTASK_PHASE0_PASS kv_total_gib=12.0 kv_per_gpu_gib=3.0 concurrency=4 nonzero_adv_prompts=14`
