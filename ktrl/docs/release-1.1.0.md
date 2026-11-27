# KTRL 1.1.0

Release scope: Qwen3 online GRPO and RTX 4090 profiling (2026-11-27).

## Main changes

- **BF16 gradient accumulation on GPU.** Explicit BF16 shard buffers accumulate gradients across microbatches. Gradients are materialized in FP32 at the clipping/optimizer boundary. This does not mean Adam states or master weights are stored in BF16.
- **Reduce Actor GPU memory during rollout.** Preserve the Actor process and optimizer continuity, move its GPU training state to CPU while rollout runs, and restore it before training. This is what earlier notes called “Actor parking”. CPU routed experts remain on CPU; this optimization does not eliminate transfers of the GPU training state. The measured four-card rollout Actor footprint was approximately 0.62–0.64 GiB per card, not zero.
- **SynSQL integration.** Add SQL dataset handling and execution-based rewards to the online generation, reward, GRPO advantage, Actor update and weight-publication loop. SQL reward reliability and learning quality require separate evaluation.
- **RTX 4090 profiling.** Add profiling support and collect rollout throughput, latency, phase timings, GPU memory and container CPU memory for KTRL and reference runs. Compare equal GPU counts and explicitly label cold-start versus resident-Actor rounds. Fine-grained operator coverage, hardware counters and unexplained memory residuals are not all complete.

The release also includes Qwen3Moe support and a CPU expert weight-buffer ownership fix in the weight-publication path.

## Validation and limits

The existing Qwen3-30B-A3B four-card RTX 4090 D run completed six online batches: five had nonzero advantages and performed ten optimizer updates, publishing policy v10. Zero-advantage batches are recorded separately from effective training rounds. These observations establish online execution for the tested configuration; they do not establish convergence, universally correct SQL rewards or training/inference numerical parity.

The Linux source tree and an isolated copy of the locally prepared release each passed 85 offline regression tests. This release-note follow-up does not rerun GPU training or claim new measurements. Historical H20 results are separate evidence and must not be substituted for RTX 4090 results.

## Publication

The implementation was published in commit `80870da4a1301b54178a1cd1c9ed5700c442bd28`. This follow-up documents the four principal changes with the user-approved commit title and preserves that history. The local primary repository is the publication source; server worktrees are verified execution copies. See [the workflow](local-primary-workflow.md).
