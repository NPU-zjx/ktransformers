#!/usr/bin/env bash
# qjh005-only, with the official wheel overlay verified in checks/ktrl-m2-r01.
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:?choose prepare, layer or update}
tag=${2:?supply a fresh attempt tag}
[[ $tag =~ ^[a-z0-9-]+$ ]] || exit 2
root=/raid/data/zjx/kt-baseline
check=$root/checks/ktrl-m2-r01
out=$root/results/ktrl-m2-r01
model=/mnt/qjh007/models/Qwen3.6-35B-A3B
jh=/mnt/qjh007/jiahengdai
copy=/raid/data/zjx/jiaheng-repro/reference/full-engineering$jh
data=/raid/data/zjx/gpu-baseline/data/full20k-20261025
roll=/raid/data/zjx/gpu-baseline/runs/full20k-20261025-r02-formal/rollouts/1.jsonl
example=$jh/qwen36-native-smoke-20260930/kt-sft-actor-20261104-a02/fixed-batch.json
image=verl-ci-cn-beijing.cr.volces.com/verlai/verl@sha256:aaf5271484842d12a40de5e92ea0ce5beb3b478f6c7988e330baebb170d383c2
python=$jh/venvs/kt-py31115/bin/python
work=$out/$stage-$tag
case $stage in
  prepare)
    devices=none
    args=(-m ktrl.trainer.fixed_batch --model "$model" --parquet "$data/train-13376.parquet"
          --rollout "$roll" --reference "$example" --out "/out/$stage-$tag")
    ;;
  layer)
    devices=0,1
    args=(-m torch.distributed.run --standalone --nproc_per_node=2 --module ktrl.workers.engine.fsdp.layer_check
          --out "/out/$stage-$tag")
    ;;
  update)
    devices=0,1,2,3,4,5,6,7
    test -e "$out/prepare-${3:?supply accepted prepare tag}/batch-audit.json"
    test -e "$out/layer-${4:?supply accepted layer tag}/accepted.json"
    test ! -e "$out/optimizer-started.json" || { echo 'An update already reached optimizer.step' >&2; exit 2; }
    args=(-m torch.distributed.run --standalone --nproc_per_node=8 --module ktrl.workers.engine.fsdp.actor_update
          --config /repo/examples/ktrl/qwen36_35b_grpo.yaml --batch-dir "/out/prepare-$3" --out "/out/$stage-$tag")
    ;;
  restore)
    devices=0,1,2,3,4,5,6,7
    completed=/out/update-${4:?supply completed update tag}
    test -f "$out/update-$4/accepted.json"
    args=(-m torch.distributed.run --standalone --nproc_per_node=8 --module ktrl.workers.engine.fsdp.actor_update
          --config /repo/examples/ktrl/qwen36_35b_grpo.yaml --batch-dir "/out/prepare-${3:?supply prepare tag}"
          --restore-from "$completed/trainer/checkpoint-1" --out "/out/$stage-$tag")
    ;;
  *) exit 2 ;;
esac
test -r "$check/site/kt_kernel/sft/lora.py"
mkdir "$work"
if [[ $devices != none ]]; then
  active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z $active ]] || { echo "GPU occupied: $active" >&2; exit 1; }
fi
docker=(bash /raid/data/zjx/gpu-baseline/scripts/docker/client.sh)
name=zjx-ktrl-m2-$stage-$tag
"${docker[@]}" run --name "$name" --cidfile "$work/container.id" --runtime=nvidia --network host \
  --read-only --ipc private --shm-size 32g --memory 1550g --memory-swap 1550g --tmpfs /tmp:rw,exec,size=24g \
  -e NVIDIA_VISIBLE_DEVICES="$devices" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e KTRL_FORWARD_TRACE="${KTRL_FORWARD_TRACE:-}" \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/out/cache/home -e XDG_CACHE_HOME=/out/cache/xdg -e HF_HUB_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false -e WANDB_DISABLED=true -e KT_SFT_PROFILE=1 \
  -e PYTHONPATH=/env/site:/env/extras:/repo -e LD_LIBRARY_PATH="$jh/qwen36-native-smoke-20260930/sglang-kt-migration-20261102-p01/service-deps-cu12-cudnn916/nvidia/cudnn/lib" \
  -e TRITON_CACHE_DIR=/out/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/out/cache/inductor \
  -v "$PWD:/repo:ro" -v "$check:/env:ro" -v "$out:/out:rw" -v "$copy:$jh:ro" \
  -v "$model:$model:ro" -v "$data:$data:ro" -v "$roll:$roll:ro" \
  -w /repo --entrypoint "$python" "$image" -B "${args[@]}"
