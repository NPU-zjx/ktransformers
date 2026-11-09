#!/usr/bin/env bash
set -euo pipefail
mode=${1:?formal, torch, nsys or ncu}
attempt=${2:-01}
[[ $attempt =~ ^[0-9]+$ ]]
repo=/raid/data/zjx/kt-baseline/ktransformers
gpu=/raid/data/zjx/gpu-baseline
model=/mnt/qjh007/models/Qwen3.6-35B-A3B
suite=/raid/data/zjx/kt-baseline/results/gpu-verl-r01
out=$suite/$mode-$attempt
venv=$gpu/envs/main-cu130-py312
image=verl-ci-cn-beijing.cr.volces.com/verlai/verl@sha256:aaf5271484842d12a40de5e92ea0ce5beb3b478f6c7988e330baebb170d383c2
docker=(bash "$gpu/scripts/docker/client.sh")
prefix=()
extra=()
case $mode in
  formal|torch) ;;
  nsys)
    tool=/opt/nvidia/nsight-systems/2024.6.2
    extra=(-v "$tool:$tool:ro")
    prefix=("$tool/bin/nsys" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --wait=all --kill=none "--output=$out/rollout") ;;
  ncu)
    tool=/opt/nvidia/nsight-compute/2025.3.1
    extra=(--cap-add SYS_ADMIN -e GPU_NCU_APP_REPLAY=1)
    prefix=("$tool/ncu" --target-processes all --devices 0 --profile-from-start off --nvtx --nvtx-include kt_extend/ --kernel-name 'regex:^_fused_qk_rmsnorm_rope_gate_kernel' --launch-count 1 --graph-profiling node --replay-mode application --app-replay-match grid --clock-control none --cache-control none --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --section LaunchStats --export "$out/hotspot") ;;
  *) exit 2 ;;
esac
mkdir -p "$suite"
exec 9>"$suite/launch.lock"
flock -n 9
if [[ $mode == formal ]] && compgen -G "$suite/formal-*/ledger.json" >/dev/null; then echo 'Formal rollout already submitted'; exit 1; fi
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'GPU occupied'; exit 1; }
mkdir "$out"
mkdir -p "$out/home" "$out/analysis" "$suite/cache"
args=(
  --model "$model"
  --tp 8
  --context 9216
  --output-cap 8192
  --concurrency 32
  --chunk 1024
  --memory-fraction 0.35
  --seed 42
  --mode "$mode"
)
opts=(
  --name "zjx-gpu-verl-$mode-$attempt"
  --runtime=nvidia
  --workdir "$gpu/repos/verl-main"
  --shm-size 16g
  --memory 1024g
  -v "$gpu:$gpu:ro"
  -v "$model:$model:ro"
  -v "$repo/profiling:/bench:ro"
  -v "$out:$out"
  -v "$suite/cache:/cache"
  -e GPU_RUN="$out"
  -e HOME="$out/home"
  -e VIRTUAL_ENV="$venv"
  -e PATH="$venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONUNBUFFERED=1
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  -e OMP_NUM_THREADS=4
  -e MKL_NUM_THREADS=4
  -e HF_HUB_OFFLINE=1
  -e XDG_CACHE_HOME=/cache
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor
  -e TRITON_CACHE_DIR=/cache/triton
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer
)
cid=$("${docker[@]}" create "${opts[@]}" "${extra[@]}" "$image" "${prefix[@]}" /usr/bin/env "PYTHONPATH=/bench/gpu_reference:/bench:$gpu/repos/verl-main" "$venv/bin/python" /bench/gpu_reference/engine.py "${args[@]}")
printf '{"container_id":"%s"}\n' "$cid" > "$out/state.json"
cp "$out/state.json" "$out/launch.json"
"${docker[@]}" start "$cid"
python3 "$repo/profiling/resource_monitor.py" --run-dir "$out" --interval 2 >"$out/monitor.log" 2>&1 9>&- &
mon=$!
for i in {1..60}; do [[ -s "$out/telemetry/samples.jsonl" ]] && break; sleep 1; done
[[ -s "$out/telemetry/samples.jsonl" ]] && touch "$out/host-ready"
code=$("${docker[@]}" wait "$cid")
"${docker[@]}" logs "$cid" >"$out/container.log" 2>&1
printf '{"exit_code":%s}\n' "$code" > "$out/completion.json"
wait "$mon"
[[ $code == 0 ]]
if [[ $mode == ncu ]]; then
  [[ -s "$out/hotspot.ncu-rep" ]]
  for pass in "$out"/pass-*; do [[ -s "$pass/phase/summary.json" ]]; done
else
  [[ -s "$out/phase/summary.json" ]]
fi
