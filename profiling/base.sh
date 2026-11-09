#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Environment and model (qjh005)
DOCKER=/raid/data/zjx/gpu-baseline/scripts/docker/client.sh
IMAGE=verl-ci-cn-beijing.cr.volces.com/verlai/verl@sha256:aaf5271484842d12a40de5e92ea0ce5beb3b478f6c7988e330baebb170d383c2
JH=/mnt/qjh007/jiahengdai
JH_COPY=/raid/data/zjx/jiaheng-repro/reference/full-engineering/mnt/qjh007/jiahengdai
MODEL=/mnt/qjh007/models/Qwen3.6-35B-A3B
OUT=/raid/data/zjx/kt-baseline/results/base-r01
CUDNN=$JH/qwen36-native-smoke-20260930/sglang-kt-migration-20261102-p01/service-deps-cu12-cudnn916/nvidia/cudnn/lib
NAME=zjx-kt-base-r01
PORT=35439
# Sampling for the one formal rollout
MAX_NEW_TOKENS=8192
TEMPERATURE=1
TOP_P=1
TOP_K=-1
TORCH_DIAGNOSTIC=0
NSYS_ROOT=/opt/nvidia/nsight-systems/2024.6.2
tool_mount=()
tool_prefix=()
tool_env=()
if [[ ${1:-} == torch-* ]]; then
  OUT=/raid/data/zjx/kt-baseline/results/base-torch-r01
  NAME=zjx-kt-base-torch-r01
  TORCH_DIAGNOSTIC=1
fi
if [[ ${1:-} == nsys-* ]]; then
  OUT=/raid/data/zjx/kt-baseline/results/base-nsys-r01
  NAME=zjx-kt-base-nsys-r01
  TORCH_DIAGNOSTIC=1  # Share existing worker NVTX markers; request only CUDA_PROFILER.
  tool_mount=(-v "$NSYS_ROOT:$NSYS_ROOT:ro")
  tool_prefix=(
    "$NSYS_ROOT/bin/nsys" profile
    --trace=cuda,nvtx,osrt
    --sample=none
    --cpuctxsw=none
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --cuda-graph-trace=node
    --wait=all
    --kill=none
    --output=/output/rollout
  )
fi

if [[ ${KT_FULL_BASELINE:-0} == 1 ]]; then
  root=/raid/data/zjx/kt-baseline/results/base-all-r01
  case ${1:-} in
    torch-*) mode=torch ;;
    nsys-*) mode=nsys ;;
    ncu-*) mode=ncu ;;
    *) mode=rollout ;;
  esac
  OUT="$root/$mode"
  NAME="zjx-kt-base-all-$mode-r01"
  if [[ $mode == ncu ]]; then
    ncu_root=/opt/nvidia/nsight-compute/2025.1.1
    tool_mount=(--cap-add SYS_ADMIN -v "$ncu_root:$ncu_root:ro")
    tool_env=(-e PYTHONPATH= -e KT_PHASE_HOOKS=0 -e KT_TORCH_DIAGNOSTIC=0)
    tool_prefix=(
      "$ncu_root/ncu"
      --target-processes all
      --profile-from-start off
      --devices 0
      --nvtx
      --nvtx-include kt_extend/
      --kernel-name regex:FusedAddRMSNormKernel
      --launch-count 1
      --graph-profiling node
      --replay-mode kernel
      --section SpeedOfLight
      --section MemoryWorkloadAnalysis
      --section Occupancy
      --section WarpStateStats
      --section LaunchStats
      --export /output/hotspot
      /usr/bin/env
      KT_PHASE_HOOKS=1
      KT_TORCH_DIAGNOSTIC=1
      PYTHONPATH=/repo/profiling/worker:/repo/third_party/sglang/python
    )
  fi
fi

serve() {
  local active_pids
  active_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
  test -z "$active_pids" || { echo 'GPU in use'; return 1; }
  test -r "$MODEL/config.json"
  bash "$DOCKER" image inspect "$IMAGE" >/dev/null
  mkdir -p "$(dirname "$OUT")"
  mkdir "$OUT"
  mkdir "$OUT/home"
  bash "$DOCKER" run \
    -d \
    --name "$NAME" \
    --network host \
    --read-only \
    --user 0:0 \
    --runtime=nvidia \
    --ipc private \
    --shm-size 16g \
    --memory 1024g \
    --memory-swap 1024g \
    --tmpfs /tmp:rw,size=16g \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e OMP_NUM_THREADS=4 \
    -e MKL_NUM_THREADS=4 \
    -e HOME=/output/home \
    -e HF_HUB_OFFLINE=1 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONUNBUFFERED=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e LD_LIBRARY_PATH="$CUDNN" \
    -e FLASHINFER_WORKSPACE_BASE=/output/cache/flashinfer \
    -e TORCHINDUCTOR_CACHE_DIR=/output/cache/inductor \
    -e KT_PHASE_HOOKS=1 \
    -e KT_TORCH_DIAGNOSTIC="$TORCH_DIAGNOSTIC" \
    -e SGLANG_PROFILE_V2=0 \
    -e PYTHONPATH=/repo/profiling/worker:/repo/third_party/sglang/python \
    "${tool_env[@]}" \
    -v "$PWD:/repo:ro" \
    -v "$JH_COPY:$JH:ro" \
    -v "$MODEL:$MODEL:ro" \
    -v "$OUT:/output:rw" \
    "${tool_mount[@]}" \
    -w /repo \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -c 'exec "$@" > /output/server.log 2>&1' \
    -- \
    "${tool_prefix[@]}" \
    "$JH/venvs/kt-py31115/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" \
    --kt-weight-path "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tp-size 8 \
    --dtype bfloat16 \
    --context-length 9216 \
    --kt-method BF16 \
    --kt-cpuinfer 32 \
    --kt-threadpool-count 2 \
    --kt-num-gpu-experts 0 \
    --mem-fraction-static 0.35 \
    --max-running-requests 32 \
    --chunked-prefill-size 1024 \
    --cuda-graph-max-bs 32 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --random-seed 42 \
    > "$OUT/container.id"
  cat "$OUT/container.id"
}

rollout() {
  python3 -B profiling/client.py \
    --run-dir "$OUT" \
    --container "$NAME" \
    --url "http://127.0.0.1:$PORT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K"
}

stop() {
  local cid
  cid=$(cat "$OUT/container.id")
  [[ "$cid" =~ ^[a-f0-9]{64}$ ]] || { echo "Invalid container ID" >&2; return 1; }
  [[ "$(bash "$DOCKER" inspect --format '{{.Name}}' "$cid")" == "/$NAME" ]] || return 1
  bash "$DOCKER" stop --time 60 "$cid"
}

case "${1:-}" in
  serve) serve ;;
  rollout) rollout ;;
  stop) stop ;;
  torch-serve) serve ;;
  torch-profile)
    python3 -B profiling/torch_profile.py \
      --run-dir "$OUT" \
      --container "$NAME" \
      --url "http://127.0.0.1:$PORT"
    ;;
  torch-stop) stop ;;
  nsys-serve) test -x "$NSYS_ROOT/bin/nsys" && serve ;;
  nsys-profile)
    python3 -B profiling/nsys_profile.py \
      --run-dir "$OUT" \
      --container "$NAME" \
      --url "http://127.0.0.1:$PORT"
    "$NSYS_ROOT/bin/nsys" export \
      --type sqlite \
      --output "$OUT/rollout.sqlite" \
      "$OUT/rollout.nsys-rep"
    python3 -B profiling/analysis/analyze_nsys.py "$OUT/rollout.sqlite"
    ;;
  nsys-stop) stop ;;
  ncu-serve) [[ ${KT_FULL_BASELINE:-0} == 1 ]] && serve ;;
  ncu-stop) [[ ${KT_FULL_BASELINE:-0} == 1 ]] && stop ;;
  *-group-profile)
    [[ ${KT_FULL_BASELINE:-0} == 1 ]] || exit 2
    python3 -B profiling/group_profile.py \
      --mode "$mode" \
      --run-dir "$OUT" \
      --container "$NAME" \
      --url "http://127.0.0.1:$PORT"
    if [[ $mode == nsys ]]; then
      "$NSYS_ROOT/bin/nsys" export --type sqlite --output "$OUT/rollout.sqlite" "$OUT/rollout.nsys-rep"
      python3 -B profiling/analysis/analyze_nsys.py "$OUT/rollout.sqlite"
    fi
    if [[ $mode == ncu ]]; then
      "$ncu_root/ncu" --import "$OUT/hotspot.ncu-rep" --csv --page raw > "$OUT/raw-metrics.csv"
      "$ncu_root/ncu" --import "$OUT/hotspot.ncu-rep" --page details > "$OUT/details.txt"
      python3 -B profiling/group_profile.py --validate-ncu "$OUT"
    fi
    ;;
  *) echo "Unknown action: ${1:-}" >&2; exit 2 ;;
esac
