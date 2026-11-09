#!/usr/bin/env bash
# Sample system GPU metrics in the active Actor phase after its Torch window.
set -euo pipefail
root=${1:?usage: sample_actor_nsys.sh RUN_ROOT}
controller=${2:?controller PID required}
run="$root/rl-online-profile01"
profiles="$root/profiles"
client=/raid/data/zjx/gpu-baseline/scripts/docker/client.sh
nsys=/opt/nvidia/nsight-systems/2024.6.2
image=verl-ci-cn-beijing.cr.volces.com/verlai/verl@sha256:aaf5271484842d12a40de5e92ea0ce5beb3b478f6c7988e330baebb170d383c2
name="zjx-ktrl-actor-metrics-$controller"
report="$profiles/nsys-actor-device0.nsys-rep"

for (( attempt=0; attempt<540; attempt++ )); do
    kill -0 "$controller" 2>/dev/null || exit 3
    if grep -qs 'actor-microbatch-rank0' "$profiles"/windows-*.jsonl; then
        grep -q '"phase": "actor_update"' "$run/round-01/current-phase.json" || exit 4
        test ! -e "$report"
        date -Ins > "$profiles/nsys-actor-device0.start"
        bash "$client" run --name "$name" --pull=never --network none --gpus device=0 \
            --cap-add SYS_ADMIN --entrypoint "$nsys/bin/nsys" -v "$nsys:$nsys:ro" "$image" \
            profile --trace=none --sample=none --cpuctxsw=none --gpu-metrics-devices=0 \
            --gpu-metrics-frequency=1000 --output=/tmp/metrics /bin/sleep 10
        bash "$client" cp "$name:/tmp/metrics.nsys-rep" "$report"
        test -s "$report"
        bash "$client" rm "$name"
        date -Ins > "$profiles/nsys-actor-device0.end"
        exit 0
    fi
    sleep 10
done
exit 124
