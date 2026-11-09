#!/usr/bin/env bash
# Run INSIDE the dedicated profiling container with performance-counter access.
set -euo pipefail
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo 'usage: run_ncu.sh SOURCE RECIPE NEW_OUTPUT_ROOT [actor]' >&2
  exit 2
fi
source_dir=$(realpath "$1")
recipe=$(realpath "$2")
output_root=$(realpath -m "$3")
scope=${4:-all}
roles=(rollout actor)
launch_scope=()
case "$scope" in
  all) ;;
  actor) roles=(actor); launch_scope=(--ncu-actor-only) ;;
  *) exit 2 ;;
esac
case "$output_root" in
  /raid/data/zjx/kt-baseline/results/*) ;;
  *) exit 2 ;;
esac
test ! -e "$output_root/rl-online-profile01"
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
python_bin=/mnt/qjh007/jiahengdai/venvs/kt-py31115/bin/python
ncu_bin=/raid/data/zjx/kt-baseline/tools/ncu-2025.4.1-isolated/unpacked/opt/nvidia/nsight-compute/2025.4.1/ncu
"$python_bin" -B "$script_dir/launch.py" \
  --source "$source_dir" \
  --recipe "$recipe" \
  --root "$output_root" \
  --mode ncu \
  --check
mkdir -p "$output_root/ncu"
"$ncu_bin" --version > "$output_root/ncu/version.txt"
# Bounded norm kernels, GPU0 only, first observed batch/microbatch per stage.
# Kernel replay repeats selected GPU kernels, never the Python RL update.
ncu_options=( \
  --target-processes all \
  --devices 0 \
  --profile-from-start off \
  --replay-mode kernel \
  --graph-profiling node \
  --clock-control none \
  --cache-control none \
  --nvtx \
  --nvtx-push-pop-scope process \
  --nvtx-include 'regex:rollout-(extend|decode)-tp0/' \
  --nvtx-include 'regex:actor_(actor_forward_start|backward_start)_rank0/' \
  --kernel-name 'regex:(RMSNormKernel|layer_norm_(gated_)?(fwd|bwd)_kernel)' \
  --kernel-name-base function \
  --launch-count 400 \
  --set full \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active \
)
"$ncu_bin" "${ncu_options[@]}" \
  --export "$output_root/ncu/smoke" \
  "$python_bin" -B "$script_dir/ncu_smoke.py" "$source_dir" "$output_root/ncu"
"$ncu_bin" --import "$output_root/ncu/smoke.ncu-rep" --page raw --csv > "$output_root/ncu/smoke.csv"
"$python_bin" -B "$script_dir/validate_ncu_smoke.py" "$output_root/ncu"
mkdir "$output_root/overlay"
cp "$script_dir/launch.py" "$output_root/overlay/launch.py"
cp "$script_dir/ncu_sitecustomize.py" "$output_root/overlay/sitecustomize.py"
cp "$script_dir/cpu_observer.py" "$output_root/overlay/cpu_observer.py"
sha256sum "$output_root/overlay/"*.py > "$output_root/overlay/SHA256.txt"
"$python_bin" -B "$script_dir/validate_overlay_imports.py" "$source_dir" "$output_root"
"$python_bin" - "$output_root/ncu/options.json" "${ncu_options[@]}" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:], indent=2))
PY
printf '%s\n' "$ncu_bin" > "$output_root/ncu/binary.txt"
"$python_bin" -B -u "$output_root/overlay/launch.py" \
  --source "$source_dir" \
  --recipe "$recipe" \
  --root "$output_root" \
  --mode ncu \
  --ncu-roles "${launch_scope[@]}"
for role in "${roles[@]}"; do
  "$ncu_bin" --import "$output_root/ncu/$role-core.ncu-rep" --page raw --csv > "$output_root/ncu/$role-core.csv"
  if [ "$role" = rollout ]; then
    stages=(prefill decode)
  else
    stages=(forward backward)
  fi
  "$python_bin" -B "$script_dir/analyze_ncu_online.py" \
    "$output_root/ncu/$role-core.csv" "$output_root/ncu/$role-validated.json" \
    --required-stages "${stages[@]}"
done
"$python_bin" - "$output_root/ncu" "${roles[@]}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
roles={role:json.loads((root/(role+'-validated.json')).read_text()) for role in sys.argv[2:]}
assert all(r['verified'] for r in roles.values())
(root/'validated.json').write_text(json.dumps(dict(
    verified=set(roles)=={'rollout','actor'}, actor_counters_verified=True,
    missing_roles=sorted({'rollout','actor'}-roles.keys()), roles=roles,
    scope='Complete online controller; full metric set on the listed roles'),indent=2))
PY
