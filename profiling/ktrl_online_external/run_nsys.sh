#!/usr/bin/env bash
# Full native online cycle; bounded CUDA profiling windows, no source changes.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo 'usage: run_nsys.sh SOURCE RECIPE NEW_OUTPUT_ROOT' >&2
  exit 2
fi
source_dir=$(realpath "$1")
recipe=$(realpath "$2")
output_root=$(realpath -m "$3")
case "$output_root" in
  /raid/data/zjx/kt-baseline/results/*) ;;
  *) echo 'new diagnostics must write to /raid results' >&2; exit 2 ;;
esac
test ! -e "$output_root"
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
python_bin=/mnt/qjh007/jiahengdai/venvs/kt-py31115/bin/python
nsys_bin=/opt/nvidia/nsight-systems/2024.6.2/bin/nsys
"$python_bin" -B "$script_dir/launch.py" \
  --source "$source_dir" \
  --recipe "$recipe" \
  --root "$output_root" \
  --mode nsys \
  --check
mkdir -p "$output_root/nsys"
exec "$nsys_bin" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat:8 \
  --kill=none \
  --wait=all \
  --output="$output_root/nsys/online" \
  "$python_bin" -B -u "$script_dir/launch.py" \
  --source "$source_dir" \
  --recipe "$recipe" \
  --root "$output_root" \
  --mode nsys
